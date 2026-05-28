use std::collections::HashMap;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use clap::Parser;
use diloco_core::{params, CharTokenizer, Config, GptModel, OuterOptimizer};
use diloco_net::{
    AllReduceRequest, Diloco, DilocoServer, InitRequest, ParamMsg, MAX_MESSAGE_SIZE,
};
use std::path::PathBuf;
use tokio::sync::{oneshot, Mutex};
use tonic::{transport::Server, Request, Response, Status};
use tracing::info;

#[derive(Parser)]
#[command(about = "DiLoCo coordinator (outer optimizer)")]
struct Args {
    /// Address to listen on for worker connections.
    #[arg(long, default_value = "127.0.0.1:7000")]
    listen: String,
    /// Number of workers to wait for each round.
    #[arg(long)]
    world_size: usize,
    /// Corpus used only to derive the vocabulary (must match the workers').
    #[arg(long, default_value = "data/input.txt")]
    corpus: PathBuf,
    /// Outer-optimizer learning rate.
    #[arg(long, default_value_t = 0.7)]
    outer_lr: f64,
    /// Outer-optimizer Nesterov momentum.
    #[arg(long, default_value_t = 0.9)]
    outer_momentum: f64,
}

/// Mutable state guarded by a single async mutex. Round handling is fully
/// serialized, which is fine: rounds are infrequent and the work per round is a
/// handful of tensor ops.
struct Inner {
    global: HashMap<String, Tensor>,
    outer: OuterOptimizer,
    /// Round currently being collected (1-based).
    round: u64,
    /// Local parameter sets submitted so far this round.
    pending: Vec<HashMap<String, Tensor>>,
    /// One sender per worker blocked in `all_reduce`, fired when the round closes.
    waiters: Vec<oneshot::Sender<Vec<u8>>>,
}

struct CoordinatorService {
    world_size: usize,
    device: Device,
    /// Serialized current global params, served by `Init`.
    init_bytes: Vec<u8>,
    inner: Mutex<Inner>,
}

#[tonic::async_trait]
impl Diloco for CoordinatorService {
    async fn init(&self, request: Request<InitRequest>) -> Result<Response<ParamMsg>, Status> {
        let rank = request.into_inner().rank;
        info!(rank, "worker fetched initial parameters");
        Ok(Response::new(ParamMsg {
            params: self.init_bytes.clone(),
        }))
    }

    async fn all_reduce(
        &self,
        request: Request<AllReduceRequest>,
    ) -> Result<Response<ParamMsg>, Status> {
        let req = request.into_inner();
        let local = params::deserialize(&req.params, &self.device)
            .map_err(|e| Status::invalid_argument(format!("bad params payload: {e}")))?;

        let rx = {
            let mut st = self.inner.lock().await;
            if req.round != st.round {
                return Err(Status::failed_precondition(format!(
                    "worker {} sent round {} but coordinator is on round {}",
                    req.rank, req.round, st.round
                )));
            }

            let (tx, rx) = oneshot::channel();
            st.pending.push(local);
            st.waiters.push(tx);
            info!(
                round = st.round,
                rank = req.rank,
                submitted = st.pending.len(),
                world_size = self.world_size,
                "received local parameters"
            );

            // Last worker in: close the round and broadcast the new global.
            if st.pending.len() == self.world_size {
                let pending = std::mem::take(&mut st.pending);
                let waiters = std::mem::take(&mut st.waiters);
                let st = &mut *st; // split borrow of global + outer
                st.outer
                    .step(&mut st.global, &pending)
                    .map_err(|e| Status::internal(format!("outer step failed: {e}")))?;
                let bytes = params::serialize(&st.global)
                    .map_err(|e| Status::internal(format!("serialize failed: {e}")))?;
                st.round += 1;
                info!(round = st.round - 1, "round complete, broadcasting new global");
                for waiter in waiters {
                    let _ = waiter.send(bytes.clone());
                }
            }
            rx
        };

        let params = rx
            .await
            .map_err(|_| Status::internal("round was canceled"))?;
        Ok(Response::new(ParamMsg { params }))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let args = Args::parse();
    let device = Device::Cpu;

    // Build the initial global parameters: construct the model once with random
    // init, then snapshot its VarMap. The coordinator is the single source of
    // truth for theta^(0), which it broadcasts to workers via Init.
    let text = std::fs::read_to_string(&args.corpus)
        .with_context(|| format!("reading corpus from {}", args.corpus.display()))?;
    let tokenizer = CharTokenizer::from_text(&text);
    let cfg = Config::tiny(tokenizer.vocab_size());

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let _model = GptModel::new(&cfg, vb)?;
    let global = params::varmap_tensors(&varmap);
    let init_bytes = params::serialize(&global)?;
    let outer = OuterOptimizer::new(&global, args.outer_lr, args.outer_momentum)?;

    info!(
        world_size = args.world_size,
        vocab_size = cfg.vocab_size,
        outer_lr = args.outer_lr,
        outer_momentum = args.outer_momentum,
        listen = %args.listen,
        "coordinator ready"
    );

    let service = CoordinatorService {
        world_size: args.world_size,
        device,
        init_bytes,
        inner: Mutex::new(Inner {
            global,
            outer,
            round: 1,
            pending: Vec::new(),
            waiters: Vec::new(),
        }),
    };

    let server = DilocoServer::new(service)
        .max_decoding_message_size(MAX_MESSAGE_SIZE)
        .max_encoding_message_size(MAX_MESSAGE_SIZE);

    Server::builder()
        .add_service(server)
        .serve(args.listen.parse()?)
        .await?;
    Ok(())
}
