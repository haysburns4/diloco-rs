use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use clap::Parser;
use diloco_core::{params, CharTokenizer, Config, GptModel, OuterOptimizer};
use diloco_net::{
    AllReduceRequest, Diloco, DilocoServer, InitRequest, ParamMsg, MAX_MESSAGE_SIZE,
};
use tokio::sync::{oneshot, Mutex};
use tonic::{transport::Server, Request, Response, Status};
use tracing::{info, warn};

#[derive(Parser)]
#[command(about = "DiLoCo coordinator (outer optimizer)")]
struct Args {
    /// Address to listen on for worker connections. Defaults to port 7070
    #[arg(long, default_value = "127.0.0.1:7070")]
    listen: String,
    /// Number of workers expected each round.
    #[arg(long)]
    world_size: usize,
    #[arg(long, default_value = "data/input.txt")]
    corpus: PathBuf,
    #[arg(long, default_value_t = 0.7)]
    outer_lr: f64,
    #[arg(long, default_value_t = 0.9)]
    outer_momentum: f64,
    #[arg(long)]
    init: Option<PathBuf>,
    /// Seconds after the first submission before the coordinator closes a
    /// round even if fewer than world_size workers have submitted. With the
    /// default of world_size for min_workers this is effectively a stuck-round
    /// detector: the coordinator logs a warning but does not force-close.
    #[arg(long, default_value_t = 60)]
    round_timeout_secs: u64,
    /// Minimum number of worker submissions required for a timeout-triggered
    /// close. Defaults to world_size, meaning the timeout only fires a warning
    /// unless you explicitly lower this. Set to ceil(world_size/2) for
    /// majority-vote semantics.
    #[arg(long)]
    min_workers: Option<usize>,
}

struct Inner {
    global: HashMap<String, Tensor>,
    outer: OuterOptimizer,
    /// The round currently being collected (1-based). Workers must submit this
    /// round number or receive FailedPrecondition (and call Init to resync).
    round: u64,
    /// Parameter sets submitted so far this round, one per worker.
    pending: Vec<HashMap<String, Tensor>>,
    /// One sender per blocked AllReduce handler; all fired together when the
    /// round closes. Payload is (serialized_global, new_round).
    waiters: Vec<oneshot::Sender<(Vec<u8>, u64)>>,
    /// True once the timeout task for the current round has been spawned, so
    /// we only spawn one per round.
    timer_spawned: bool,
}

/// Close `round` if it is still the current round and at least `min_workers`
/// submissions are present. Called both from the natural path (all submitted)
/// and from the timeout task.
async fn close_round(
    inner: &Arc<Mutex<Inner>>,
    expected_round: u64,
    min_workers: usize,
    reason: &str,
) {
    let mut st = inner.lock().await;

    // Round already closed naturally — nothing to do.
    if st.round != expected_round {
        return;
    }

    if st.pending.len() < min_workers {
        warn!(
            round = expected_round,
            submitted = st.pending.len(),
            min_workers,
            "{reason}: not enough submissions to close round"
        );
        return;
    }

    let submitted = st.pending.len();
    let pending = std::mem::take(&mut st.pending);
    let waiters = std::mem::take(&mut st.waiters);
    st.timer_spawned = false;

    // The borrow checker can't split-borrow through MutexGuard's DerefMut, but
    // can split borrows on fields of a plain &mut Inner. Reborrow first.
    let st_inner: &mut Inner = &mut *st;
    if let Err(e) = st_inner.outer.step(&mut st_inner.global, &pending) {
        warn!(round = expected_round, error = %e, "outer step failed; dropping round waiters");
        return; // waiters dropped → their rx.await gets Err → Status::internal
    }

    let bytes = match params::serialize(&st.global) {
        Ok(b) => b,
        Err(e) => {
            warn!(round = expected_round, error = %e, "serialize failed; dropping round waiters");
            return;
        }
    };

    st.round += 1;
    let new_round = st.round;
    info!(
        round = expected_round,
        new_round,
        submitted,
        "{reason}: round closed"
    );
    for waiter in waiters {
        let _ = waiter.send((bytes.clone(), new_round));
    }
}

/// The service is cloned per connection by tonic, so all shared state lives
/// behind an Arc<Mutex<_>>.
#[derive(Clone)]
struct CoordinatorService {
    world_size: usize,
    min_workers: usize,
    round_timeout: Duration,
    device: Device,
    inner: Arc<Mutex<Inner>>,
}

#[tonic::async_trait]
impl Diloco for CoordinatorService {
    /// Serve the current global parameters and the current round number. Works
    /// for both fresh worker starts (returns round=1, θ⁰) and mid-run resyncs
    /// (returns the live round and the current θ).
    async fn init(&self, request: Request<InitRequest>) -> Result<Response<ParamMsg>, Status> {
        let rank = request.into_inner().rank;
        let st = self.inner.lock().await;
        let bytes = params::serialize(&st.global)
            .map_err(|e| Status::internal(format!("serialize failed: {e}")))?;
        info!(rank, round = st.round, "worker fetched current parameters");
        Ok(Response::new(ParamMsg {
            params: bytes,
            round: st.round,
        }))
    }

    async fn all_reduce(
        &self,
        request: Request<AllReduceRequest>,
    ) -> Result<Response<ParamMsg>, Status> {
        let req = request.into_inner();
        let local = params::deserialize(&req.params, &self.device)
            .map_err(|e| Status::invalid_argument(format!("bad params payload: {e}")))?;

        let (rx, spawn_timer, current_round, all_in) = {
            let mut st = self.inner.lock().await;

            if req.round != st.round {
                return Err(Status::failed_precondition(format!(
                    "worker {} sent round {} but coordinator is on round {} — call Init to resync",
                    req.rank, req.round, st.round
                )));
            }

            let (tx, rx) = oneshot::channel();
            st.pending.push(local);
            st.waiters.push(tx);
            let submitted = st.pending.len();

            info!(
                round = st.round,
                rank = req.rank,
                submitted,
                world_size = self.world_size,
                "received local parameters"
            );

            let current_round = st.round;
            let spawn_timer = !st.timer_spawned;
            if spawn_timer {
                st.timer_spawned = true;
            }
            let all_in = submitted == self.world_size;
            (rx, spawn_timer, current_round, all_in)
        }; // lock released here

        // Spawn at most one timeout task per round. It fires close_round with
        // min_workers so a dead-worker situation doesn't stall indefinitely.
        if spawn_timer {
            let inner_arc = Arc::clone(&self.inner);
            let timeout = self.round_timeout;
            let min_workers = self.min_workers;
            tokio::spawn(async move {
                tokio::time::sleep(timeout).await;
                close_round(&inner_arc, current_round, min_workers, "timeout").await;
            });
        }

        // Natural close: only attempt when all world_size workers have submitted
        // to avoid the spurious "not enough submissions" warning on every arrival.
        if all_in {
            close_round(&self.inner, current_round, self.world_size, "natural").await;
        }

        let (params, round) = rx
            .await
            .map_err(|_| Status::internal("round was canceled (outer step failed)"))?;
        Ok(Response::new(ParamMsg { params, round }))
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

    let text = std::fs::read_to_string(&args.corpus)
        .with_context(|| format!("reading corpus from {}", args.corpus.display()))?;
    let tokenizer = CharTokenizer::from_text(&text);
    let cfg = Config::tiny(tokenizer.vocab_size());

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let _model = GptModel::new(&cfg, vb)?;

    let global = match &args.init {
        Some(path) if path.exists() => {
            info!(path = %path.display(), "loading shared initial parameters");
            params::load_file(path, &device)?
        }
        Some(path) => {
            let global = params::varmap_tensors(&varmap);
            params::save_file(path, &global)?;
            info!(path = %path.display(), "saved initial parameters for sharing");
            global
        }
        None => params::varmap_tensors(&varmap),
    };

    let outer = OuterOptimizer::new(&global, args.outer_lr, args.outer_momentum)?;
    let min_workers = args.min_workers.unwrap_or(args.world_size);

    info!(
        world_size = args.world_size,
        min_workers,
        round_timeout_secs = args.round_timeout_secs,
        vocab_size = cfg.vocab_size,
        outer_lr = args.outer_lr,
        outer_momentum = args.outer_momentum,
        listen = %args.listen,
        "coordinator ready"
    );

    let service = CoordinatorService {
        world_size: args.world_size,
        min_workers,
        round_timeout: Duration::from_secs(args.round_timeout_secs),
        device,
        inner: Arc::new(Mutex::new(Inner {
            global,
            outer,
            round: 1,
            pending: Vec::new(),
            waiters: Vec::new(),
            timer_spawned: false,
        })),
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
