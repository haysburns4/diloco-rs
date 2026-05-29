//! A tiny CSV metrics logger shared by the DiLoCo worker (rank 0) and the
//! synchronous baseline. Both write the identical schema through this type, so
//! the two runs' output files are directly comparable by the analysis script.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::Result;

/// One row per logical checkpoint (a DiLoCo round, or every K baseline steps).
/// The header is written on creation and every row is flushed immediately, so a
/// partial run still leaves a readable file.
pub struct MetricsLogger {
    writer: BufWriter<File>,
}

impl MetricsLogger {
    pub fn create(path: impl AsRef<Path>) -> Result<Self> {
        let mut writer = BufWriter::new(File::create(path)?);
        writeln!(
            writer,
            "round,total_samples,wall_clock_s,comm_bytes,val_loss,train_loss"
        )?;
        writer.flush()?;
        Ok(Self { writer })
    }

    /// `total_samples` is the cumulative number of sequences processed across
    /// all workers (the compute axis); `comm_bytes` is the cumulative bytes
    /// communicated. Both runs compute these on the same basis — see
    /// [`crate::params::allreduce_bytes_per_worker`].
    pub fn log(
        &mut self,
        round: u64,
        total_samples: u64,
        wall_clock_s: f64,
        comm_bytes: u64,
        val_loss: f32,
        train_loss: f32,
    ) -> Result<()> {
        writeln!(
            self.writer,
            "{round},{total_samples},{wall_clock_s:.4},{comm_bytes},{val_loss:.6},{train_loss:.6}"
        )?;
        self.writer.flush()?;
        Ok(())
    }
}
