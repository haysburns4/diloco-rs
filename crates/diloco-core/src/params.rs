//! Conversions between a model's parameters and the flat byte buffers that
//! travel between workers and the coordinator. The wire format is safetensors,
//! so a payload is self-describing (names, shapes, dtypes) and the coordinator
//! can average tensors without separately knowing the architecture.

use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::VarMap;

/// Snapshot every parameter in `varmap` as a name -> tensor map. Tensor clones
/// are cheap (the storage is reference-counted).
pub fn varmap_tensors(varmap: &VarMap) -> HashMap<String, Tensor> {
    varmap
        .data()
        .lock()
        .unwrap()
        .iter()
        .map(|(name, var)| (name.clone(), var.as_tensor().clone()))
        .collect()
}

/// Overwrite the values of the variables in `varmap` with `tensors`, matched by
/// name. The variables themselves are reused, so any optimizer holding
/// references to them stays valid. Errors if a name is missing.
pub fn load_into_varmap(varmap: &mut VarMap, tensors: &HashMap<String, Tensor>) -> Result<()> {
    varmap.set(tensors.iter())?;
    Ok(())
}

/// Serialize a parameter map to a safetensors byte buffer.
pub fn serialize(tensors: &HashMap<String, Tensor>) -> Result<Vec<u8>> {
    Ok(safetensors::serialize(tensors.iter(), None)?)
}

/// Inverse of [`serialize`]: load a safetensors buffer back into a map on
/// `device`.
pub fn deserialize(bytes: &[u8], device: &Device) -> Result<HashMap<String, Tensor>> {
    Ok(candle_core::safetensors::load_buffer(bytes, device)?)
}

/// Persist a parameter map to a safetensors file. Used to share a single
/// initial `theta^(0)` between the coordinator and the synchronous baseline so
/// both runs start from identical weights.
pub fn save_file(path: impl AsRef<Path>, tensors: &HashMap<String, Tensor>) -> Result<()> {
    std::fs::write(path, serialize(tensors)?)?;
    Ok(())
}

/// Inverse of [`save_file`]: load a safetensors file into a map on `device`.
pub fn load_file(path: impl AsRef<Path>, device: &Device) -> Result<HashMap<String, Tensor>> {
    deserialize(&std::fs::read(path)?, device)
}

/// Bytes a single worker moves per all-reduce event: it uploads its local
/// parameters and downloads the reduced result, so `2 * payload`. The payload
/// size is taken from the actual safetensors encoding, so DiLoCo and the
/// baseline account for communication on the same measured basis (their model
/// shapes are identical, so the per-event cost matches and only the number of
/// events differs).
pub fn allreduce_bytes_per_worker(tensors: &HashMap<String, Tensor>) -> Result<usize> {
    Ok(2 * serialize(tensors)?.len())
}
