//! Conversions between a model's parameters and the flat byte buffers that
//! travel between workers and the coordinator. The wire format is safetensors,
//! so a payload is self-describing (names, shapes, dtypes) and the coordinator
//! can average tensors without separately knowing the architecture.

use std::collections::HashMap;

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
