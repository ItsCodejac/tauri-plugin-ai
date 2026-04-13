use std::collections::HashMap;
use std::sync::Mutex;

use ort::session::Session;
use ort::value::Tensor;

use crate::backend::{InferenceBackend, InferenceInput, InferenceOutput, TensorData};
use crate::config::ModelConfig;
use crate::error::{Error, Result};

pub struct OnnxBackend {
    /// Sessions wrapped in Mutex because `Session::run` requires `&mut self`.
    sessions: HashMap<String, Mutex<Session>>,
}

impl OnnxBackend {
    pub fn new() -> Result<Self> {
        Ok(Self {
            sessions: HashMap::new(),
        })
    }
}

impl InferenceBackend for OnnxBackend {
    fn name(&self) -> &str {
        "onnx"
    }

    fn load_model(&mut self, config: &ModelConfig) -> Result<()> {
        let path = config
            .model_path
            .as_ref()
            .ok_or_else(|| Error::Config("model_path required for ONNX models".into()))?;

        let session = Session::builder()
            .map_err(|e| Error::Backend(format!("ONNX session builder error: {e}")))?
            .commit_from_file(path)
            .map_err(|e| Error::Backend(format!("Failed to load ONNX model '{}': {e}", path)))?;

        self.sessions
            .insert(config.name.clone(), Mutex::new(session));
        log::info!("ONNX model '{}' loaded from {}", config.name, path);
        Ok(())
    }

    fn unload_model(&mut self, model_name: &str) -> Result<()> {
        self.sessions
            .remove(model_name)
            .ok_or_else(|| Error::ModelNotFound(model_name.to_string()))?;
        log::info!("ONNX model '{}' unloaded", model_name);
        Ok(())
    }

    fn is_loaded(&self, model_name: &str) -> bool {
        self.sessions.contains_key(model_name)
    }

    fn loaded_models(&self) -> Vec<String> {
        self.sessions.keys().cloned().collect()
    }

    fn infer(&self, model_name: &str, input: InferenceInput) -> Result<InferenceOutput> {
        let session_mutex = self
            .sessions
            .get(model_name)
            .ok_or_else(|| Error::ModelNotLoaded(model_name.to_string()))?;

        let mut session = session_mutex
            .lock()
            .map_err(|e| Error::Backend(format!("Session lock poisoned: {e}")))?;

        // Convert each TensorData into an owned ort DynValue.
        // Build a Vec<(String, SessionInputValue)> which converts into SessionInputs.
        let ort_inputs: Vec<(String, ort::value::DynValue)> = input
            .tensors
            .iter()
            .map(|(name, td)| {
                let value = tensor_data_to_ort_value(td)?;
                Ok((name.clone(), value))
            })
            .collect::<Result<Vec<_>>>()?;

        // Get output names before running (run borrows session mutably)
        let output_names: Vec<String> = session
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();

        // Run inference
        let outputs = session
            .run(ort_inputs)
            .map_err(|e| Error::Backend(format!("ONNX inference error: {e}")))?;

        // Convert outputs back to TensorData
        let mut result_tensors = HashMap::new();

        for (i, (_, output_ref)) in outputs.iter().enumerate() {
            let out_name = output_names
                .get(i)
                .cloned()
                .unwrap_or_else(|| format!("output_{i}"));

            let tensor_data = ort_value_to_tensor_data(&output_ref)?;
            result_tensors.insert(out_name, tensor_data);
        }

        Ok(InferenceOutput {
            tensors: result_tensors,
            metadata: None,
        })
    }
}

/// Convert our generic TensorData (bytes + shape + dtype) into an owned ort DynValue.
fn tensor_data_to_ort_value(td: &TensorData) -> Result<ort::value::DynValue> {
    let shape = td.shape.clone();
    let data = &td.data;

    match td.dtype.as_str() {
        "f32" => {
            let expected_len = td.shape.iter().product::<usize>() * 4;
            if data.len() != expected_len {
                return Err(Error::Backend(format!(
                    "f32 tensor data length mismatch: expected {expected_len} bytes, got {}",
                    data.len()
                )));
            }
            let float_data: Vec<f32> = data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let tensor = Tensor::from_array((shape, float_data))
                .map_err(|e| Error::Backend(format!("Failed to create f32 tensor: {e}")))?;
            Ok(tensor.into_dyn())
        }
        "f64" => {
            let expected_len = td.shape.iter().product::<usize>() * 8;
            if data.len() != expected_len {
                return Err(Error::Backend(format!(
                    "f64 tensor data length mismatch: expected {expected_len} bytes, got {}",
                    data.len()
                )));
            }
            let float_data: Vec<f64> = data
                .chunks_exact(8)
                .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                .collect();
            let tensor = Tensor::from_array((shape, float_data))
                .map_err(|e| Error::Backend(format!("Failed to create f64 tensor: {e}")))?;
            Ok(tensor.into_dyn())
        }
        "i32" => {
            let expected_len = td.shape.iter().product::<usize>() * 4;
            if data.len() != expected_len {
                return Err(Error::Backend(format!(
                    "i32 tensor data length mismatch: expected {expected_len} bytes, got {}",
                    data.len()
                )));
            }
            let int_data: Vec<i32> = data
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let tensor = Tensor::from_array((shape, int_data))
                .map_err(|e| Error::Backend(format!("Failed to create i32 tensor: {e}")))?;
            Ok(tensor.into_dyn())
        }
        "i64" => {
            let expected_len = td.shape.iter().product::<usize>() * 8;
            if data.len() != expected_len {
                return Err(Error::Backend(format!(
                    "i64 tensor data length mismatch: expected {expected_len} bytes, got {}",
                    data.len()
                )));
            }
            let int_data: Vec<i64> = data
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                .collect();
            let tensor = Tensor::from_array((shape, int_data))
                .map_err(|e| Error::Backend(format!("Failed to create i64 tensor: {e}")))?;
            Ok(tensor.into_dyn())
        }
        "u8" => {
            let expected_len = td.shape.iter().product::<usize>();
            if data.len() != expected_len {
                return Err(Error::Backend(format!(
                    "u8 tensor data length mismatch: expected {expected_len} bytes, got {}",
                    data.len()
                )));
            }
            let tensor = Tensor::from_array((shape, data.to_vec()))
                .map_err(|e| Error::Backend(format!("Failed to create u8 tensor: {e}")))?;
            Ok(tensor.into_dyn())
        }
        other => Err(Error::Backend(format!(
            "Unsupported tensor dtype '{other}'. Supported: f32, f64, i32, i64, u8"
        ))),
    }
}

/// Convert an ort output value back to our generic TensorData format.
///
/// Tries f32 first (most common), then falls back to other numeric types.
fn ort_value_to_tensor_data(
    value: &ort::value::ValueRef<'_>,
) -> Result<TensorData> {
    // Try f32 (most common output type)
    if let Ok((shape, data)) = value.try_extract_tensor::<f32>() {
        let shape_vec: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        return Ok(TensorData {
            shape: shape_vec,
            dtype: "f32".to_string(),
            data: bytes,
        });
    }

    // Try i64
    if let Ok((shape, data)) = value.try_extract_tensor::<i64>() {
        let shape_vec: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        return Ok(TensorData {
            shape: shape_vec,
            dtype: "i64".to_string(),
            data: bytes,
        });
    }

    // Try i32
    if let Ok((shape, data)) = value.try_extract_tensor::<i32>() {
        let shape_vec: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        return Ok(TensorData {
            shape: shape_vec,
            dtype: "i32".to_string(),
            data: bytes,
        });
    }

    // Try u8
    if let Ok((shape, data)) = value.try_extract_tensor::<u8>() {
        let shape_vec: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        let bytes: Vec<u8> = data.to_vec();
        return Ok(TensorData {
            shape: shape_vec,
            dtype: "u8".to_string(),
            data: bytes,
        });
    }

    // Try f64
    if let Ok((shape, data)) = value.try_extract_tensor::<f64>() {
        let shape_vec: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        return Ok(TensorData {
            shape: shape_vec,
            dtype: "f64".to_string(),
            data: bytes,
        });
    }

    Err(Error::Backend(
        "Failed to extract output tensor: unsupported element type".to_string(),
    ))
}
