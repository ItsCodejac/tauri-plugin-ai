use crate::config::ModelConfig;
use crate::error::Result;
use serde::{Deserialize, Serialize};

/// Tensor data for model inference input/output.
///
/// This is the generic format for passing data to and from
/// non-LLM models (image classifiers, embedding models,
/// depth estimators, audio processors, etc.)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorData {
    /// Tensor shape (e.g. [1, 3, 224, 224] for a batch of images)
    pub shape: Vec<usize>,
    /// Data type ("f32", "f16", "i32", "u8", etc.)
    pub dtype: String,
    /// Raw data as bytes (frontend encodes, backend decodes)
    pub data: Vec<u8>,
}

/// Input for general model inference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceInput {
    /// Named inputs (model-specific, e.g. "input_ids", "pixel_values", "audio")
    pub tensors: std::collections::HashMap<String, TensorData>,
    /// Optional parameters (model-specific, e.g. threshold, top_k)
    pub params: Option<serde_json::Value>,
}

/// Output from general model inference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceOutput {
    /// Named outputs (model-specific, e.g. "logits", "embeddings", "depth_map")
    pub tensors: std::collections::HashMap<String, TensorData>,
    /// Optional metadata (timing, model info, etc.)
    pub metadata: Option<serde_json::Value>,
}

/// Trait for local model inference backends.
///
/// This is for NON-LLM models: image models, embedding models,
/// audio models, etc. For LLM chat/completion, use the Provider trait
/// or the chat-specific backend (Candle LLM).
///
/// Implementations are feature-gated:
/// - `local-onnx`: ONNX Runtime backend for any .onnx model
/// - `local-llm`: Candle backend for GGUF/Safetensors LLMs
pub trait InferenceBackend: Send + Sync {
    /// Backend name (e.g. "onnx", "candle")
    fn name(&self) -> &str;

    /// Load a model into memory.
    fn load_model(&mut self, config: &ModelConfig) -> Result<()>;

    /// Unload a model from memory.
    fn unload_model(&mut self, model_name: &str) -> Result<()>;

    /// Check if a model is currently loaded.
    fn is_loaded(&self, model_name: &str) -> bool;

    /// List loaded models.
    fn loaded_models(&self) -> Vec<String>;

    /// Run inference on a loaded model.
    fn infer(&self, model_name: &str, input: InferenceInput) -> Result<InferenceOutput>;
}
