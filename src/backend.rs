use crate::config::{CompletionRequest, CompletionResponse, ModelConfig};
use crate::error::Result;

/// Input types for local inference backends.
pub enum InferenceInput {
    Text(CompletionRequest),
    // Future: Tensor, Image, Audio
}

/// Output types from local inference backends.
pub enum InferenceOutput {
    Text(CompletionResponse),
    // Future: Tensor, Embedding
}

/// Trait for local inference backends (ONNX Runtime, Candle, etc.)
///
/// Implementations are feature-gated and will be added behind
/// `local-onnx` and `local-llm` feature flags.
pub trait InferenceBackend: Send + Sync {
    /// Backend name (e.g. "onnx", "candle")
    fn name(&self) -> &str;

    /// Load a model into memory.
    fn load_model(&mut self, config: &ModelConfig) -> Result<()>;

    /// Unload a model from memory.
    fn unload_model(&mut self, model_name: &str) -> Result<()>;

    /// Check if a model is currently loaded.
    fn is_loaded(&self, model_name: &str) -> bool;

    /// Run inference on a loaded model.
    fn infer(&self, model_name: &str, input: InferenceInput) -> Result<InferenceOutput>;
}
