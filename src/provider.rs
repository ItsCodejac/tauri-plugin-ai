use async_trait::async_trait;
use tokio::sync::mpsc;

use crate::config::{CompletionRequest, CompletionResponse, StreamChunk};
use crate::error::Result;

/// Trait for cloud AI providers (Anthropic, OpenAI, Ollama, etc.)
#[async_trait]
pub trait Provider: Send + Sync {
    /// Provider name (e.g. "anthropic", "openai", "ollama")
    fn name(&self) -> &str;

    /// Perform a non-streaming completion request.
    async fn complete(
        &self,
        request: CompletionRequest,
        api_key: &str,
    ) -> Result<CompletionResponse>;

    /// Perform a streaming completion request, sending chunks through the sender.
    async fn stream(
        &self,
        request: CompletionRequest,
        api_key: &str,
        sender: mpsc::Sender<StreamChunk>,
    ) -> Result<()>;

    /// List models available from this provider.
    fn available_models(&self) -> Vec<String>;
}
