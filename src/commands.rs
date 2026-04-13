use tauri::{command, AppHandle, Runtime};
use tokio::sync::mpsc;

use crate::backend::{InferenceInput, InferenceOutput};
use crate::config::{CompletionRequest, CompletionResponse, ModelConfig};
use crate::error::Error;
use crate::models::{AiState, ModelInfo};
use crate::streaming::forward_stream_to_events;

#[command]
pub async fn complete(
    state: tauri::State<'_, AiState>,
    request: CompletionRequest,
) -> Result<CompletionResponse, Error> {
    let registry = state.0.lock().await;
    registry.complete(request).await
}

#[command]
pub async fn stream<R: Runtime>(
    app: AppHandle<R>,
    state: tauri::State<'_, AiState>,
    request: CompletionRequest,
) -> Result<String, Error> {
    let request_id = format!(
        "{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
    );

    let (sender, receiver) = mpsc::channel(64);
    forward_stream_to_events(app, request_id.clone(), receiver);

    let registry = state.0.lock().await;
    registry.stream(request, sender).await?;

    Ok(request_id)
}

#[command]
pub async fn list_models(
    state: tauri::State<'_, AiState>,
) -> Result<Vec<ModelInfo>, Error> {
    let registry = state.0.lock().await;
    Ok(registry.list_models())
}

#[command]
pub async fn load_model(
    state: tauri::State<'_, AiState>,
    config: ModelConfig,
) -> Result<(), Error> {
    let mut registry = state.0.lock().await;
    registry.load_model(config)
}

#[command]
pub async fn unload_model(
    state: tauri::State<'_, AiState>,
    backend: String,
    name: String,
) -> Result<(), Error> {
    let mut registry = state.0.lock().await;
    registry.unload_model(&backend, &name)
}

#[command]
pub async fn set_api_key(
    state: tauri::State<'_, AiState>,
    provider: String,
    key: String,
) -> Result<(), Error> {
    let mut registry = state.0.lock().await;
    registry.set_api_key(&provider, key);
    Ok(())
}

#[command]
pub async fn get_api_key(
    state: tauri::State<'_, AiState>,
    provider: String,
) -> Result<Option<String>, Error> {
    let registry = state.0.lock().await;
    Ok(registry.get_api_key(&provider).cloned())
}

#[command]
pub async fn get_providers(
    state: tauri::State<'_, AiState>,
) -> Result<Vec<String>, Error> {
    let registry = state.0.lock().await;
    Ok(registry.list_providers())
}

/// Run inference on a loaded model (non-LLM).
///
/// This is the generic inference API for any model type:
/// image classifiers, embedding models, audio processors, etc.
/// For LLM chat, use `complete` or `stream` instead.
#[command]
pub async fn infer(
    state: tauri::State<'_, AiState>,
    backend: String,
    model: String,
    input: InferenceInput,
) -> Result<InferenceOutput, Error> {
    let registry = state.0.lock().await;
    registry.infer(&backend, &model, input)
}

/// List available inference backends and their loaded models.
#[command]
pub async fn list_backends(
    state: tauri::State<'_, AiState>,
) -> Result<Vec<BackendInfo>, Error> {
    let registry = state.0.lock().await;
    Ok(registry.list_backends())
}

#[derive(serde::Serialize)]
pub struct BackendInfo {
    pub name: String,
    pub loaded_models: Vec<String>,
}
