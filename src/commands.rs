use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use tauri::{command, AppHandle, Runtime};
use tokio::sync::{mpsc, Mutex};

use crate::backend::{InferenceInput, InferenceOutput};
use crate::config::{CompletionRequest, CompletionResponse, ModelConfig};
use crate::error::Error;
use crate::models::{AiState, BackendInfo, ModelInfo};
use crate::streaming::forward_stream_to_events;

/// Monotonic counter for generating unique stream request IDs.
static REQUEST_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Tracks active stream senders so they can be cancelled.
/// Dropping the sender causes the provider's stream loop to get a send error and stop.
pub struct ActiveStreams(pub Arc<Mutex<HashMap<String, mpsc::Sender<crate::config::StreamChunk>>>>);

#[command]
pub async fn complete(
    state: tauri::State<'_, AiState>,
    request: CompletionRequest,
) -> Result<CompletionResponse, Error> {
    if request.messages.is_empty() {
        return Err(Error::Config("messages array cannot be empty".into()));
    }

    // Resolve provider + key under the lock, then release before HTTP call
    let (provider, api_key) = {
        let registry = state.0.lock().await;
        registry.resolve_provider_and_optional_key(request.provider.as_deref())?
    };
    // Lock released here — other commands can proceed

    provider.complete(request, &api_key).await
}

#[command]
pub async fn stream<R: Runtime>(
    app: AppHandle<R>,
    state: tauri::State<'_, AiState>,
    active_streams: tauri::State<'_, ActiveStreams>,
    request: CompletionRequest,
) -> Result<String, Error> {
    if request.messages.is_empty() {
        return Err(Error::Config("messages array cannot be empty".into()));
    }

    let request_id = format!(
        "ai-stream-{}",
        REQUEST_COUNTER.fetch_add(1, Ordering::Relaxed)
    );

    let (sender, receiver) = mpsc::channel(64);

    // Store the sender so it can be cancelled
    {
        let mut streams = active_streams.0.lock().await;
        streams.insert(request_id.clone(), sender.clone());
    }

    forward_stream_to_events(app, request_id.clone(), receiver);

    // Resolve provider + key under the lock, then release before streaming
    let (provider, api_key) = {
        let registry = state.0.lock().await;
        registry.resolve_provider_and_optional_key(request.provider.as_deref())?
    };
    // Lock released here — other commands can proceed during the stream

    let rid = request_id.clone();
    let streams = Arc::clone(&active_streams.inner().0);

    // Spawn the stream so we can return the request_id immediately
    tauri::async_runtime::spawn(async move {
        if let Err(e) = provider.stream(request, &api_key, sender.clone()).await {
            // Send the error as a final chunk so the frontend receives it
            let _ = sender
                .send(crate::config::StreamChunk {
                    delta: String::new(),
                    done: true,
                    usage: None,
                    finish_reason: None,
                    error: Some(e.to_string()),
                })
                .await;
        }
        // Clean up when stream finishes naturally
        let mut s = streams.lock().await;
        s.remove(&rid);
    });

    Ok(request_id)
}

/// Cancel an active streaming request.
///
/// Dropping the sender causes the provider stream loop to receive a send error
/// and stop, which cleans up the HTTP connection.
#[command]
pub async fn cancel_stream(
    active_streams: tauri::State<'_, ActiveStreams>,
    request_id: String,
) -> Result<(), Error> {
    let mut streams = active_streams.0.lock().await;
    // Dropping the sender will cause the stream to stop
    streams.remove(&request_id);
    Ok(())
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
    if key.trim().is_empty() {
        return Err(Error::Config("API key cannot be empty".into()));
    }
    let mut registry = state.0.lock().await;
    registry.set_api_key(&provider, key);
    Ok(())
}

#[command]
pub async fn remove_api_key(
    state: tauri::State<'_, AiState>,
    provider: String,
) -> Result<(), Error> {
    let mut registry = state.0.lock().await;
    registry.remove_api_key(&provider);
    Ok(())
}

/// WARNING: This command exposes API keys to the renderer process.
/// It is intentionally excluded from default permissions.
/// Prefer the proxy pattern: have the renderer call `complete` or `stream`,
/// and Rust reads the key internally -- the key never leaves the backend.
/// Only enable this if the renderer absolutely needs the raw key value.
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
