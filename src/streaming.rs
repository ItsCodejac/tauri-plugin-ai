use serde::Serialize;
use tauri::{AppHandle, Emitter, Runtime};
use tokio::sync::mpsc;

use crate::config::StreamChunk;

/// Event name for streaming AI chunks.
pub const STREAM_EVENT: &str = "ai:stream:chunk";

/// Wrapper for stream events that includes a request ID so the frontend
/// can correlate chunks with requests.
#[derive(Debug, Clone, Serialize)]
pub struct StreamEvent {
    pub request_id: String,
    pub chunk: StreamChunk,
}

/// Spawn a task that forwards chunks from an mpsc receiver to Tauri events.
pub fn forward_stream_to_events<R: Runtime>(
    app: AppHandle<R>,
    request_id: String,
    mut receiver: mpsc::Receiver<StreamChunk>,
) {
    tauri::async_runtime::spawn(async move {
        while let Some(chunk) = receiver.recv().await {
            let event = StreamEvent {
                request_id: request_id.clone(),
                chunk,
            };
            if let Err(e) = app.emit(STREAM_EVENT, &event) {
                log::error!("Failed to emit stream event: {}", e);
                break;
            }
        }
    });
}
