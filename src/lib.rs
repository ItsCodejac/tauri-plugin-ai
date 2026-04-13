pub mod backend;
pub mod backends;
pub mod commands;
pub mod config;
pub mod error;
pub mod models;
pub mod provider;
pub mod streaming;

#[cfg(feature = "cloud")]
pub mod providers;

use tauri::{
    plugin::{Builder, TauriPlugin},
    Manager, Runtime,
};

use models::{AiState, ModelRegistry};

/// Initialize the AI plugin.
///
/// # Example
///
/// ```rust,no_run
/// fn main() {
///     tauri::Builder::default()
///         .plugin(tauri_plugin_ai::init())
///         .run(tauri::generate_context!())
///         .expect("error while running tauri application");
/// }
/// ```
pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::<R, ()>::new("ai")
        .invoke_handler(tauri::generate_handler![
            // Chat/Completion (LLMs)
            commands::complete,
            commands::stream,
            // Model management
            commands::list_models,
            commands::load_model,
            commands::unload_model,
            commands::set_api_key,
            commands::get_api_key,
            commands::get_providers,
            // General inference (non-LLM models)
            commands::infer,
            commands::list_backends,
        ])
        .setup(|app, _api| {
            let mut registry = ModelRegistry::new();

            // Register default cloud providers when the cloud feature is enabled
            #[cfg(feature = "cloud")]
            {
                use providers::anthropic::AnthropicProvider;
                use providers::ollama::OllamaProvider;
                use providers::openai::OpenAiProvider;

                registry.register_provider(AnthropicProvider::new(None));
                registry.register_provider(OpenAiProvider::new(None));
                registry.register_provider(OllamaProvider::new(None));
            }

            // Register ONNX Runtime backend when the local-onnx feature is enabled
            #[cfg(feature = "local-onnx")]
            {
                use backends::onnx::OnnxBackend;
                match OnnxBackend::new() {
                    Ok(backend) => {
                        registry.register_backend(Box::new(backend));
                        log::info!("ONNX Runtime backend registered");
                    }
                    Err(e) => {
                        log::warn!("Failed to initialize ONNX backend: {e}");
                    }
                }
            }

            app.manage(AiState(tokio::sync::Mutex::new(registry)));
            Ok(())
        })
        .build()
}
