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

use std::collections::HashMap;

use tauri::{
    plugin::{Builder, TauriPlugin},
    Manager, Runtime,
};

use backend::InferenceBackend;
use commands::ActiveStreams;
use models::{AiState, ModelRegistry};
use provider::Provider;

/// Initialize the AI plugin with default settings.
///
/// Registers the default cloud providers (Anthropic, OpenAI, Ollama) when the
/// `cloud` feature is enabled, and the ONNX backend when `local-onnx` is enabled.
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
    AiPluginBuilder::new().build()
}

/// Create a builder for advanced plugin configuration.
///
/// # Example
///
/// ```rust,no_run
/// fn main() {
///     tauri::Builder::default()
///         .plugin(
///             tauri_plugin_ai::builder()
///                 .no_defaults()
///                 .provider(my_custom_provider)
///                 .backend(my_custom_backend)
///                 .build()
///         )
///         .run(tauri::generate_context!())
///         .expect("error while running tauri application");
/// }
/// ```
pub fn builder() -> AiPluginBuilder {
    AiPluginBuilder::new()
}

/// Builder for configuring the AI plugin with custom providers and backends.
pub struct AiPluginBuilder {
    providers: Vec<Box<dyn Provider>>,
    backends: Vec<Box<dyn InferenceBackend>>,
    register_defaults: bool,
}

impl AiPluginBuilder {
    /// Create a new builder. By default, built-in providers and backends are registered.
    pub fn new() -> Self {
        Self {
            providers: Vec::new(),
            backends: Vec::new(),
            register_defaults: true,
        }
    }

    /// Register a custom provider.
    pub fn provider(mut self, p: impl Provider + 'static) -> Self {
        self.providers.push(Box::new(p));
        self
    }

    /// Register a custom inference backend.
    pub fn backend(mut self, b: impl InferenceBackend + 'static) -> Self {
        self.backends.push(Box::new(b));
        self
    }

    /// Skip registering the default providers (Anthropic, OpenAI, Ollama)
    /// and backends (ONNX). Only custom-registered ones will be available.
    pub fn no_defaults(mut self) -> Self {
        self.register_defaults = false;
        self
    }

    /// Build the Tauri plugin.
    pub fn build<R: Runtime>(self) -> TauriPlugin<R> {
        let AiPluginBuilder {
            providers,
            backends,
            register_defaults,
        } = self;

        Builder::<R, ()>::new("ai")
            .invoke_handler(tauri::generate_handler![
                // Chat/Completion (LLMs)
                commands::complete,
                commands::stream,
                commands::cancel_stream,
                // Model management
                commands::list_models,
                commands::load_model,
                commands::unload_model,
                commands::set_api_key,
                commands::remove_api_key,
                commands::get_api_key,
                commands::get_providers,
                // General inference (non-LLM models)
                commands::infer,
                commands::list_backends,
            ])
            .setup(move |app, _api| {
                let mut registry = ModelRegistry::new();

                // Register default cloud providers when the cloud feature is enabled
                if register_defaults {
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
                }

                // Register custom providers
                for p in providers {
                    registry.register_provider_boxed(p);
                }

                // Register custom backends
                for b in backends {
                    registry.register_backend(b);
                }

                app.manage(AiState(tokio::sync::Mutex::new(registry)));
                app.manage(ActiveStreams(std::sync::Arc::new(tokio::sync::Mutex::new(HashMap::new()))));
                Ok(())
            })
            .build()
    }
}

impl Default for AiPluginBuilder {
    fn default() -> Self {
        Self::new()
    }
}
