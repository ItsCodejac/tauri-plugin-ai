use std::collections::HashMap;
use std::sync::Arc;

use serde::Serialize;

use crate::backend::{InferenceBackend, InferenceInput, InferenceOutput};
use crate::config::{CompletionRequest, CompletionResponse, ModelConfig, StreamChunk};
use crate::error::{Error, Result};
use crate::provider::Provider;

/// Information about an available model.
#[derive(Debug, Clone, Serialize)]
pub struct ModelInfo {
    pub name: String,
    pub provider: String,
    pub is_local: bool,
    pub is_loaded: bool,
}

/// Central registry that tracks providers, backends, API keys,
/// and routes requests to the right destination.
pub struct ModelRegistry {
    providers: HashMap<String, Arc<dyn Provider>>,
    backends: HashMap<String, Box<dyn InferenceBackend>>,
    api_keys: HashMap<String, String>,
    default_provider: Option<String>,
}

impl ModelRegistry {
    pub fn new() -> Self {
        Self {
            providers: HashMap::new(),
            backends: HashMap::new(),
            api_keys: HashMap::new(),
            default_provider: None,
        }
    }

    /// Register a cloud provider.
    pub fn register_provider(&mut self, provider: impl Provider + 'static) {
        let name = provider.name().to_string();
        if self.default_provider.is_none() {
            self.default_provider = Some(name.clone());
        }
        self.providers.insert(name, Arc::new(provider));
    }

    /// Register a cloud provider from a boxed trait object.
    pub fn register_provider_boxed(&mut self, provider: Box<dyn Provider>) {
        let name = provider.name().to_string();
        if self.default_provider.is_none() {
            self.default_provider = Some(name.clone());
        }
        self.providers.insert(name, Arc::from(provider));
    }

    /// Register a local inference backend.
    pub fn register_backend(&mut self, backend: Box<dyn InferenceBackend>) {
        let name = backend.name().to_string();
        self.backends.insert(name, backend);
    }

    /// Store an API key for a provider.
    pub fn set_api_key(&mut self, provider: &str, key: String) {
        self.api_keys.insert(provider.to_string(), key);
    }

    /// Get the API key for a provider.
    pub fn get_api_key(&self, provider: &str) -> Option<&String> {
        self.api_keys.get(provider)
    }

    /// List all available models across all providers and backends.
    pub fn list_models(&self) -> Vec<ModelInfo> {
        let mut models = Vec::new();

        for (provider_name, provider) in &self.providers {
            for model_name in provider.available_models() {
                models.push(ModelInfo {
                    name: model_name,
                    provider: provider_name.clone(),
                    is_local: false,
                    is_loaded: false,
                });
            }
        }

        models
    }

    /// List configured provider names.
    pub fn list_providers(&self) -> Vec<String> {
        self.providers.keys().cloned().collect()
    }

    /// Resolve which provider to use for a request (returns an Arc clone).
    fn resolve_provider(&self, name: Option<&str>) -> Result<Arc<dyn Provider>> {
        let provider_name = name
            .or(self.default_provider.as_deref())
            .ok_or_else(|| Error::Config("No provider specified and no default set".into()))?;

        self.providers
            .get(provider_name)
            .cloned()
            .ok_or_else(|| Error::Config(format!("Provider '{}' not registered", provider_name)))
    }

    /// Resolve the provider and its API key in one step.
    ///
    /// Returns an `Arc<dyn Provider>` and the API key so the caller can
    /// release the registry lock before making HTTP calls.
    pub fn resolve_provider_and_key(
        &self,
        name: Option<&str>,
    ) -> Result<(Arc<dyn Provider>, String)> {
        let provider = self.resolve_provider(name)?;
        let api_key = self
            .api_keys
            .get(provider.name())
            .ok_or_else(|| {
                Error::Config(format!(
                    "No API key set for provider '{}'. Call set_api_key first.",
                    provider.name()
                ))
            })?
            .clone();
        Ok((provider, api_key))
    }

    /// Resolve the provider and its API key, allowing an empty key for
    /// providers that don't require one (e.g. Ollama).
    pub fn resolve_provider_and_optional_key(
        &self,
        name: Option<&str>,
    ) -> Result<(Arc<dyn Provider>, String)> {
        let provider = self.resolve_provider(name)?;
        let api_key = self
            .api_keys
            .get(provider.name())
            .cloned()
            .unwrap_or_default();
        Ok((provider, api_key))
    }

    /// Perform a non-streaming completion.
    ///
    /// NOTE: Prefer calling `resolve_provider_and_key` from the command layer
    /// and releasing the lock before the HTTP call. This method is kept for
    /// convenience in non-concurrent contexts.
    pub async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        let provider = self.resolve_provider(request.provider.as_deref())?;
        let api_key = self
            .api_keys
            .get(provider.name())
            .ok_or_else(|| {
                Error::Config(format!(
                    "No API key set for provider '{}'. Call set_api_key first.",
                    provider.name()
                ))
            })?;
        provider.complete(request, api_key).await
    }

    /// Perform a streaming completion.
    ///
    /// NOTE: Prefer calling `resolve_provider_and_key` from the command layer
    /// and releasing the lock before the HTTP call. This method is kept for
    /// convenience in non-concurrent contexts.
    pub async fn stream(
        &self,
        request: CompletionRequest,
        sender: tokio::sync::mpsc::Sender<StreamChunk>,
    ) -> Result<()> {
        let provider = self.resolve_provider(request.provider.as_deref())?;
        let api_key = self
            .api_keys
            .get(provider.name())
            .ok_or_else(|| {
                Error::Config(format!(
                    "No API key set for provider '{}'. Call set_api_key first.",
                    provider.name()
                ))
            })?;
        provider.stream(request, api_key, sender).await
    }

    /// Load a local model via the appropriate backend.
    pub fn load_model(&mut self, config: ModelConfig) -> Result<()> {
        let backend = self
            .backends
            .get_mut(&config.provider)
            .ok_or_else(|| Error::Config(format!("Backend '{}' not registered", config.provider)))?;
        backend.load_model(&config)
    }

    /// Unload a local model.
    pub fn unload_model(&mut self, backend_name: &str, model_name: &str) -> Result<()> {
        let backend = self
            .backends
            .get_mut(backend_name)
            .ok_or_else(|| Error::Config(format!("Backend '{}' not registered", backend_name)))?;
        backend.unload_model(model_name)
    }

    /// Run inference on a loaded model (non-LLM).
    pub fn infer(
        &self,
        backend_name: &str,
        model_name: &str,
        input: InferenceInput,
    ) -> Result<InferenceOutput> {
        let backend = self
            .backends
            .get(backend_name)
            .ok_or_else(|| Error::Config(format!("Backend '{}' not registered", backend_name)))?;
        if !backend.is_loaded(model_name) {
            return Err(Error::ModelNotLoaded(model_name.to_string()));
        }
        backend.infer(model_name, input)
    }

    /// List all backends and their loaded models.
    pub fn list_backends(&self) -> Vec<BackendInfo> {
        self.backends
            .iter()
            .map(|(name, backend)| BackendInfo {
                name: name.clone(),
                loaded_models: backend.loaded_models(),
            })
            .collect()
    }
}

/// Info about a registered inference backend and its loaded models.
#[derive(Debug, Clone, Serialize)]
pub struct BackendInfo {
    pub name: String,
    pub loaded_models: Vec<String>,
}

/// Thread-safe wrapper for the model registry, stored as Tauri managed state.
/// Uses tokio::sync::Mutex so the lock can be held across await points.
pub struct AiState(pub tokio::sync::Mutex<ModelRegistry>);
