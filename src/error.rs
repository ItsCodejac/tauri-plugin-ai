use serde::Serialize;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Provider error: {0}")]
    Provider(String),

    #[error("Backend error: {0}")]
    Backend(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Model not loaded: {0}")]
    ModelNotLoaded(String),

    #[error("Streaming error: {0}")]
    Streaming(String),

    #[error(transparent)]
    Tauri(#[from] tauri::Error),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error("HTTP error: {0}")]
    Http(String),
}

impl Serialize for Error {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

pub type Result<T> = std::result::Result<T, Error>;
