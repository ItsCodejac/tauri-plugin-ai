use std::time::Duration;

use async_trait::async_trait;
use futures_util::StreamExt;
use reqwest::Client;
use serde_json::json;
use tokio::sync::mpsc;

use crate::config::{
    CompletionRequest, CompletionResponse, Content, ContentPart, StreamChunk, Usage,
};
use crate::error::{Error, Result};
use crate::provider::Provider;

pub struct OllamaProvider {
    client: Client,
    api_base: String,
}

impl OllamaProvider {
    pub fn new(api_base: Option<String>) -> Self {
        Self {
            client: Client::new(),
            api_base: api_base.unwrap_or_else(|| "http://localhost:11434".to_string()),
        }
    }

    fn build_body(&self, request: &CompletionRequest, stream: bool) -> serde_json::Value {
        let model = request
            .model
            .clone()
            .unwrap_or_else(|| "llama3.2".to_string());

        let messages: Vec<serde_json::Value> = request
            .messages
            .iter()
            .map(|m| {
                let mut msg = json!({
                    "role": m.role,
                });
                match &m.content {
                    Content::Text(s) => {
                        msg["content"] = json!(s);
                    }
                    Content::Parts(parts) => {
                        // Ollama uses "content" for text and "images" for base64 images
                        let text: String = parts
                            .iter()
                            .filter_map(|p| match p {
                                ContentPart::Text { text } => Some(text.as_str()),
                                _ => None,
                            })
                            .collect::<Vec<_>>()
                            .join("");
                        let images: Vec<&str> = parts
                            .iter()
                            .filter_map(|p| match p {
                                ContentPart::Image { data, .. } => Some(data.as_str()),
                                _ => None,
                            })
                            .collect();
                        msg["content"] = json!(text);
                        if !images.is_empty() {
                            msg["images"] = json!(images);
                        }
                    }
                }
                msg
            })
            .collect();

        let mut body = json!({
            "model": model,
            "messages": messages,
            "stream": stream,
        });

        let mut options = serde_json::Map::new();
        if let Some(temp) = request.temperature {
            options.insert("temperature".to_string(), json!(temp));
        }
        if let Some(max_tokens) = request.max_tokens {
            options.insert("num_predict".to_string(), json!(max_tokens));
        }
        if !options.is_empty() {
            body["options"] = json!(options);
        }

        body
    }
}

#[async_trait]
impl Provider for OllamaProvider {
    fn name(&self) -> &str {
        "ollama"
    }

    async fn complete(
        &self,
        request: CompletionRequest,
        _api_key: &str,
    ) -> Result<CompletionResponse> {
        let timeout = Duration::from_secs(request.timeout_secs.unwrap_or(60) as u64);
        let body = self.build_body(&request, false);

        let resp = self
            .client
            .post(format!("{}/api/chat", self.api_base))
            .header("Content-Type", "application/json")
            .timeout(timeout)
            .json(&body)
            .send()
            .await
            .map_err(|e| Error::Http(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(Error::Provider(format!(
                "Ollama API error {}: {}",
                status, text
            )));
        }

        let data: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| Error::Provider(e.to_string()))?;

        let content = data["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string();

        let usage = data.get("eval_count").map(|eval| Usage {
            prompt_tokens: data
                .get("prompt_eval_count")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32,
            completion_tokens: eval.as_u64().unwrap_or(0) as u32,
            total_tokens: (data
                .get("prompt_eval_count")
                .and_then(|v| v.as_u64())
                .unwrap_or(0)
                + eval.as_u64().unwrap_or(0)) as u32,
        });

        Ok(CompletionResponse {
            content,
            model: data["model"].as_str().unwrap_or("unknown").to_string(),
            provider: "ollama".to_string(),
            usage,
            finish_reason: Some("stop".to_string()),
        })
    }

    async fn stream(
        &self,
        request: CompletionRequest,
        _api_key: &str,
        sender: mpsc::Sender<StreamChunk>,
    ) -> Result<()> {
        let timeout = Duration::from_secs(request.timeout_secs.unwrap_or(120) as u64);
        let body = self.build_body(&request, true);

        let resp = self
            .client
            .post(format!("{}/api/chat", self.api_base))
            .header("Content-Type", "application/json")
            .timeout(timeout)
            .json(&body)
            .send()
            .await
            .map_err(|e| Error::Http(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(Error::Provider(format!(
                "Ollama API error {}: {}",
                status, text
            )));
        }

        let mut buffer = String::new();
        let mut bytes_stream = resp.bytes_stream();

        while let Some(chunk_result) = bytes_stream.next().await {
            let chunk_bytes = chunk_result.map_err(|e| Error::Streaming(e.to_string()))?;
            buffer.push_str(&String::from_utf8_lossy(&chunk_bytes));

            while let Some(pos) = buffer.find('\n') {
                let line = buffer[..pos].to_string();
                buffer = buffer[pos + 1..].to_string();

                if line.trim().is_empty() {
                    continue;
                }

                if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&line) {
                    let done = parsed["done"].as_bool().unwrap_or(false);
                    let content = parsed["message"]["content"]
                        .as_str()
                        .unwrap_or("")
                        .to_string();

                    let usage = if done {
                        parsed.get("eval_count").map(|eval| Usage {
                            prompt_tokens: parsed
                                .get("prompt_eval_count")
                                .and_then(|v| v.as_u64())
                                .unwrap_or(0)
                                as u32,
                            completion_tokens: eval.as_u64().unwrap_or(0) as u32,
                            total_tokens: (parsed
                                .get("prompt_eval_count")
                                .and_then(|v| v.as_u64())
                                .unwrap_or(0)
                                + eval.as_u64().unwrap_or(0))
                                as u32,
                        })
                    } else {
                        None
                    };

                    if sender
                        .send(StreamChunk {
                            delta: content,
                            done,
                            usage,
                            finish_reason: if done {
                                Some("stop".to_string())
                            } else {
                                None
                            },
                            error: None,
                        })
                        .await
                        .is_err()
                    {
                        // Receiver dropped (cancelled), stop streaming
                        return Ok(());
                    }

                    if done {
                        return Ok(());
                    }
                }
            }
        }

        Ok(())
    }

    /// Ollama models are dynamic — they depend on what's pulled locally.
    /// Use the Ollama `/api/tags` endpoint to discover available models at runtime.
    fn available_models(&self) -> Vec<String> {
        Vec::new()
    }
}
