use async_trait::async_trait;
use futures_util::StreamExt;
use reqwest::Client;
use serde_json::json;
use tokio::sync::mpsc;

use crate::config::{CompletionRequest, CompletionResponse, StreamChunk, Usage};
use crate::error::{Error, Result};
use crate::provider::Provider;

pub struct OpenAiProvider {
    client: Client,
    api_base: String,
}

impl OpenAiProvider {
    pub fn new(api_base: Option<String>) -> Self {
        Self {
            client: Client::new(),
            api_base: api_base.unwrap_or_else(|| "https://api.openai.com".to_string()),
        }
    }

    fn build_body(&self, request: &CompletionRequest, stream: bool) -> serde_json::Value {
        let model = request
            .model
            .clone()
            .unwrap_or_else(|| "gpt-4o".to_string());

        let messages: Vec<serde_json::Value> = request
            .messages
            .iter()
            .map(|m| {
                json!({
                    "role": m.role,
                    "content": m.content,
                })
            })
            .collect();

        let mut body = json!({
            "model": model,
            "messages": messages,
        });

        if let Some(max_tokens) = request.max_tokens {
            body["max_tokens"] = json!(max_tokens);
        }
        if let Some(temp) = request.temperature {
            body["temperature"] = json!(temp);
        }
        if stream {
            body["stream"] = json!(true);
        }
        if let Some(ref tools) = request.tools {
            body["tools"] = json!(tools);
        }

        body
    }
}

#[async_trait]
impl Provider for OpenAiProvider {
    fn name(&self) -> &str {
        "openai"
    }

    async fn complete(
        &self,
        request: CompletionRequest,
        api_key: &str,
    ) -> Result<CompletionResponse> {
        let body = self.build_body(&request, false);

        let resp = self
            .client
            .post(format!("{}/v1/chat/completions", self.api_base))
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| Error::Http(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(Error::Provider(format!(
                "OpenAI API error {}: {}",
                status, text
            )));
        }

        let data: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| Error::Provider(e.to_string()))?;

        let content = data["choices"]
            .as_array()
            .and_then(|arr| arr.first())
            .and_then(|choice| choice["message"]["content"].as_str())
            .unwrap_or("")
            .to_string();

        let finish_reason = data["choices"]
            .as_array()
            .and_then(|arr| arr.first())
            .and_then(|choice| choice["finish_reason"].as_str())
            .map(|s| s.to_string());

        let usage = data["usage"].as_object().map(|u| Usage {
            prompt_tokens: u
                .get("prompt_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32,
            completion_tokens: u
                .get("completion_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32,
            total_tokens: u
                .get("total_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32,
        });

        Ok(CompletionResponse {
            content,
            model: data["model"].as_str().unwrap_or("unknown").to_string(),
            provider: "openai".to_string(),
            usage,
            finish_reason,
        })
    }

    async fn stream(
        &self,
        request: CompletionRequest,
        api_key: &str,
        sender: mpsc::Sender<StreamChunk>,
    ) -> Result<()> {
        let body = self.build_body(&request, true);

        let resp = self
            .client
            .post(format!("{}/v1/chat/completions", self.api_base))
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| Error::Http(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(Error::Provider(format!(
                "OpenAI API error {}: {}",
                status, text
            )));
        }

        let mut buffer = String::new();
        let mut bytes_stream = resp.bytes_stream();

        while let Some(chunk_result) = bytes_stream.next().await {
            let chunk_bytes = chunk_result.map_err(|e| Error::Streaming(e.to_string()))?;
            buffer.push_str(&String::from_utf8_lossy(&chunk_bytes));

            while let Some(pos) = buffer.find("\n\n") {
                let event_block = buffer[..pos].to_string();
                buffer = buffer[pos + 2..].to_string();

                for line in event_block.lines() {
                    if let Some(data) = line.strip_prefix("data: ") {
                        if data.trim() == "[DONE]" {
                            let _ = sender
                                .send(StreamChunk {
                                    delta: String::new(),
                                    done: true,
                                    usage: None,
                                })
                                .await;
                            return Ok(());
                        }

                        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(data) {
                            let delta = parsed["choices"]
                                .as_array()
                                .and_then(|arr| arr.first())
                                .and_then(|c| c["delta"]["content"].as_str())
                                .unwrap_or("")
                                .to_string();

                            if !delta.is_empty() {
                                let _ = sender
                                    .send(StreamChunk {
                                        delta,
                                        done: false,
                                        usage: None,
                                    })
                                    .await;
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn available_models(&self) -> Vec<String> {
        vec![
            "gpt-4o".to_string(),
            "gpt-4o-mini".to_string(),
            "gpt-4-turbo".to_string(),
            "o1".to_string(),
            "o1-mini".to_string(),
        ]
    }
}
