use async_trait::async_trait;
use futures_util::StreamExt;
use reqwest::Client;
use serde_json::json;
use tokio::sync::mpsc;

use crate::config::{CompletionRequest, CompletionResponse, StreamChunk, Usage};
use crate::error::{Error, Result};
use crate::provider::Provider;

pub struct AnthropicProvider {
    client: Client,
    api_base: String,
}

impl AnthropicProvider {
    pub fn new(api_base: Option<String>) -> Self {
        Self {
            client: Client::new(),
            api_base: api_base.unwrap_or_else(|| "https://api.anthropic.com".to_string()),
        }
    }

    fn build_body(&self, request: &CompletionRequest, stream: bool) -> serde_json::Value {
        let model = request
            .model
            .clone()
            .unwrap_or_else(|| "claude-sonnet-4-20250514".to_string());

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
            "max_tokens": request.max_tokens.unwrap_or(1024),
        });

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
impl Provider for AnthropicProvider {
    fn name(&self) -> &str {
        "anthropic"
    }

    async fn complete(
        &self,
        request: CompletionRequest,
        api_key: &str,
    ) -> Result<CompletionResponse> {
        let body = self.build_body(&request, false);

        let resp = self
            .client
            .post(format!("{}/v1/messages", self.api_base))
            .header("x-api-key", api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| Error::Http(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(Error::Provider(format!(
                "Anthropic API error {}: {}",
                status, text
            )));
        }

        let data: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| Error::Provider(e.to_string()))?;

        let content = data["content"]
            .as_array()
            .and_then(|arr| arr.first())
            .and_then(|block| block["text"].as_str())
            .unwrap_or("")
            .to_string();

        let usage = data["usage"].as_object().map(|u| Usage {
            prompt_tokens: u
                .get("input_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32,
            completion_tokens: u
                .get("output_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32,
            total_tokens: (u
                .get("input_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0)
                + u.get("output_tokens")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0)) as u32,
        });

        Ok(CompletionResponse {
            content,
            model: data["model"].as_str().unwrap_or("unknown").to_string(),
            provider: "anthropic".to_string(),
            usage,
            finish_reason: data["stop_reason"].as_str().map(|s| s.to_string()),
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
            .post(format!("{}/v1/messages", self.api_base))
            .header("x-api-key", api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| Error::Http(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(Error::Provider(format!(
                "Anthropic API error {}: {}",
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
                        if data == "[DONE]" {
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
                            let event_type = parsed["type"].as_str().unwrap_or("");

                            match event_type {
                                "content_block_delta" => {
                                    if let Some(text) = parsed["delta"]["text"].as_str() {
                                        let _ = sender
                                            .send(StreamChunk {
                                                delta: text.to_string(),
                                                done: false,
                                                usage: None,
                                            })
                                            .await;
                                    }
                                }
                                "message_delta" => {
                                    let usage =
                                        parsed["usage"].as_object().map(|u| Usage {
                                            prompt_tokens: 0,
                                            completion_tokens: u
                                                .get("output_tokens")
                                                .and_then(|v| v.as_u64())
                                                .unwrap_or(0)
                                                as u32,
                                            total_tokens: u
                                                .get("output_tokens")
                                                .and_then(|v| v.as_u64())
                                                .unwrap_or(0)
                                                as u32,
                                        });
                                    let _ = sender
                                        .send(StreamChunk {
                                            delta: String::new(),
                                            done: true,
                                            usage,
                                        })
                                        .await;
                                }
                                _ => {}
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
            "claude-sonnet-4-20250514".to_string(),
            "claude-haiku-4-20250414".to_string(),
            "claude-opus-4-20250514".to_string(),
        ]
    }
}
