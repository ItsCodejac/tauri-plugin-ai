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
                let content_value = match &m.content {
                    Content::Text(s) => json!(s),
                    Content::Parts(parts) => {
                        let blocks: Vec<serde_json::Value> = parts
                            .iter()
                            .map(|p| match p {
                                ContentPart::Text { text } => json!({
                                    "type": "text",
                                    "text": text,
                                }),
                                ContentPart::Image { data, media_type } => json!({
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": data,
                                    },
                                }),
                            })
                            .collect();
                        json!(blocks)
                    }
                };
                json!({
                    "role": m.role,
                    "content": content_value,
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
        let timeout = Duration::from_secs(request.timeout_secs.unwrap_or(60) as u64);
        let body = self.build_body(&request, false);

        let resp = self
            .client
            .post(format!("{}/v1/messages", self.api_base))
            .header("x-api-key", api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .timeout(timeout)
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
        let timeout = Duration::from_secs(request.timeout_secs.unwrap_or(120) as u64);
        let body = self.build_body(&request, true);

        let resp = self
            .client
            .post(format!("{}/v1/messages", self.api_base))
            .header("x-api-key", api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .timeout(timeout)
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
        // Track input_tokens from message_start so we can include it in the final usage
        let mut input_tokens: u32 = 0;

        while let Some(chunk_result) = bytes_stream.next().await {
            let chunk_bytes = chunk_result.map_err(|e| Error::Streaming(e.to_string()))?;
            buffer.push_str(&String::from_utf8_lossy(&chunk_bytes));

            while let Some(pos) = buffer.find("\n\n") {
                let event_block = buffer[..pos].to_string();
                buffer = buffer[pos + 2..].to_string();

                for line in event_block.lines() {
                    if let Some(data) = line.strip_prefix("data: ") {
                        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(data) {
                            let event_type = parsed["type"].as_str().unwrap_or("");

                            match event_type {
                                "message_start" => {
                                    // Capture input_tokens from the message_start event
                                    if let Some(tokens) = parsed["message"]["usage"]
                                        .get("input_tokens")
                                        .and_then(|v| v.as_u64())
                                    {
                                        input_tokens = tokens as u32;
                                    }
                                }
                                "content_block_delta" => {
                                    if let Some(text) = parsed["delta"]["text"].as_str() {
                                        if sender
                                            .send(StreamChunk {
                                                delta: text.to_string(),
                                                done: false,
                                                usage: None,
                                                finish_reason: None,
                                            })
                                            .await
                                            .is_err()
                                        {
                                            // Receiver dropped (cancelled), stop streaming
                                            return Ok(());
                                        }
                                    }
                                }
                                "message_delta" => {
                                    let output_tokens = parsed["usage"]
                                        .get("output_tokens")
                                        .and_then(|v| v.as_u64())
                                        .unwrap_or(0)
                                        as u32;

                                    let usage = Some(Usage {
                                        prompt_tokens: input_tokens,
                                        completion_tokens: output_tokens,
                                        total_tokens: input_tokens + output_tokens,
                                    });

                                    let finish_reason = parsed["delta"]["stop_reason"]
                                        .as_str()
                                        .map(|s| s.to_string());

                                    let _ = sender
                                        .send(StreamChunk {
                                            delta: String::new(),
                                            done: true,
                                            usage,
                                            finish_reason,
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
