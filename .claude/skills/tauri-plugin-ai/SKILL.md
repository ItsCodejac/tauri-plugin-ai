---
name: tauri-plugin-ai
description: Expert guidance for using tauri-plugin-ai in Tauri v2 apps. Use when integrating AI features -- cloud LLM providers (Anthropic, OpenAI, Ollama), local model inference via ONNX Runtime, streaming completions, API key management, or model loading/unloading. Activate when working with AI integration, ML inference, or LLM APIs in Tauri desktop apps.
---

# tauri-plugin-ai Skill

## Installation and Registration

### Cargo.toml

```toml
[dependencies]
tauri-plugin-ai = { version = "0.1", features = ["cloud"] }
# Add "local-onnx" for ONNX Runtime support
```

### Register in main.rs

```rust
fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_ai::init())
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

### Frontend

```bash
npm install tauri-plugin-ai-api
```

### Permissions (src-tauri/capabilities/default.json)

```json
{ "permissions": ["ai:default"] }
```

## Cloud Completion

```typescript
import { setApiKey, complete } from 'tauri-plugin-ai-api';

await setApiKey('anthropic', 'sk-ant-...');

const res = await complete({
  provider: 'anthropic',
  model: 'claude-sonnet-4-20250514',
  messages: [{ role: 'user', content: 'Hello!' }],
  max_tokens: 1024,
});
```

Supported providers: `anthropic`, `openai`, `ollama`. Each is auto-registered when the `cloud` feature is enabled.

## Streaming

```typescript
import { AIStream } from 'tauri-plugin-ai-api';

const stream = new AIStream();
await stream.start(
  { provider: 'openai', messages: [{ role: 'user', content: 'Hi' }] },
  {
    onChunk: (c) => appendToUI(c.delta),
    onComplete: (usage) => console.log('Done', usage),
    onError: (err) => console.error(err),
  },
);
```

Or collect the full text:

```typescript
import { streamToString } from 'tauri-plugin-ai-api';

const response = await streamToString(
  { provider: 'anthropic', messages: [...] },
  (token) => appendToUI(token),
);
```

Streaming uses Tauri events (`ai:stream:chunk`) with request-correlated IDs.

## ONNX Inference

Enable with `features = ["local-onnx"]`.

```typescript
import { loadModel, infer, tensorFromFloat32, tensorToFloat32, unloadModel } from 'tauri-plugin-ai-api';

await loadModel({
  name: 'my-model',
  provider: 'onnx',
  model_id: 'classifier-v1',
  model_path: '/path/to/model.onnx',
  options: {},
});

const output = await infer('onnx', 'my-model', {
  tensors: { input: tensorFromFloat32(inputData, [1, 3, 224, 224]) },
});

const logits = tensorToFloat32(output.tensors['logits']);

await unloadModel('onnx', 'my-model');
```

## Security: Proxy Pattern

API keys should stay in the Rust backend. The renderer calls `complete` or `stream`, and Rust reads the key internally -- the key never crosses the IPC boundary. The `getApiKey` command is intentionally excluded from `ai:default` permissions because it exposes raw keys to the renderer. Only enable `allow-get-api-key` if the renderer absolutely needs the raw key value.

## API Key Management

Keys are in-memory only (no persistence by default). For persistent keys, use tauri-plugin-keyring alongside:

```typescript
import { setApiKey, getApiKey } from 'tauri-plugin-ai-api';

// Check if key exists in memory
const key = await getApiKey('anthropic');

// On startup: load from keyring, then set in memory
import { get as keyringGet } from 'tauri-plugin-keyring-api';
const saved = await keyringGet('my-app', 'anthropic-api-key');
if (saved) await setApiKey('anthropic', saved);
```

## Feature Flags

| Feature      | Default | Description                                    |
| ------------ | ------- | ---------------------------------------------- |
| `cloud`      | Yes     | Anthropic, OpenAI, Ollama providers            |
| `local-onnx` | No      | ONNX Runtime inference backend                 |
| `local-llm`  | No      | Candle-based local LLM inference (planned)     |

## Adding a Custom Provider

Implement the `Provider` trait in `src/provider.rs`:

```rust
use async_trait::async_trait;
use tauri_plugin_ai::provider::Provider;
use tauri_plugin_ai::config::*;
use tauri_plugin_ai::error::Result;
use tokio::sync::mpsc;

pub struct MyProvider { /* ... */ }

#[async_trait]
impl Provider for MyProvider {
    fn name(&self) -> &str { "my-provider" }

    async fn complete(&self, request: CompletionRequest, api_key: &str) -> Result<CompletionResponse> {
        // Make HTTP request to your API
        todo!()
    }

    async fn stream(&self, request: CompletionRequest, api_key: &str, sender: mpsc::Sender<StreamChunk>) -> Result<()> {
        // Stream chunks via sender
        todo!()
    }

    fn available_models(&self) -> Vec<String> {
        vec!["my-model-v1".into()]
    }
}
```

Register it before building the app (requires a custom `init` or using the builder pattern).

## Adding a Custom Backend

Implement the `InferenceBackend` trait in `src/backend.rs`:

```rust
use tauri_plugin_ai::backend::{InferenceBackend, InferenceInput, InferenceOutput};
use tauri_plugin_ai::config::ModelConfig;
use tauri_plugin_ai::error::Result;

pub struct MyBackend { /* ... */ }

impl InferenceBackend for MyBackend {
    fn name(&self) -> &str { "my-backend" }
    fn load_model(&mut self, config: &ModelConfig) -> Result<()> { todo!() }
    fn unload_model(&mut self, name: &str) -> Result<()> { todo!() }
    fn is_loaded(&self, name: &str) -> bool { false }
    fn loaded_models(&self) -> Vec<String> { vec![] }
    fn infer(&self, model: &str, input: InferenceInput) -> Result<InferenceOutput> { todo!() }
}
```

## Architecture Notes

- **Main process only**: All AI operations run in the Tauri main process (Rust side). The frontend communicates via IPC commands.
- **State**: `AiState` wraps a `ModelRegistry` behind a `tokio::sync::Mutex`. All commands acquire this lock.
- **Streaming**: Uses `mpsc` channels internally; a spawned task forwards chunks as Tauri events with a `request_id` for correlation.
- **Providers vs Backends**: Providers handle cloud LLM APIs (chat/completion). Backends handle local model inference (tensor I/O).

See README.md for the full API reference and detailed examples.
