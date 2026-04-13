# tauri-plugin-ai

AI integration plugin for Tauri v2 apps. Provides a unified interface for both cloud API providers (Anthropic, OpenAI, Ollama) and local inference backends (ONNX Runtime, Candle).

> **Status:** Early alpha. API will change.

## Features

- **Cloud providers:** Anthropic (Claude), OpenAI (GPT), Ollama (local server)
- **Local inference:** ONNX Runtime and Candle backends (feature-gated, coming soon)
- **Streaming:** Token-by-token streaming over Tauri events with request correlation
- **Model registry:** Discover and manage models across all providers

## Installation

### Rust side

```toml
# src-tauri/Cargo.toml
[dependencies]
tauri-plugin-ai = "0.1"
```

Register the plugin in your Tauri app:

```rust
fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_ai::init())
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

### Frontend side

```bash
npm install tauri-plugin-ai-api
```

### Permissions

Add to your `src-tauri/capabilities/default.json`:

```json
{
  "permissions": ["ai:default"]
}
```

## Feature Flags

| Feature      | Default | Description                              |
| ------------ | ------- | ---------------------------------------- |
| `cloud`      | Yes     | Cloud provider support (Anthropic, OpenAI, Ollama) |
| `local-onnx` | No      | ONNX Runtime inference backend           |
| `local-llm`  | No      | Candle-based local LLM inference         |

## Usage

### Non-streaming completion

```typescript
import { complete, setApiKey } from 'tauri-plugin-ai-api';

await setApiKey('anthropic', 'sk-ant-...');

const response = await complete({
  provider: 'anthropic',
  model: 'claude-sonnet-4-20250514',
  messages: [{ role: 'user', content: 'Hello!' }],
  max_tokens: 1024,
});

console.log(response.content);
```

### Streaming completion

```typescript
import { AIStream } from 'tauri-plugin-ai-api';

const stream = new AIStream();

await stream.start(
  {
    provider: 'anthropic',
    messages: [{ role: 'user', content: 'Write a haiku about Rust.' }],
    max_tokens: 256,
  },
  {
    onChunk: (chunk) => process.stdout.write(chunk.delta),
    onComplete: (usage) => console.log('\nDone.', usage),
    onError: (err) => console.error(err),
  },
);
```

Or collect the full response while still receiving tokens:

```typescript
import { streamToString } from 'tauri-plugin-ai-api';

const response = await streamToString(
  { provider: 'openai', messages: [{ role: 'user', content: 'Hi' }] },
  (token) => appendToUI(token),
);
```

### Model management

```typescript
import { listModels, getProviders, loadModel, unloadModel } from 'tauri-plugin-ai-api';

const providers = await getProviders();
const models = await listModels();
```

## API Reference

| Function | Description |
| --- | --- |
| `complete(request)` | Non-streaming completion |
| `new AIStream()` | Streaming completion manager |
| `streamToString(request, onToken?)` | Stream and collect full response |
| `listModels()` | List available models |
| `loadModel(config)` | Load a local model |
| `unloadModel(backend, name)` | Unload a local model |
| `setApiKey(provider, key)` | Set API key (in-memory) |
| `getProviders()` | List registered provider names |

## License

MIT
