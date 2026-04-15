# tauri-plugin-ai

AI integration plugin for Tauri v2 apps. Provides a unified interface for cloud LLM providers and local model inference through a single, consistent API.

> **Status:** Alpha. The API is not yet stable and will change.

> **Starting a new app?** Check out [TASK (Tauri App Starter Kit)](https://github.com/youruser/tauri-app-starter-kit) — a complete desktop app skeleton with menus, settings, crash recovery, and more. This plugin integrates with TASK out of the box.

## Two APIs

**Chat / Completion** -- for LLMs (text generation, chat, tool use):

- Cloud providers: Anthropic (Claude), OpenAI (GPT), Ollama (local server)
- Local LLM: Candle backend (planned, feature-gated)

**General Inference** -- for any model type (classification, embeddings, depth estimation, audio):

- ONNX Runtime backend (feature-gated)
- Tensor-based I/O that works with any model architecture

## Installation

### Rust

Add to `src-tauri/Cargo.toml`:

```toml
[dependencies]
tauri-plugin-ai = { version = "0.1", features = ["cloud"] }
```

Register the plugin:

```rust
fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_ai::init())
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

### JavaScript / TypeScript

```bash
npm install tauri-plugin-ai-api
```

### Permissions

Add to `src-tauri/capabilities/default.json`:

```json
{
  "permissions": ["ai:default"]
}
```

The `ai:default` permission grants access to most commands. You can also grant individual permissions for fine-grained control. See the permissions table below for which are included by default.

## Quick Start: Cloud Completion

```typescript
import { setApiKey, complete } from 'tauri-plugin-ai-api';

// Set API key (stored in-memory for the session)
await setApiKey('anthropic', 'sk-ant-...');

const response = await complete({
  provider: 'anthropic',
  model: 'claude-sonnet-4-20250514',
  messages: [{ role: 'user', content: 'Explain quantum computing in one sentence.' }],
  max_tokens: 256,
});

console.log(response.content);
```

## Streaming

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
    onChunk: (chunk) => appendToUI(chunk.delta),
    onComplete: (usage) => console.log('Done.', usage),
    onError: (err) => console.error(err),
  },
);
```

Or collect the full response while still receiving tokens:

```typescript
import { streamToString } from 'tauri-plugin-ai-api';

const response = await streamToString(
  {
    provider: 'openai',
    model: 'gpt-4o',
    messages: [{ role: 'user', content: 'Hello!' }],
  },
  (token) => appendToUI(token),
);

console.log(response.content);
```

## ONNX Inference (Local Models)

Requires the `local-onnx` feature flag.

```toml
[dependencies]
tauri-plugin-ai = { version = "0.1", features = ["cloud", "local-onnx"] }
```

```typescript
import { loadModel, infer, tensorFromFloat32, tensorToFloat32 } from 'tauri-plugin-ai-api';

// Load an ONNX model
await loadModel({
  name: 'my-classifier',
  provider: 'onnx',         // backend name
  model_id: 'classifier-v1',
  model_path: '/path/to/model.onnx',
  options: {},
});

// Prepare input tensor (e.g. preprocessed image: 1x3x224x224)
const inputData = new Float32Array(1 * 3 * 224 * 224);
// ... fill with preprocessed pixel values ...

const output = await infer('onnx', 'my-classifier', {
  tensors: {
    input: tensorFromFloat32(inputData, [1, 3, 224, 224]),
  },
});

// Read output tensor
const logits = tensorToFloat32(output.tensors['logits']);
console.log('Predictions:', logits);
```

## API Key Management

API keys are stored **in-memory only** by default. They do not persist across app restarts.

```typescript
import { setApiKey, getApiKey } from 'tauri-plugin-ai-api';

// Set a key for the current session
await setApiKey('anthropic', 'sk-ant-...');

// Check if a key is loaded
const key = await getApiKey('anthropic');
if (!key) {
  console.log('No API key set for Anthropic');
}
```

### Persistent Keys with Keyring

For production apps, use [tauri-plugin-keyring](https://github.com/nicknisi/tauri-plugin-keyring) (or any OS keychain plugin) alongside this plugin. Load keys from the keychain on startup and pass them to `setApiKey`:

```typescript
import { setApiKey } from 'tauri-plugin-ai-api';
import { get as keyringGet } from 'tauri-plugin-keyring-api';

// On app startup: load key from OS keychain into memory
const key = await keyringGet('my-app', 'anthropic-api-key');
if (key) {
  await setApiKey('anthropic', key);
}
```

To save a new key:

```typescript
import { setApiKey } from 'tauri-plugin-ai-api';
import { set as keyringSet } from 'tauri-plugin-keyring-api';

// Save to both keychain (persistent) and memory (current session)
await keyringSet('my-app', 'anthropic-api-key', apiKey);
await setApiKey('anthropic', apiKey);
```

This separation keeps the AI plugin lightweight and avoids a hard dependency on any particular keychain implementation.

### Security: Proxy Pattern

API keys should stay in the Rust backend. The renderer calls `complete` or `stream`, and Rust reads the key internally -- the key never crosses the IPC boundary. The `get_api_key` command is intentionally **excluded** from `ai:default` permissions because it exposes raw keys to the renderer. Only add `allow-get-api-key` to your capabilities if the renderer absolutely needs the raw key value.

## Feature Flags

| Feature      | Default | Description                                        |
| ------------ | ------- | -------------------------------------------------- |
| `cloud`      | Yes     | Cloud providers (Anthropic, OpenAI, Ollama)        |
| `local-onnx` | No      | ONNX Runtime backend for local model inference     |
| `local-llm`  | No      | Candle-based local LLM inference (planned)         |

## API Reference

| Function           | Description                                    |
| ------------------ | ---------------------------------------------- |
| `complete(req)`    | Non-streaming LLM completion                   |
| `AIStream`         | Streaming completion manager (event-based)     |
| `streamToString()` | Stream and collect full response as string     |
| `cancelStream()`   | Cancel an active streaming request             |
| `setApiKey()`      | Set API key in memory for a provider           |
| `removeApiKey()`   | Remove the in-memory API key for a provider    |
| `getApiKey()`      | Get current in-memory API key for a provider   |
| `getProviders()`   | List registered provider names                 |
| `listModels()`     | List available models across all providers     |
| `loadModel()`      | Load a local model via a backend               |
| `unloadModel()`    | Unload a local model                           |
| `infer()`          | Run inference on a loaded model (any type)     |
| `listBackends()`   | List inference backends and their loaded models|

### Tensor Helpers

| Function              | Description                              |
| --------------------- | ---------------------------------------- |
| `tensorFromFloat32()` | Create TensorData from Float32Array      |
| `tensorFromUint8()`   | Create TensorData from Uint8Array        |
| `tensorToFloat32()`   | Extract Float32Array from TensorData     |

## Permissions

| Permission              | Command          | In `ai:default`? |
| ----------------------- | ---------------- | ---------------- |
| `allow-complete`        | `complete`       | Yes              |
| `allow-stream`          | `stream`         | Yes              |
| `allow-cancel-stream`   | `cancel_stream`  | Yes              |
| `allow-list-models`     | `list_models`    | Yes              |
| `allow-load-model`      | `load_model`     | Yes              |
| `allow-unload-model`    | `unload_model`   | Yes              |
| `allow-set-api-key`     | `set_api_key`    | Yes              |
| `allow-remove-api-key`  | `remove_api_key` | Yes              |
| `allow-get-api-key`     | `get_api_key`    | **No** (exposes secrets) |
| `allow-get-providers`   | `get_providers`  | Yes              |
| `allow-infer`           | `infer`          | Yes              |
| `allow-list-backends`   | `list_backends`  | Yes              |

## License

MIT
