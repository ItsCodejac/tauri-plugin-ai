import { invoke } from '@tauri-apps/api/core';
import { listen, type UnlistenFn } from '@tauri-apps/api/event';

// ---------------------------------------------------------------------------
// Types — mirror the Rust structs in config.rs / models.rs / streaming.rs
// ---------------------------------------------------------------------------

export type Role = 'system' | 'user' | 'assistant' | 'tool';

export interface TextContentPart {
  type: 'text';
  text: string;
}

export interface ImageContentPart {
  type: 'image';
  data: string;       // base64 encoded
  media_type: string;  // e.g. "image/png"
}

export type ContentPart = TextContentPart | ImageContentPart;

/** Message content — either a plain string or structured multi-modal parts. */
export type Content = string | ContentPart[];

export interface Message {
  role: Role;
  content: Content;
}

export interface CompletionRequest {
  provider?: string;
  model?: string;
  messages: Message[];
  max_tokens?: number;
  temperature?: number;
  stream?: boolean;
  tools?: unknown[];
  /** Request timeout in seconds. Defaults to 60 for complete, 120 for stream. */
  timeout_secs?: number;
}

export interface CompletionResponse {
  content: string;
  model: string;
  provider: string;
  usage?: Usage;
  finish_reason?: string;
}

export interface Usage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

export interface StreamChunk {
  delta: string;
  done: boolean;
  usage?: Usage;
  finish_reason?: string;
}

/** Wrapper emitted by the Rust side on the `ai:stream:chunk` event. */
interface StreamEvent {
  request_id: string;
  chunk: StreamChunk;
}

export interface ModelInfo {
  name: string;
  provider: string;
  is_local: boolean;
  is_loaded: boolean;
}

export interface ModelConfig {
  name: string;
  provider: string;
  model_id: string;
  model_path?: string;
  options?: Record<string, unknown>;
}

// ---------------------------------------------------------------------------
// Non-streaming completion
// ---------------------------------------------------------------------------

/**
 * Send a completion request and wait for the full response.
 *
 * ```ts
 * const res = await complete({
 *   provider: 'anthropic',
 *   model: 'claude-sonnet-4-20250514',
 *   messages: [{ role: 'user', content: 'Hello!' }],
 *   max_tokens: 1024,
 * });
 * console.log(res.content);
 * ```
 */
export async function complete(request: CompletionRequest): Promise<CompletionResponse> {
  return invoke<CompletionResponse>('plugin:ai|complete', { request });
}

// ---------------------------------------------------------------------------
// Streaming completion
// ---------------------------------------------------------------------------

export interface StreamCallbacks {
  onChunk: (chunk: StreamChunk) => void;
  onError?: (error: string) => void;
  onComplete?: (usage?: Usage, finishReason?: string) => void;
}

/**
 * Manages a streaming completion request.
 *
 * ```ts
 * const stream = new AIStream();
 * await stream.start(
 *   { provider: 'anthropic', messages: [{ role: 'user', content: 'Hi' }] },
 *   {
 *     onChunk: (c) => process.stdout.write(c.delta),
 *     onComplete: (u) => console.log('\nDone', u),
 *   },
 * );
 * // later: stream.stop();
 * ```
 */
export class AIStream {
  private unlisten: UnlistenFn | null = null;
  private requestId: string | null = null;

  async start(request: CompletionRequest, callbacks: StreamCallbacks): Promise<void> {
    // Listen FIRST so no events are missed between invoke resolving and
    // the listener being established.
    this.unlisten = await listen<StreamEvent>('ai:stream:chunk', (event) => {
      const { request_id, chunk } = event.payload;
      if (request_id !== this.requestId) return;

      if (chunk.done) {
        callbacks.onComplete?.(chunk.usage, chunk.finish_reason);
        this.stop();
      } else {
        callbacks.onChunk(chunk);
      }
    });

    try {
      // THEN start the stream — events will be caught by the listener above
      this.requestId = await invoke<string>('plugin:ai|stream', { request });
    } catch (e) {
      this.stop();
      callbacks.onError?.(String(e));
    }
  }

  /** Cancel the stream. Signals the Rust side to drop the sender, stopping the provider. */
  stop(): void {
    if (this.requestId) {
      // Fire-and-forget cancel on the Rust side
      invoke('plugin:ai|cancel_stream', { requestId: this.requestId }).catch(() => {
        // Ignore errors — stream may already be finished
      });
    }
    this.unlisten?.();
    this.unlisten = null;
    this.requestId = null;
  }
}

/**
 * Stream a completion and collect the full text.
 *
 * Optionally receive each token as it arrives via `onToken`.
 */
export async function streamToString(
  request: CompletionRequest,
  onToken?: (token: string) => void,
): Promise<CompletionResponse> {
  return new Promise((resolve, reject) => {
    const parts: string[] = [];
    const stream = new AIStream();

    stream
      .start(request, {
        onChunk: (chunk) => {
          parts.push(chunk.delta);
          onToken?.(chunk.delta);
        },
        onComplete: (usage, finishReason) => {
          stream.stop();
          resolve({
            content: parts.join(''),
            model: request.model ?? 'unknown',
            provider: request.provider ?? 'default',
            usage,
            finish_reason: finishReason,
          });
        },
        onError: (error) => {
          stream.stop();
          reject(new Error(error));
        },
      })
      .catch(reject);
  });
}

// ---------------------------------------------------------------------------
// Model management
// ---------------------------------------------------------------------------

/** List all available models across all providers and backends. */
export async function listModels(): Promise<ModelInfo[]> {
  return invoke<ModelInfo[]>('plugin:ai|list_models');
}

/** Load a local model via the appropriate backend. */
export async function loadModel(config: ModelConfig): Promise<void> {
  return invoke('plugin:ai|load_model', { config });
}

/** Unload a previously loaded local model. */
export async function unloadModel(backend: string, name: string): Promise<void> {
  return invoke('plugin:ai|unload_model', { backend, name });
}

// ---------------------------------------------------------------------------
// API key management
// ---------------------------------------------------------------------------

/** Set an API key for a cloud provider (stored in-memory only). */
export async function setApiKey(provider: string, key: string): Promise<void> {
  return invoke('plugin:ai|set_api_key', { provider, key });
}

/**
 * Get the current in-memory API key for a provider.
 *
 * Returns `null` if no key is set. Useful for checking whether
 * a key has been loaded (e.g. from keyring) before making requests.
 */
export async function getApiKey(provider: string): Promise<string | null> {
  return invoke<string | null>('plugin:ai|get_api_key', { provider });
}

// ---------------------------------------------------------------------------
// Provider info
// ---------------------------------------------------------------------------

/** List registered provider names. */
export async function getProviders(): Promise<string[]> {
  return invoke<string[]>('plugin:ai|get_providers');
}

// ---------------------------------------------------------------------------
// Stream cancellation
// ---------------------------------------------------------------------------

/** Cancel an active streaming request by its request ID. */
export async function cancelStream(requestId: string): Promise<void> {
  return invoke('plugin:ai|cancel_stream', { requestId });
}

// ---------------------------------------------------------------------------
// General inference (non-LLM models)
// ---------------------------------------------------------------------------

/**
 * Tensor data for model input/output.
 *
 * TODO: The `data: number[]` encoding is slow for large tensors because JSON
 * serialization of number arrays is expensive. A future improvement should add
 * a `base64` field as an alternative encoding (base64-encoded raw bytes) for
 * performance-critical use cases. The Rust side would decode base64 when present.
 * Both options should be kept so users can choose: number[] for small/debug
 * tensors, base64 for production performance.
 */
export interface TensorData {
  /** Shape (e.g. [1, 3, 224, 224]) */
  shape: number[];
  /** Data type: "f32", "f16", "i32", "u8", etc. */
  dtype: string;
  /** Raw data as byte array */
  data: number[];
}

/** Input for general model inference. */
export interface InferenceInput {
  /** Named tensor inputs (model-specific, e.g. "pixel_values", "input_ids") */
  tensors: Record<string, TensorData>;
  /** Optional model-specific parameters */
  params?: Record<string, unknown>;
}

/** Output from general model inference. */
export interface InferenceOutput {
  /** Named tensor outputs (model-specific, e.g. "logits", "embeddings") */
  tensors: Record<string, TensorData>;
  /** Optional metadata (timing, model info) */
  metadata?: Record<string, unknown>;
}

/** Backend info (name + loaded models). */
export interface BackendInfo {
  name: string;
  loaded_models: string[];
}

/**
 * Run inference on a loaded model.
 *
 * This is the generic inference API for any model type:
 * image classifiers, embedding models, audio processors, etc.
 * For LLM chat, use `complete()` or `AIStream` instead.
 */
export async function infer(
  backend: string,
  model: string,
  input: InferenceInput,
): Promise<InferenceOutput> {
  return invoke<InferenceOutput>('plugin:ai|infer', { backend, model, input });
}

/** List all inference backends and their loaded models. */
export async function listBackends(): Promise<BackendInfo[]> {
  return invoke<BackendInfo[]>('plugin:ai|list_backends');
}

// ---------------------------------------------------------------------------
// Tensor helpers
// ---------------------------------------------------------------------------

/**
 * Create a TensorData from a Float32Array.
 *
 * TODO: For large tensors, consider using `tensorFromFloat32Base64` (not yet
 * implemented) which encodes as base64 instead of a JSON number array for
 * significantly better serialization performance.
 */
export function tensorFromFloat32(data: Float32Array, shape: number[]): TensorData {
  // Use byteOffset and byteLength to correctly handle views into larger buffers
  const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  return {
    shape,
    dtype: 'f32',
    data: Array.from(bytes),
  };
}

/** Create a TensorData from a Uint8Array (e.g. image bytes). */
export function tensorFromUint8(data: Uint8Array, shape: number[]): TensorData {
  return {
    shape,
    dtype: 'u8',
    data: Array.from(data),
  };
}

/** Extract a Float32Array from a TensorData output. */
export function tensorToFloat32(tensor: TensorData): Float32Array {
  return new Float32Array(new Uint8Array(tensor.data).buffer);
}

// ---------------------------------------------------------------------------
// Convenience namespace export
// ---------------------------------------------------------------------------

export const ai = {
  // Chat/Completion (LLMs)
  complete,
  streamToString,
  AIStream,

  // Model management
  listModels,
  loadModel,
  unloadModel,
  setApiKey,
  getApiKey,
  getProviders,

  // Stream cancellation
  cancelStream,

  // General inference (any model)
  infer,
  listBackends,

  // Tensor helpers
  tensorFromFloat32,
  tensorFromUint8,
  tensorToFloat32,
} as const;

export default ai;
