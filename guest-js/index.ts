import { invoke } from '@tauri-apps/api/core';
import { listen, type UnlistenFn } from '@tauri-apps/api/event';

// ---------------------------------------------------------------------------
// Types — mirror the Rust structs in config.rs / models.rs / streaming.rs
// ---------------------------------------------------------------------------

export interface Message {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export interface CompletionRequest {
  provider?: string;
  model?: string;
  messages: Message[];
  max_tokens?: number;
  temperature?: number;
  stream?: boolean;
  tools?: unknown[];
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
  onComplete?: (usage?: Usage) => void;
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
    // Invoke the command first to get the request_id, then listen for events
    // filtered to that id.
    try {
      const requestId = await invoke<string>('plugin:ai|stream', { request });
      this.requestId = requestId;

      this.unlisten = await listen<StreamEvent>('ai:stream:chunk', (event) => {
        const { request_id, chunk } = event.payload;
        if (request_id !== this.requestId) return;

        if (chunk.done) {
          callbacks.onComplete?.(chunk.usage);
          this.stop();
        } else {
          callbacks.onChunk(chunk);
        }
      });
    } catch (e) {
      callbacks.onError?.(String(e));
    }
  }

  stop(): void {
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
        onComplete: (usage) => {
          stream.stop();
          resolve({
            content: parts.join(''),
            model: request.model ?? 'unknown',
            provider: request.provider ?? 'default',
            usage,
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

// ---------------------------------------------------------------------------
// Provider info
// ---------------------------------------------------------------------------

/** List registered provider names. */
export async function getProviders(): Promise<string[]> {
  return invoke<string[]>('plugin:ai|get_providers');
}

// ---------------------------------------------------------------------------
// Convenience namespace export
// ---------------------------------------------------------------------------

export const ai = {
  complete,
  streamToString,
  listModels,
  loadModel,
  unloadModel,
  setApiKey,
  getProviders,
  AIStream,
} as const;

export default ai;
