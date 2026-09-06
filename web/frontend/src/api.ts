/**
 * Client for the FastAPI backend.
 *
 * The chat endpoint is a POST that responds with a Server-Sent Events stream,
 * so the browser's built-in `EventSource` is not an option (it only does GET).
 * Instead we read the `fetch` response body as a stream and parse SSE frames
 * ourselves - which is about twenty lines and removes a dependency.
 */

export type Role = "system" | "user" | "assistant";

export interface ChatMessage {
  role: Role;
  content: string;
}

export interface GenerationOptions {
  max_new_tokens?: number;
  temperature?: number;
  top_k?: number;
  top_p?: number;
  repetition_penalty?: number;
  seed?: number | null;
}

export interface HealthInfo {
  status: string;
  engine: string;
  info: Record<string, unknown>;
}

export type StreamEvent =
  | { type: "token"; text: string }
  | { type: "done" }
  | { type: "error"; message: string };

export async function getHealth(): Promise<HealthInfo> {
  const res = await fetch("/health");
  if (!res.ok) throw new Error(`health check failed: ${res.status}`);
  return res.json();
}

/**
 * Stream a reply, yielding one event per SSE frame.
 *
 * `signal` lets the caller abort mid-generation (the Stop button); the server
 * notices the disconnect and stops generating.
 */
export async function* streamChat(
  messages: ChatMessage[],
  options: GenerationOptions = {},
  signal?: AbortSignal,
): AsyncGenerator<StreamEvent> {
  const res = await fetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages, ...options }),
    signal,
  });

  if (!res.ok || !res.body) {
    const detail = await res.text().catch(() => "");
    yield { type: "error", message: `request failed (${res.status}) ${detail}`.trim() };
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // Frames are separated by a blank line; a partial frame stays in the buffer
    // until the rest of it arrives.
    let split: number;
    while ((split = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, split);
      buffer = buffer.slice(split + 2);
      const event = parseFrame(frame);
      if (event) yield event;
    }
  }
}

function parseFrame(frame: string): StreamEvent | null {
  let eventName = "message";
  const dataLines: string[] = [];
  for (const line of frame.split("\n")) {
    if (line.startsWith("event:")) eventName = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
  }
  if (dataLines.length === 0) return null;

  let payload: any;
  try {
    payload = JSON.parse(dataLines.join("\n"));
  } catch {
    return null;
  }

  if (eventName === "error") return { type: "error", message: payload.message ?? "unknown error" };
  if (eventName === "done") return { type: "done" };
  if (typeof payload.token === "string") return { type: "token", text: payload.token };
  return null;
}
