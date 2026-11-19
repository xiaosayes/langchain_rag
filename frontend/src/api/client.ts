export interface AuthSession {
  csrf_token: string;
  expires_at: string;
}

export interface Conversation {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface Citation {
  label: string;
  document_id: string;
  parent_id: string;
  text: string;
  metadata: Record<string, unknown>;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations: Citation[];
  degraded: boolean;
  generation_id: string | null;
  created_at: string;
  error?: string;
}

export interface ChatRequest {
  conversation_id?: string;
  question: string;
}

export interface StreamStart {
  conversation_id: string;
  message_id: string;
  generation_id: string;
  degraded: boolean;
}

export interface StreamToken {
  content: string;
}

export interface StreamCitations {
  citations: Citation[];
}

export interface StreamDone {
  conversation_id: string;
  message_id: string;
}

export interface StreamError {
  message: string;
  message_id?: string;
}

export interface StreamHandlers {
  onStart?: (event: StreamStart) => void;
  onToken?: (event: StreamToken) => void;
  onCitations?: (event: StreamCitations) => void;
  onDone?: (event: StreamDone) => void;
  onError?: (event: StreamError) => void;
}

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

const mutationMethods = new Set(["POST", "PUT", "PATCH", "DELETE"]);

async function errorFromResponse(response: Response): Promise<ApiError> {
  let message = `请求失败（${response.status}）`;
  try {
    const body = (await response.json()) as { detail?: string };
    if (typeof body.detail === "string") message = body.detail;
  } catch {
    // Keep the status-based message when the server did not return JSON.
  }
  return new ApiError(message, response.status);
}

export async function apiFetch<T>(
  input: string,
  options: RequestInit = {},
  csrfToken?: string | null,
): Promise<T> {
  const method = (options.method ?? "GET").toUpperCase();
  const headers = new Headers(options.headers);
  headers.set("Accept", "application/json");
  if (options.body !== undefined) headers.set("Content-Type", "application/json");
  if (mutationMethods.has(method) && csrfToken) {
    headers.set("X-CSRF-Token", csrfToken);
  }

  const response = await fetch(input, {
    ...options,
    method,
    headers,
    credentials: "same-origin",
  });
  if (!response.ok) throw await errorFromResponse(response);
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export function me(): Promise<AuthSession> {
  return apiFetch<AuthSession>("/api/auth/me");
}

export function login(username: string, password: string): Promise<AuthSession> {
  return apiFetch<AuthSession>("/api/auth/login", {
    method: "POST",
    body: JSON.stringify({ username, password }),
  });
}

export function logout(csrfToken: string): Promise<void> {
  return apiFetch<void>(
    "/api/auth/logout",
    { method: "POST" },
    csrfToken,
  );
}

function parseEvent(block: string): { event: string; data: string } | null {
  let event = "message";
  const data: string[] = [];
  for (const line of block.split(/\r\n|\n|\r/)) {
    if (!line || line.startsWith(":")) continue;
    const separator = line.indexOf(":");
    const field = separator === -1 ? line : line.slice(0, separator);
    const value = separator === -1 ? "" : line.slice(separator + 1).replace(/^ /, "");
    if (field === "event") event = value;
    if (field === "data") data.push(value);
  }
  return data.length > 0 ? { event, data: data.join("\n") } : null;
}

function dispatchEvent(event: string, data: string, handlers: StreamHandlers): void {
  const payload = JSON.parse(data) as unknown;
  switch (event) {
    case "start":
      handlers.onStart?.(payload as StreamStart);
      break;
    case "token":
      handlers.onToken?.(payload as StreamToken);
      break;
    case "citations":
      handlers.onCitations?.(payload as StreamCitations);
      break;
    case "done":
      handlers.onDone?.(payload as StreamDone);
      break;
    case "error":
      handlers.onError?.(payload as StreamError);
      break;
  }
}

export async function streamChat(
  request: ChatRequest,
  handlers: StreamHandlers,
  signal: AbortSignal,
  csrfToken?: string | null,
): Promise<void> {
  const headers = new Headers({
    Accept: "text/event-stream",
    "Content-Type": "application/json",
  });
  if (csrfToken) headers.set("X-CSRF-Token", csrfToken);
  const response = await fetch("/api/chat/stream", {
    method: "POST",
    headers,
    credentials: "same-origin",
    body: JSON.stringify(request),
    signal,
  });
  if (!response.ok) throw await errorFromResponse(response);
  if (!response.body) throw new Error("浏览器不支持流式响应");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  const consume = (block: string) => {
    const parsed = parseEvent(block);
    if (parsed) dispatchEvent(parsed.event, parsed.data, handlers);
  };

  try {
    while (true) {
      const { done, value } = await reader.read();
      buffer += decoder.decode(value, { stream: !done });
      let boundary = /\r\n\r\n|\n\n|\r\r/.exec(buffer);
      while (boundary?.index !== undefined) {
        consume(buffer.slice(0, boundary.index));
        buffer = buffer.slice(boundary.index + boundary[0].length);
        boundary = /\r\n\r\n|\n\n|\r\r/.exec(buffer);
      }
      if (done) break;
    }
    if (buffer.trim()) consume(buffer);
  } finally {
    reader.releaseLock();
  }
}
