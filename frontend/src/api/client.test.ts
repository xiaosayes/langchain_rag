import { afterEach, describe, expect, it, vi } from "vitest";
import { apiFetch, streamChat } from "./client";

function sseResponse(chunks: string[]): Response {
  const encoder = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
      controller.close();
    },
  });
  return new Response(stream, {
    status: 200,
    headers: { "Content-Type": "text/event-stream" },
  });
}

afterEach(() => vi.unstubAllGlobals());

describe("API client", () => {
  it("uses same-origin cookies and adds CSRF only to mutations", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ ok: true }), {
        headers: { "Content-Type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await apiFetch("/api/example", { method: "POST", body: "{}" }, "csrf-only-in-memory");

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/example",
      expect.objectContaining({
        method: "POST",
        credentials: "same-origin",
        headers: expect.any(Headers),
      }),
    );
    const headers = fetchMock.mock.calls[0][1].headers as Headers;
    expect(headers.get("X-CSRF-Token")).toBe("csrf-only-in-memory");
  });

  it("parses fragmented SSE events and multiline JSON data in order", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        sseResponse([
          "event: start\ndata: {\"conversation_id\":\"c1\",",
          "\"message_id\":\"m1\",\"generation_id\":\"g1\",\"degraded\":true}\n\n",
          "event: token\ndata: {\"content\":\"hello \"}\n\nevent: token\n",
          "data: {\"content\":\"world\"}\n\n",
          "event: citations\ndata: {\"citations\":[{\"label\":\"S1\",\"document_id\":\"d1\",\"parent_id\":\"p1\",\"text\":\"source\",\"metadata\":{}}]}\n\n",
          "event: done\ndata: {\"conversation_id\":\"c1\",\"message_id\":\"m1\"}\n\n",
        ]),
      ),
    );
    const events: string[] = [];
    const tokens: string[] = [];

    await streamChat(
      { question: "question" },
      {
        onStart: (event) => events.push(`start:${event.conversation_id}`),
        onToken: (event) => {
          events.push("token");
          tokens.push(event.content);
        },
        onCitations: (event) => events.push(`citations:${event.citations[0]?.label}`),
        onDone: (event) => events.push(`done:${event.message_id}`),
      },
      new AbortController().signal,
      "csrf-only-in-memory",
    );

    expect(tokens).toEqual(["hello ", "world"]);
    expect(events).toEqual(["start:c1", "token", "token", "citations:S1", "done:m1"]);
  });

  it("delivers server error events without rendering their text as HTML", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        sseResponse(["event: error\ndata: {\"message\":\"<img src=x onerror=alert(1)>\"}\n\n"]),
      ),
    );
    const onError = vi.fn();
    await streamChat(
      { question: "question" },
      { onError },
      new AbortController().signal,
    );
    expect(onError).toHaveBeenCalledWith({ message: "<img src=x onerror=alert(1)>" });
  });
});
