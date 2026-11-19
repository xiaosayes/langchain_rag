import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "../App";

const conversation = {
  id: "c1",
  title: "Existing conversation",
  created_at: "2026-01-01T00:00:00Z",
  updated_at: "2026-01-01T00:00:00Z",
};

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function streamResponse(
  chunks: string[],
  signal?: AbortSignal,
  onAbort?: () => void,
): Response {
  const encoder = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
      if (onAbort) {
        signal?.addEventListener("abort", () => {
          onAbort();
          controller.error(new DOMException("Aborted", "AbortError"));
        });
      } else {
        controller.close();
      }
    },
  });
  return new Response(stream, {
    status: 200,
    headers: { "Content-Type": "text/event-stream" },
  });
}

function renderAt(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <App />
    </MemoryRouter>,
  );
}

beforeEach(() => {
  window.localStorage.setItem("sentinel", "unchanged");
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("authentication and chat", () => {
  it("redirects unauthenticated users to login and never persists credentials", async () => {
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url === "/api/auth/me") return Promise.resolve(jsonResponse({}, 401));
      if (url === "/api/auth/login") {
        expect(init?.credentials).toBe("same-origin");
        expect(init?.headers).toBeInstanceOf(Headers);
        expect((init?.headers as Headers).get("X-CSRF-Token")).toBeNull();
        return Promise.resolve(jsonResponse({ csrf_token: "memory-csrf", expires_at: "tomorrow" }));
      }
      if (url === "/api/sessions") return Promise.resolve(jsonResponse([conversation]));
      if (url === "/api/sessions/c1/messages") return Promise.resolve(jsonResponse([]));
      return Promise.reject(new Error(`Unexpected request: ${url}`));
    });
    vi.stubGlobal("fetch", fetchMock);

    renderAt("/chat/c1");
    expect(await screen.findByRole("heading", { name: "登录" })).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("用户名"), { target: { value: "admin" } });
    fireEvent.change(screen.getByLabelText("密码"), { target: { value: "secret-password" } });
    fireEvent.click(screen.getByRole("button", { name: "登录" }));

    expect(await screen.findByText("Existing conversation")).toBeInTheDocument();
    expect(window.localStorage.getItem("sentinel")).toBe("unchanged");
    expect(window.localStorage.length).toBe(1);
    expect(JSON.stringify(window.localStorage)).not.toContain("memory-csrf");
    expect(JSON.stringify(window.localStorage)).not.toContain("secret-password");
  });

  it("streams ordered tokens, citations, degraded state and submits feedback", async () => {
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url === "/api/auth/me") {
        return Promise.resolve(jsonResponse({ csrf_token: "memory-csrf", expires_at: "tomorrow" }));
      }
      if (url === "/api/sessions") return Promise.resolve(jsonResponse([conversation]));
      if (url === "/api/sessions/c1/messages") return Promise.resolve(jsonResponse([]));
      if (url === "/api/chat/stream") {
        expect(init?.credentials).toBe("same-origin");
        expect((init?.headers as Headers).get("X-CSRF-Token")).toBe("memory-csrf");
        return Promise.resolve(
          streamResponse([
            "event: start\ndata: {\"conversation_id\":\"c1\",\"message_id\":\"m1\",\"generation_id\":\"g1\",\"degraded\":true}\n\n",
            "event: token\ndata: {\"content\":\"Grounded \"}\n\n",
            "event: token\ndata: {\"content\":\"answer[S1]\"}\n\n",
            "event: citations\ndata: {\"citations\":[{\"label\":\"S1\",\"document_id\":\"DOC-1\",\"parent_id\":\"P1\",\"text\":\"Source excerpt\",\"metadata\":{\"page\":3}}]}\n\n",
            "event: done\ndata: {\"conversation_id\":\"c1\",\"message_id\":\"m1\"}\n\n",
          ]),
        );
      }
      if (url === "/api/feedback") {
        expect(init?.method).toBe("POST");
        expect(JSON.parse(String(init?.body))).toEqual({ message_id: "m1", score: 1, comment: "Helpful" });
        expect((init?.headers as Headers).get("X-CSRF-Token")).toBe("memory-csrf");
        return Promise.resolve(jsonResponse({ id: "f1" }));
      }
      if (url === "/api/auth/logout") {
        expect((init?.headers as Headers).get("X-CSRF-Token")).toBe("memory-csrf");
        return Promise.resolve(new Response(null, { status: 204 }));
      }
      return Promise.reject(new Error(`Unexpected request: ${url}`));
    });
    vi.stubGlobal("fetch", fetchMock);
    renderAt("/chat/c1");
    await screen.findByRole("link", { name: "Existing conversation" });

    fireEvent.change(screen.getByLabelText("问题"), { target: { value: "What is grounded?" } });
    fireEvent.click(screen.getByRole("button", { name: "发送" }));

    expect(await screen.findByText(/Grounded answer/)).toBeInTheDocument();
    expect(screen.getByText("降级模式")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /S1 DOC-1/ }));
    expect(await screen.findByText("Source excerpt")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /引用元数据/ }));
    expect(await screen.findByText(/"page": 3/)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "有帮助" }));
    fireEvent.change(screen.getByLabelText("反馈评论"), { target: { value: "Helpful" } });
    fireEvent.click(screen.getByRole("button", { name: "提交反馈" }));
    expect(await screen.findByText("感谢反馈")).toBeInTheDocument();
    expect(window.localStorage.length).toBe(1);

    fireEvent.click(screen.getByRole("button", { name: /退出登录/ }));
    expect(await screen.findByRole("heading", { name: "登录" })).toBeInTheDocument();
    expect(window.localStorage.length).toBe(1);
  });

  it("renders Markdown without allowing raw HTML", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn((input: RequestInfo | URL) => {
        const url = String(input);
        if (url === "/api/auth/me") return Promise.resolve(jsonResponse({ csrf_token: "csrf", expires_at: "tomorrow" }));
        if (url === "/api/sessions") return Promise.resolve(jsonResponse([conversation]));
        if (url === "/api/sessions/c1/messages") {
          return Promise.resolve(jsonResponse([{
            id: "stored-1",
            role: "assistant",
            content: "**Safe Markdown** <img src=x onerror=alert(1)>",
            citations: [],
            degraded: false,
            generation_id: "g1",
            created_at: "2026-01-01T00:00:00Z",
          }]));
        }
        return Promise.reject(new Error(`Unexpected request: ${url}`));
      }),
    );
    renderAt("/chat/c1");

    expect(await screen.findByText("Safe Markdown", { selector: "strong" })).toBeInTheDocument();
    expect(screen.getByText(/<img src=x onerror=alert\(1\)>/)).toBeInTheDocument();
    expect(document.querySelector("img[src=x]")).not.toBeInTheDocument();
  });

  it("stops and aborts an active stream", async () => {
    const aborted = vi.fn();
    vi.stubGlobal(
      "fetch",
      vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input);
        if (url === "/api/auth/me") return Promise.resolve(jsonResponse({ csrf_token: "csrf", expires_at: "tomorrow" }));
        if (url === "/api/sessions") return Promise.resolve(jsonResponse([conversation]));
        if (url === "/api/sessions/c1/messages") return Promise.resolve(jsonResponse([]));
        if (url === "/api/chat/stream") {
          return Promise.resolve(
            streamResponse(
              ["event: start\ndata: {\"conversation_id\":\"c1\",\"message_id\":\"m1\",\"generation_id\":\"g1\",\"degraded\":false}\n\nevent: token\ndata: {\"content\":\"Partial\"}\n\n"],
              init?.signal ?? undefined,
              aborted,
            ),
          );
        }
        return Promise.reject(new Error(`Unexpected request: ${url}`));
      }),
    );
    renderAt("/chat/c1");
    await screen.findByRole("link", { name: "Existing conversation" });

    fireEvent.change(screen.getByLabelText("问题"), { target: { value: "Long answer" } });
    fireEvent.click(screen.getByRole("button", { name: "发送" }));
    const stop = await screen.findByRole("button", { name: "停止" });
    fireEvent.click(stop);

    await waitFor(() => expect(aborted).toHaveBeenCalledOnce());
    expect(await screen.findByText("已停止，回答可能不完整")).toBeInTheDocument();
    expect(screen.getByLabelText("问题")).toBeEnabled();
  });
});
