import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AuthProvider } from "../auth/AuthProvider";
import DocumentsPage from "./DocumentsPage";

interface TestDocument {
  id: string;
  filename: string;
  media_type: string;
  status: string;
  current_version_id: string | null;
  latest_version_id: string | null;
  parent_count: number;
  child_count: number;
  source_metadata: Record<string, unknown>;
  error: string | null;
  created_at: string;
  updated_at: string;
}

function documentFixture(update: Partial<TestDocument> = {}): TestDocument {
  return {
    id: "doc-1",
    filename: "handbook.txt",
    media_type: "text/plain",
    status: "ready",
    current_version_id: "version-1",
    latest_version_id: "version-1",
    parent_count: 2,
    child_count: 4,
    source_metadata: { kind: "document" },
    error: null,
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    ...update,
  };
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function renderPage() {
  return render(
    <MemoryRouter>
      <AuthProvider>
        <DocumentsPage />
      </AuthProvider>
    </MemoryRouter>,
  );
}

function fileInput(container: HTMLElement): HTMLInputElement {
  const input = container.querySelector<HTMLInputElement>('input[type="file"]');
  if (!input) throw new Error("Upload input not found");
  return input;
}

async function advanceTimers(milliseconds: number) {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(milliseconds);
  });
}

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("document management", () => {
  it("uploads a selected XLSX file as QA with multipart CSRF and validates type and size", async () => {
    let documents: TestDocument[] = [];
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url === "/api/auth/me") {
        return Promise.resolve(jsonResponse({ csrf_token: "csrf", expires_at: "tomorrow" }));
      }
      if (url === "/api/documents" && (!init?.method || init.method === "GET")) {
        return Promise.resolve(jsonResponse(documents));
      }
      if (url === "/api/documents/upload") {
        expect(init?.method).toBe("POST");
        expect(init?.credentials).toBe("same-origin");
        expect((init?.headers as Headers).get("X-CSRF-Token")).toBe("csrf");
        expect((init?.headers as Headers).get("Content-Type")).toBeNull();
        expect(init?.body).toBeInstanceOf(FormData);
        const body = init?.body as FormData;
        expect(body.get("kind")).toBe("qa");
        expect((body.get("file") as File).name).toBe("questions.xlsx");
        documents = [documentFixture({
          id: "doc-uploaded",
          filename: "questions.xlsx",
          status: "uploaded",
          current_version_id: null,
          latest_version_id: "version-uploaded",
          parent_count: 0,
          child_count: 0,
          source_metadata: { kind: "qa" },
        })];
        return Promise.resolve(jsonResponse({
          document_id: "doc-uploaded",
          version_id: "version-uploaded",
          task_id: "task-uploaded",
        }, 202));
      }
      return Promise.reject(new Error(`Unexpected request: ${url}`));
    });
    vi.stubGlobal("fetch", fetchMock);
    const { container } = renderPage();
    expect(await screen.findByRole("heading", { name: "文档管理" })).toBeInTheDocument();
    expect(screen.getByText(/TXT、DOCX、PDF、XLSX/)).toBeInTheDocument();
    expect(screen.getByText(/50 MB/)).toBeInTheDocument();

    fireEvent.change(fileInput(container), {
      target: { files: [new File(["bad"], "notes.md", { type: "text/markdown" })] },
    });
    expect(await screen.findByText("仅支持 TXT、DOCX、PDF、XLSX 文件")).toBeInTheDocument();
    const oversized = new File(["x"], "large.pdf", { type: "application/pdf" });
    Object.defineProperty(oversized, "size", { value: 50 * 1024 * 1024 + 1 });
    fireEvent.change(fileInput(container), { target: { files: [oversized] } });
    expect(await screen.findByText("文件不能超过 50 MB")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("radio", { name: "问答对（QA）" }));
    fireEvent.change(fileInput(container), {
      target: { files: [new File(["workbook"], "questions.xlsx", {
        type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
      })] },
    });
    expect(screen.getByText("questions.xlsx")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "上传文档" }));

    expect(await screen.findByText("排队中")).toBeInTheDocument();
    expect(screen.getByText("questions.xlsx")).toBeInTheDocument();
    expect(fetchMock.mock.calls.filter(([url]) => String(url) === "/api/documents/upload")).toHaveLength(1);
  });

  it("blocks a non-XLSX file selected in document mode after switching to QA", async () => {
    const fetchMock = vi.fn((input: RequestInfo | URL) => {
      const url = String(input);
      if (url === "/api/auth/me") {
        return Promise.resolve(jsonResponse({ csrf_token: "csrf", expires_at: "tomorrow" }));
      }
      if (url === "/api/documents") return Promise.resolve(jsonResponse([]));
      return Promise.reject(new Error(`Unexpected request: ${url}`));
    });
    vi.stubGlobal("fetch", fetchMock);
    const { container } = renderPage();
    expect(await screen.findByRole("heading", { name: "文档管理" })).toBeInTheDocument();

    fireEvent.change(fileInput(container), {
      target: { files: [new File(["plain text"], "notes.txt", { type: "text/plain" })] },
    });
    expect(screen.getByText("notes.txt")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("radio", { name: "问答对（QA）" }));
    fireEvent.click(screen.getByRole("button", { name: "上传文档" }));

    expect(await screen.findByText("问答对（QA）需上传 XLSX 文件")).toBeInTheDocument();
    expect(fetchMock.mock.calls.filter(([url]) => String(url) === "/api/documents/upload")).toHaveLength(0);
  });

  it("shows ready and sanitized failed states without exposing metadata paths and provides the download URL", async () => {
    const documents = [
      documentFixture(),
      documentFixture({
        id: "doc-failed",
        filename: "broken.pdf",
        media_type: "application/pdf",
        status: "failed",
        current_version_id: "version-accepted",
        latest_version_id: "version-failed",
        parent_count: 0,
        child_count: 0,
        error: "Operation failed",
        source_metadata: { path: "/private/SECRET_PATH", token: "SECRET_TOKEN" },
      }),
    ];
    vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL) => {
      const url = String(input);
      if (url === "/api/auth/me") {
        return Promise.resolve(jsonResponse({ csrf_token: "csrf", expires_at: "tomorrow" }));
      }
      if (url === "/api/documents") return Promise.resolve(jsonResponse(documents));
      return Promise.reject(new Error(`Unexpected request: ${url}`));
    }));
    const { container } = renderPage();

    expect(await screen.findByText("就绪")).toBeInTheDocument();
    expect(screen.getByText("失败")).toBeInTheDocument();
    expect(screen.getByText("Operation failed")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "下载 handbook.txt" })).toHaveAttribute(
      "href",
      "/api/documents/doc-1/content",
    );
    expect(screen.getByRole("link", { name: "下载 broken.pdf" })).toHaveAttribute(
      "href",
      "/api/documents/doc-failed/content",
    );
    expect(container).not.toHaveTextContent("/private/SECRET_PATH");
    expect(container).not.toHaveTextContent("SECRET_TOKEN");
  });

  it("requires confirmation before reindexing and deleting", async () => {
    const documents = [
      documentFixture(),
      documentFixture({ id: "doc-2", filename: "remove.docx" }),
    ];
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url === "/api/auth/me") {
        return Promise.resolve(jsonResponse({ csrf_token: "csrf", expires_at: "tomorrow" }));
      }
      if (url === "/api/documents" && (!init?.method || init.method === "GET")) {
        return Promise.resolve(jsonResponse(documents));
      }
      if (url === "/api/documents/doc-1/reindex") {
        expect(init?.method).toBe("POST");
        expect((init?.headers as Headers).get("X-CSRF-Token")).toBe("csrf");
        return Promise.resolve(jsonResponse({
          document_id: "doc-1",
          version_id: "version-1",
          task_id: "task-reindex",
        }, 202));
      }
      if (url === "/api/documents/doc-2") {
        expect(init?.method).toBe("DELETE");
        expect((init?.headers as Headers).get("X-CSRF-Token")).toBe("csrf");
        return Promise.resolve(jsonResponse({
          document_id: "doc-2",
          version_id: "version-1",
          task_id: "task-delete",
        }, 202));
      }
      return Promise.reject(new Error(`Unexpected request: ${url}`));
    });
    vi.stubGlobal("fetch", fetchMock);
    renderPage();
    expect(await screen.findByText("handbook.txt")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "重新索引 handbook.txt" }));
    expect(await screen.findByText("重新索引 handbook.txt？")).toBeInTheDocument();
    expect(fetchMock).not.toHaveBeenCalledWith(
      "/api/documents/doc-1/reindex",
      expect.anything(),
    );
    fireEvent.click(screen.getByRole("button", { name: "确认重新索引" }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(
      "/api/documents/doc-1/reindex",
      expect.objectContaining({ method: "POST" }),
    ));

    fireEvent.click(screen.getByRole("button", { name: "删除 remove.docx" }));
    expect(await screen.findByText("删除 remove.docx？")).toBeInTheDocument();
    expect(fetchMock).not.toHaveBeenCalledWith(
      "/api/documents/doc-2",
      expect.objectContaining({ method: "DELETE" }),
    );
    fireEvent.click(screen.getByRole("button", { name: "确认删除" }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(
      "/api/documents/doc-2",
      expect.objectContaining({ method: "DELETE" }),
    ));
  });

  it("polls active tasks every two seconds, displays task errors, and stops at a terminal state", async () => {
    let taskStatus = "queued";
    let documents = [documentFixture()];
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url === "/api/auth/me") {
        return Promise.resolve(jsonResponse({ csrf_token: "csrf", expires_at: "tomorrow" }));
      }
      if (url === "/api/documents" && (!init?.method || init.method === "GET")) {
        if (taskStatus === "failed") {
          documents = [documentFixture({ status: "failed", error: "Operation failed" })];
        }
        return Promise.resolve(jsonResponse(documents));
      }
      if (url === "/api/documents/doc-1/reindex") {
        return Promise.resolve(jsonResponse({
          document_id: "doc-1",
          version_id: "version-1",
          task_id: "task-1",
        }, 202));
      }
      if (url === "/api/tasks/task-1") {
        taskStatus = taskStatus === "queued" ? "running" : "failed";
        return Promise.resolve(jsonResponse({
          id: "task-1",
          kind: "reindex",
          status: taskStatus,
          attempts: taskStatus === "failed" ? 2 : 1,
          error: taskStatus === "failed" ? "Operation failed" : null,
        }));
      }
      return Promise.reject(new Error(`Unexpected request: ${url}`));
    });
    vi.stubGlobal("fetch", fetchMock);
    renderPage();
    expect(await screen.findByText("handbook.txt")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "重新索引 handbook.txt" }));
    fireEvent.click(await screen.findByRole("button", { name: "确认重新索引" }));
    expect(await screen.findByText("排队中")).toBeInTheDocument();

    await new Promise((resolve) => setTimeout(resolve, 1900));
    expect(fetchMock.mock.calls.filter(([url]) => String(url) === "/api/tasks/task-1")).toHaveLength(0);
    expect(await screen.findByText("处理中", {}, { timeout: 1000 })).toBeInTheDocument();
    expect(await screen.findByText("Operation failed", {}, { timeout: 2500 })).toBeInTheDocument();
    expect(screen.getByText("失败")).toBeInTheDocument();
    const terminalCalls = fetchMock.mock.calls.filter(([url]) => String(url) === "/api/tasks/task-1").length;
    await new Promise((resolve) => setTimeout(resolve, 2200));
    expect(fetchMock.mock.calls.filter(([url]) => String(url) === "/api/tasks/task-1")).toHaveLength(terminalCalls);
  });

  it("does not poll idle documents and clears active polling on unmount", async () => {
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url === "/api/auth/me") {
        return Promise.resolve(jsonResponse({ csrf_token: "csrf", expires_at: "tomorrow" }));
      }
      if (url === "/api/documents" && (!init?.method || init.method === "GET")) {
        return Promise.resolve(jsonResponse([documentFixture()]));
      }
      if (url === "/api/documents/doc-1/reindex") {
        return Promise.resolve(jsonResponse({
          document_id: "doc-1",
          version_id: "version-1",
          task_id: "task-1",
        }, 202));
      }
      if (url === "/api/tasks/task-1") {
        return Promise.resolve(jsonResponse({
          id: "task-1",
          kind: "reindex",
          status: "running",
          attempts: 1,
          error: null,
        }));
      }
      return Promise.reject(new Error(`Unexpected request: ${url}`));
    });
    vi.stubGlobal("fetch", fetchMock);
    const view = renderPage();
    expect(await screen.findByText("handbook.txt")).toBeInTheDocument();

    vi.useFakeTimers();
    await advanceTimers(4000);
    expect(fetchMock.mock.calls.filter(([url]) => String(url).startsWith("/api/tasks/"))).toHaveLength(0);
    vi.useRealTimers();

    fireEvent.click(screen.getByRole("button", { name: "重新索引 handbook.txt" }));
    fireEvent.click(await screen.findByRole("button", { name: "确认重新索引" }));
    expect(await screen.findByText("排队中")).toBeInTheDocument();
    expect(await screen.findByText("处理中", {}, { timeout: 3000 })).toBeInTheDocument();
    const activeCalls = fetchMock.mock.calls.filter(([url]) => String(url) === "/api/tasks/task-1").length;

    view.unmount();
    await new Promise((resolve) => setTimeout(resolve, 2200));
    expect(fetchMock.mock.calls.filter(([url]) => String(url) === "/api/tasks/task-1")).toHaveLength(activeCalls);
  });
});
