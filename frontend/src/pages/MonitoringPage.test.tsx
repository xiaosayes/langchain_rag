import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import App from "../App";
import type {
  MonitoringRunDetail,
  MonitoringRunSummary,
  MonitoringSummary,
} from "../api/monitoring";

const summary: MonitoringSummary = {
  hours: 24,
  total: 8,
  running: 1,
  succeeded: 3,
  failed: 2,
  cancelled: 1,
  degraded: 2,
  average_duration_ms: 125.5,
  p95_duration_ms: 800,
};

const runs: MonitoringRunSummary[] = [
  {
    id: "run-success",
    conversation_id: "conversation-1",
    message_id: "message-1",
    question: "How does grounding work?",
    status: "succeeded",
    degraded: false,
    created_at: "2026-01-01T00:00:00Z",
    ended_at: "2026-01-01T00:00:01Z",
    duration_ms: 1000,
  },
  {
    id: "run-failed",
    conversation_id: "conversation-2",
    message_id: "message-2",
    question: "Failed question",
    status: "failed",
    degraded: true,
    created_at: "2026-01-01T01:00:00Z",
    ended_at: "2026-01-01T01:00:02Z",
    duration_ms: 2000,
  },
  {
    id: "run-cancelled",
    conversation_id: null,
    message_id: null,
    question: "Cancelled question",
    status: "cancelled",
    degraded: false,
    created_at: "2026-01-01T02:00:00Z",
    ended_at: "2026-01-01T02:00:01Z",
    duration_ms: 1000,
  },
];

const failedDetail: MonitoringRunDetail = {
  ...runs[1],
  question: "Failed question <img src=x onerror=alert(1)>",
  retrieval_query: "retrieval query",
  prompt_messages: [
    { type: "system", content: "System <b>prompt</b>" },
    { type: "human", content: "Human prompt" },
  ],
  retrieved_text: "Retrieved <script>alert(1)</script> text",
  answer: "Partial answer <img src=x onerror=alert(2)>",
  generation_id: "generation-1",
  error: "Generation failed <strong>safely as text</strong>",
  events: [
    {
      id: 1,
      run_id: "run-failed",
      stage: "retrieval",
      status: "succeeded",
      started_at: "2026-01-01T01:00:00Z",
      ended_at: "2026-01-01T01:00:01Z",
      duration_ms: 1000,
      details: { source: "<i>retriever</i>" },
    },
    {
      id: 2,
      run_id: "run-failed",
      stage: "generation",
      status: "failed",
      started_at: "2026-01-01T01:00:01Z",
      ended_at: "2026-01-01T01:00:02Z",
      duration_ms: 1000,
      details: {},
    },
  ],
};

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function renderPage() {
  return render(
    <MemoryRouter initialEntries={["/monitoring"]}>
      <App />
    </MemoryRouter>,
  );
}

function successfulFetch(input: RequestInfo | URL): Promise<Response> {
  const url = String(input);
  if (url === "/api/auth/me") {
    return Promise.resolve(jsonResponse({ csrf_token: "csrf", expires_at: "tomorrow" }));
  }
  if (url === "/api/monitoring/summary?hours=24") {
    return Promise.resolve(jsonResponse(summary));
  }
  if (url === "/api/monitoring/runs?limit=50") {
    return Promise.resolve(jsonResponse(runs));
  }
  if (url === "/api/monitoring/runs?limit=50&status=failed") {
    return Promise.resolve(jsonResponse([runs[1]]));
  }
  if (url === "/api/monitoring/runs/run-failed") {
    return Promise.resolve(jsonResponse(failedDetail));
  }
  return Promise.reject(new Error(`Unexpected request: ${url}`));
}

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("local monitoring", () => {
  it("shows every summary metric, recent statuses, and safe run detail without feedback statistics", async () => {
    vi.stubGlobal("fetch", vi.fn(successfulFetch));
    renderPage();

    expect(await screen.findByRole("heading", { name: "本地运行监控" })).toBeInTheDocument();
    const summaryRegion = screen.getByRole("region", { name: "运行摘要" });
    expect(await within(summaryRegion).findByText("总请求")).toBeInTheDocument();
    for (const [label, value] of [
      ["总请求", "8"],
      ["成功", "3"],
      ["失败", "2"],
      ["已取消", "1"],
      ["运行中", "1"],
      ["降级", "2"],
      ["平均耗时", "125.5 ms"],
      ["P95 耗时", "800 ms"],
    ]) {
      expect(within(summaryRegion).getByText(label).parentElement).toHaveTextContent(value);
    }

    const recent = screen.getByRole("region", { name: "最近运行" });
    expect(within(recent).getByText("How does grounding work?")).toBeInTheDocument();
    expect(within(recent).getByText("成功")).toBeInTheDocument();
    expect(within(recent).getByText("失败")).toBeInTheDocument();
    expect(within(recent).getByText("已取消")).toBeInTheDocument();
    expect(within(recent).getByText("降级")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "查看详情 run-failed" }));
    expect(await screen.findByRole("region", { name: "运行详情" })).toBeInTheDocument();
    expect(screen.getByText(/Failed question <img src=x onerror=alert\(1\)>/)).toBeInTheDocument();
    expect(screen.getByText("System <b>prompt</b>")).toBeInTheDocument();
    expect(screen.getByText("Human prompt")).toBeInTheDocument();
    expect(screen.getByText("Retrieved <script>alert(1)</script> text")).toBeInTheDocument();
    expect(screen.getByText("Partial answer <img src=x onerror=alert(2)>")).toBeInTheDocument();
    expect(screen.getByText("Generation failed <strong>safely as text</strong>")).toBeInTheDocument();
    expect(screen.getByText("retrieval")).toBeInTheDocument();
    expect(screen.getByText("generation")).toBeInTheDocument();
    expect(screen.getByText(/<i>retriever<\/i>/)).toBeInTheDocument();
    expect(document.querySelector("img[src=x], script, i, b, strong")).not.toBeInTheDocument();

    expect(screen.queryByText(/点赞|点踩|喜欢|不喜欢|满意率/)).not.toBeInTheDocument();
    expect(screen.queryByText(/\b(?:likes?|dislikes?)\b/i)).not.toBeInTheDocument();
    expect(screen.getByRole("link", { name: "文档管理" })).toHaveAttribute("href", "/documents");
    expect(screen.getByRole("link", { name: "返回对话" })).toHaveAttribute("href", "/chat");
  });

  it("filters by status and refreshes only on demand without polling", async () => {
    const fetchMock = vi.fn(successfulFetch);
    vi.stubGlobal("fetch", fetchMock);
    renderPage();
    expect(await screen.findByText("How does grounding work?")).toBeInTheDocument();

    fireEvent.mouseDown(screen.getByRole("combobox", { name: "状态筛选" }));
    fireEvent.click(await screen.findByText("失败", {
      selector: ".ant-select-item-option-content",
    }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(
      "/api/monitoring/runs?limit=50&status=failed",
      expect.anything(),
    ));
    expect(screen.queryByText("How does grounding work?")).not.toBeInTheDocument();
    expect(screen.getByText("Failed question")).toBeInTheDocument();

    const callsBeforeWaiting = fetchMock.mock.calls.length;
    vi.useFakeTimers();
    await act(async () => {
      await vi.advanceTimersByTimeAsync(60_000);
    });
    expect(fetchMock).toHaveBeenCalledTimes(callsBeforeWaiting);
    vi.useRealTimers();

    const summaryCallsBeforeRefresh = fetchMock.mock.calls.filter(
      ([url]) => String(url) === "/api/monitoring/summary?hours=24",
    ).length;
    fireEvent.click(screen.getByRole("button", { name: "刷新" }));
    await waitFor(() => expect(
      fetchMock.mock.calls.filter(([url]) => String(url) === "/api/monitoring/summary?hours=24"),
    ).toHaveLength(summaryCallsBeforeRefresh + 1));
    expect(fetchMock.mock.calls.filter(
      ([url]) => String(url) === "/api/monitoring/runs?limit=50&status=failed",
    )).toHaveLength(2);
  });

  it("surfaces API failures and recovers with a manual refresh", async () => {
    let unavailable = true;
    const fetchMock = vi.fn((input: RequestInfo | URL) => {
      const url = String(input);
      if (unavailable && url.startsWith("/api/monitoring/")) {
        return Promise.resolve(jsonResponse({ detail: "Monitoring storage unavailable" }, 503));
      }
      return successfulFetch(input);
    });
    vi.stubGlobal("fetch", fetchMock);
    renderPage();

    expect(await screen.findByText("Monitoring storage unavailable")).toBeInTheDocument();
    expect(screen.queryByText("How does grounding work?")).not.toBeInTheDocument();

    unavailable = false;
    fireEvent.click(screen.getByRole("button", { name: "刷新" }));
    expect(await screen.findByText("How does grounding work?")).toBeInTheDocument();
    expect(screen.queryByText("Monitoring storage unavailable")).not.toBeInTheDocument();
  });

  it("protects the monitoring route with the existing authentication flow", async () => {
    const fetchMock = vi.fn((input: RequestInfo | URL) => {
      if (String(input) === "/api/auth/me") return Promise.resolve(jsonResponse({}, 401));
      return successfulFetch(input);
    });
    vi.stubGlobal("fetch", fetchMock);
    renderPage();

    expect(await screen.findByRole("heading", { name: "登录" })).toBeInTheDocument();
    expect(fetchMock.mock.calls.filter(([url]) => String(url).startsWith("/api/monitoring/"))).toHaveLength(0);
  });
});
