import { apiFetch } from "./client";

export type MonitoringRunStatus = "running" | "succeeded" | "failed" | "cancelled";

export interface MonitoringSummary {
  hours: number;
  total: number;
  running: number;
  succeeded: number;
  failed: number;
  cancelled: number;
  degraded: number;
  average_duration_ms: number;
  p95_duration_ms: number;
}

export interface MonitoringRunSummary {
  id: string;
  conversation_id: string | null;
  message_id: string | null;
  question: string;
  status: MonitoringRunStatus;
  degraded: boolean;
  created_at: string;
  ended_at: string | null;
  duration_ms: number | null;
}

export interface MonitoringPromptMessage {
  type: string;
  content: unknown;
}

export interface MonitoringRunEvent {
  id: number;
  run_id: string;
  stage: string;
  status: string;
  started_at: string;
  ended_at: string;
  duration_ms: number;
  details: Record<string, unknown>;
}

export interface MonitoringRunDetail extends MonitoringRunSummary {
  retrieval_query: string | null;
  prompt_messages: MonitoringPromptMessage[];
  retrieved_text: string | null;
  answer: string | null;
  generation_id: string | null;
  error: string | null;
  events: MonitoringRunEvent[];
}

export function getMonitoringSummary(hours = 24): Promise<MonitoringSummary> {
  return apiFetch<MonitoringSummary>(`/api/monitoring/summary?hours=${hours}`);
}

export function getMonitoringRuns(
  status?: MonitoringRunStatus,
): Promise<MonitoringRunSummary[]> {
  const parameters = new URLSearchParams({ limit: "50" });
  if (status) parameters.set("status", status);
  return apiFetch<MonitoringRunSummary[]>(`/api/monitoring/runs?${parameters}`);
}

export function getMonitoringRun(runId: string): Promise<MonitoringRunDetail> {
  return apiFetch<MonitoringRunDetail>(
    `/api/monitoring/runs/${encodeURIComponent(runId)}`,
  );
}
