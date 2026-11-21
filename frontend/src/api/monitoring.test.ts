import { afterEach, describe, expect, it, vi } from "vitest";
import {
  getMonitoringRun,
  getMonitoringRuns,
  getMonitoringSummary,
} from "./monitoring";

afterEach(() => vi.unstubAllGlobals());

describe("monitoring API client", () => {
  it("uses the authenticated same-origin API and encodes run filters and IDs", async () => {
    const fetchMock = vi.fn().mockImplementation(() => Promise.resolve(
      new Response(JSON.stringify({}), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    ));
    vi.stubGlobal("fetch", fetchMock);

    await getMonitoringSummary();
    await getMonitoringRuns("failed");
    await getMonitoringRun("run/with spaces");

    expect(fetchMock.mock.calls.map(([url]) => String(url))).toEqual([
      "/api/monitoring/summary?hours=24",
      "/api/monitoring/runs?limit=50&status=failed",
      "/api/monitoring/runs/run%2Fwith%20spaces",
    ]);
    for (const [, options] of fetchMock.mock.calls) {
      expect(options).toEqual(expect.objectContaining({ credentials: "same-origin" }));
    }
  });
});
