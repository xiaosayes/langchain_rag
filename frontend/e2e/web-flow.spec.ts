import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";
import { spawn } from "node:child_process";
import type { ChildProcess } from "node:child_process";
import fs from "node:fs";
import net from "node:net";
import os from "node:os";
import path from "node:path";

const projectRoot = path.resolve(import.meta.dirname, "../..");
const frontendRoot = path.resolve(import.meta.dirname, "..");
const qdrantBinary = process.env.QDRANT_BINARY
  ? path.resolve(process.env.QDRANT_BINARY)
  : null;
const adminUsername = "e2e-admin";
const adminPassword = "e2e-correct-password";
const alphaText =
  "Conference travel policy: the hotel limit is 750 CNY per night. " +
  "Keep the itemized invoice for reimbursement.";
const betaText =
  "Equipment repair policy: call the internal hotline 4000 before " +
  "authorizing any external repair.";
const question =
  "Conference travel policy hotel limit 750 CNY per night " +
  "itemized invoice reimbursement";

let temporaryRoot: string | null = null;
let baseURL = "";
const children: ChildProcess[] = [];

async function freePort(): Promise<number> {
  const server = net.createServer();
  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => resolve());
  });
  const address = server.address();
  if (!address || typeof address === "string") throw new Error("No local port");
  const { port } = address;
  await new Promise<void>((resolve) => server.close(() => resolve()));
  return port;
}

function launch(
  label: string,
  command: string,
  args: string[],
  cwd: string,
  environment: NodeJS.ProcessEnv = {},
): ChildProcess {
  if (!temporaryRoot) throw new Error("Process harness is not initialized");
  const logPath = path.join(temporaryRoot, `${label}.log`);
  const log = fs.openSync(logPath, "w");
  const child = spawn(command, args, {
    cwd,
    env: { ...process.env, ...environment },
    stdio: ["ignore", log, log],
    windowsHide: true,
  });
  fs.closeSync(log);
  children.push(child);
  return child;
}

async function waitForHTTP(
  label: string,
  url: string,
  child: ChildProcess,
): Promise<void> {
  const deadline = Date.now() + 90_000;
  let lastError = "not attempted";
  while (Date.now() < deadline) {
    if (child.exitCode !== null) {
      throw new Error(`${label} exited with code ${child.exitCode}; see its log`);
    }
    try {
      const response = await fetch(url, { signal: AbortSignal.timeout(2000) });
      if (response.ok) return;
      lastError = `HTTP ${response.status}`;
    } catch (error) {
      lastError = error instanceof Error ? error.message : String(error);
    }
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  throw new Error(`${label} was not ready at ${url}: ${lastError}`);
}

async function waitForTask(page: Page, taskId: string): Promise<void> {
  await expect
    .poll(
      async () =>
        page.evaluate(async (id) => {
          const response = await fetch(`/api/tasks/${id}`);
          if (!response.ok) return `HTTP ${response.status}`;
          return (await response.json()).status;
        }, taskId),
      { timeout: 90_000 },
    )
    .toBe("succeeded");
}

test.skip(
  !qdrantBinary,
  "NOT RUN: set QDRANT_BINARY to the official Qdrant 1.15.4 executable",
);

test.beforeAll(async () => {
  if (!qdrantBinary) return;
  if (!fs.existsSync(qdrantBinary)) {
    throw new Error(`QDRANT_BINARY does not exist: ${qdrantBinary}`);
  }
  const frontendDist = path.join(frontendRoot, "dist", "index.html");
  if (!fs.existsSync(frontendDist)) {
    throw new Error("frontend/dist/index.html is missing; run the frontend build first");
  }
  const python = path.join(
    projectRoot,
    process.platform === "win32" ? ".venv/Scripts/python.exe" : ".venv/bin/python",
  );
  if (!fs.existsSync(python)) {
    throw new Error(`Project virtual environment is missing: ${python}; run uv sync`);
  }

  temporaryRoot = fs.mkdtempSync(path.join(os.tmpdir(), "rag-web-browser-"));
  const state = path.join(temporaryRoot, "state");
  const qdrantRoot = path.join(temporaryRoot, "qdrant");
  fs.mkdirSync(state, { recursive: true });
  fs.mkdirSync(qdrantRoot, { recursive: true });
  const qdrantHTTPPort = await freePort();
  const qdrantGRPCPort = await freePort();
  const apiPort = await freePort();
  const qdrantURL = `http://127.0.0.1:${qdrantHTTPPort}`;
  baseURL = `http://127.0.0.1:${apiPort}`;
  const qdrantConfig = path.join(qdrantRoot, "config.yaml");
  fs.writeFileSync(
    qdrantConfig,
    "storage:\n" +
      '  storage_path: "./storage"\n' +
      '  snapshots_path: "./snapshots"\n' +
      "service:\n" +
      '  host: "127.0.0.1"\n' +
      `  http_port: ${qdrantHTTPPort}\n` +
      `  grpc_port: ${qdrantGRPCPort}\n` +
      "telemetry_disabled: true\n",
  );
  const qdrant = launch(
    "qdrant",
    qdrantBinary,
    ["--config-path", qdrantConfig, "--disable-telemetry"],
    qdrantRoot,
  );
  await waitForHTTP("Qdrant", `${qdrantURL}/collections`, qdrant);

  const configPath = path.join(temporaryRoot, "settings.json");
  fs.writeFileSync(
    configPath,
    JSON.stringify({
      app_env: "test",
      model_provider: "fake",
      embedding: { dimension: 64 },
      qdrant: { url: qdrantURL },
      chunk: { parent_size: 180, child_size: 60, child_overlap: 10 },
      retrieval: { dense_top_k: 10, bm25_top_k: 10, final_parent_count: 4 },
      web: {
        raw_file_dir: path.join(state, "raw"),
        frontend_dist: path.join(frontendRoot, "dist"),
      },
    }),
  );
  const environment = {
    MODEL_PROVIDER: "fake",
    APP_ENV: "test",
    RAG_STATE_DIR: state,
    RAG_E2E_CONFIG: configPath,
    ADMIN_USERNAME: adminUsername,
    ADMIN_PASSWORD: adminPassword,
  };
  const entry = path.join(projectRoot, "tests", "e2e", "test_web_flow.py");
  const api = launch(
    "api",
    python,
    [entry, "--serve-api", "--port", String(apiPort)],
    projectRoot,
    environment,
  );
  await waitForHTTP("API", `${baseURL}/health/ready`, api);
  launch(
    "worker",
    python,
    [entry, "--serve-worker"],
    projectRoot,
    environment,
  );
});

test.afterAll(async () => {
  for (const child of children.reverse()) {
    if (child.exitCode === null && child.signalCode === null) child.kill();
  }
  await new Promise((resolve) => setTimeout(resolve, 1500));
  for (const child of children) {
    if (child.exitCode === null && child.signalCode === null) child.kill("SIGKILL");
  }
  if (temporaryRoot) {
    fs.rmSync(temporaryRoot, {
      recursive: true,
      force: true,
      maxRetries: 5,
      retryDelay: 200,
    });
  }
});

test("single-admin login, upload, cited chat, feedback, reindex, and delete", async ({
  page,
}) => {
  test.setTimeout(180_000);
  await page.goto(`${baseURL}/login`);
  await page.getByLabel("用户名").fill(adminUsername);
  await page.getByLabel("密码").fill(adminPassword);
  await page.getByRole("button", { name: "登录" }).click();
  await expect(page).toHaveURL(/\/chat$/);

  await page.getByRole("link", { name: "文档管理" }).click();
  for (const [name, text] of [
    ["alpha.txt", alphaText],
    ["beta.txt", betaText],
  ]) {
    await page.locator('input[type="file"]').setInputFiles({
      name,
      mimeType: "text/plain",
      buffer: Buffer.from(text),
    });
    const accepted = page.waitForResponse(
      (response) =>
        response.url().endsWith("/api/documents/upload") && response.status() === 202,
    );
    await page.getByRole("button", { name: "上传文档" }).click();
    await accepted;
  }
  const alphaRow = page.getByRole("row").filter({ hasText: "alpha.txt" });
  const betaRow = page.getByRole("row").filter({ hasText: "beta.txt" });
  await expect(alphaRow).toContainText("就绪", { timeout: 90_000 });
  await expect(betaRow).toContainText("就绪", { timeout: 90_000 });

  await page.getByRole("link", { name: "返回对话" }).click();
  await page.getByLabel("问题").fill(question);
  await page.getByRole("button", { name: "发送" }).click();
  const assistant = page.locator("article.message.assistant").last();
  await expect(assistant).toContainText("750 CNY", { timeout: 90_000 });
  await expect(assistant).toContainText("[S1]");
  await expect(assistant.locator(".citation-list")).toContainText("S1");
  await assistant.locator(".ant-collapse-header").first().click();
  await expect(assistant.locator(".citation-content")).toContainText("750 CNY");
  await assistant.getByRole("button", { name: "有帮助" }).click();
  await assistant.getByLabel("反馈评论").fill("Grounded browser answer");
  await assistant.getByRole("button", { name: "提交反馈" }).click();
  await expect(assistant.getByText("感谢反馈")).toBeVisible();

  await page.getByRole("link", { name: "文档管理" }).click();
  await alphaRow.getByRole("button", { name: "重新索引 alpha.txt" }).click();
  const reindexResponse = page.waitForResponse(
    (response) =>
      response.url().endsWith("/reindex") &&
      response.request().method() === "POST" &&
      response.status() === 202,
  );
  await page.getByRole("button", { name: "确认重新索引" }).click();
  const reindexTask = (await (await reindexResponse).json()).task_id as string;
  await waitForTask(page, reindexTask);
  await expect(alphaRow).toContainText("就绪");

  await betaRow.getByRole("button", { name: "删除 beta.txt" }).click();
  const deleteResponse = page.waitForResponse(
    (response) =>
      /\/api\/documents\/[^/]+$/.test(response.url()) &&
      response.request().method() === "DELETE" &&
      response.status() === 202,
  );
  await page.getByRole("button", { name: "确认删除" }).click();
  const deleteTask = (await (await deleteResponse).json()).task_id as string;
  await waitForTask(page, deleteTask);
  await expect(betaRow).toHaveCount(0);
  await expect(alphaRow).toHaveCount(1);
});
