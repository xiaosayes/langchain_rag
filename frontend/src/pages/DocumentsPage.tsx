import {
  DeleteOutlined,
  DownloadOutlined,
  FileTextOutlined,
  LogoutOutlined,
  MonitorOutlined,
  SyncOutlined,
} from "@ant-design/icons";
import {
  Alert,
  Button,
  Card,
  Layout,
  Popconfirm,
  Progress,
  Space,
  Table,
  Tag,
  Typography,
} from "antd";
import type { ColumnsType } from "antd/es/table";
import { useCallback, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { apiFetch } from "../api/client";
import { useAuth } from "../auth/AuthProvider";
import UploadPanel from "../components/UploadPanel";
import type { UploadAccepted } from "../components/UploadPanel";

interface DocumentOut {
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

interface TaskOut {
  id: string;
  kind: string;
  status: string;
  attempts: number;
  error: string | null;
}

interface DocumentTaskAccepted {
  document_id: string;
  version_id: string | null;
  task_id: string;
}

const ACTIVE_TASK_STATUSES = new Set(["queued", "running"]);
const STATUS_PRESENTATION: Record<string, { color: string; label: string }> = {
  queued: { color: "default", label: "排队中" },
  running: { color: "processing", label: "处理中" },
  ready: { color: "success", label: "就绪" },
  failed: { color: "error", label: "失败" },
};

function errorMessage(caught: unknown, fallback: string): string {
  return caught instanceof Error ? caught.message : fallback;
}

export default function DocumentsPage() {
  const { csrfToken, signOut } = useAuth();
  const [documents, setDocuments] = useState<DocumentOut[]>([]);
  const [tasks, setTasks] = useState<Record<string, TaskOut>>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [mutation, setMutation] = useState<string | null>(null);
  const tasksRef = useRef(tasks);
  tasksRef.current = tasks;

  const loadDocuments = useCallback(async () => {
    try {
      const result = await apiFetch<DocumentOut[]>("/api/documents");
      setDocuments(result);
    } catch (caught) {
      setError(errorMessage(caught, "无法加载文档列表"));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadDocuments();
  }, [loadDocuments]);

  const activeTaskKey = Object.values(tasks)
    .filter((task) => ACTIVE_TASK_STATUSES.has(task.status))
    .map((task) => task.id)
    .sort()
    .join(",");

  useEffect(() => {
    if (!activeTaskKey) return;
    const taskIds = new Set(activeTaskKey.split(","));
    let active = true;
    let polling = false;

    const poll = async () => {
      if (polling) return;
      const pending = Object.entries(tasksRef.current).filter(([, task]) =>
        taskIds.has(task.id) && ACTIVE_TASK_STATUSES.has(task.status)
      );
      if (pending.length === 0) return;
      polling = true;
      try {
        const updates = await Promise.all(
          pending.map(async ([documentId, tracked]) => ({
            documentId,
            task: await apiFetch<TaskOut>(`/api/tasks/${encodeURIComponent(tracked.id)}`),
          })),
        );
        if (!active) return;
        setTasks((current) => {
          const next = { ...current };
          for (const update of updates) next[update.documentId] = update.task;
          return next;
        });
        if (updates.some((update) => !ACTIVE_TASK_STATUSES.has(update.task.status))) {
          await loadDocuments();
        }
      } catch (caught) {
        if (active) setError(errorMessage(caught, "无法刷新任务进度"));
      } finally {
        polling = false;
      }
    };

    const timer = window.setInterval(() => void poll(), 2000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [activeTaskKey, loadDocuments]);

  const trackTask = (
    accepted: UploadAccepted | DocumentTaskAccepted,
    kind: string,
  ) => {
    setTasks((current) => ({
      ...current,
      [accepted.document_id]: {
        id: accepted.task_id,
        kind,
        status: "queued",
        attempts: 0,
        error: null,
      },
    }));
  };

  const acceptUpload = async (accepted: UploadAccepted) => {
    trackTask(accepted, "document");
    await loadDocuments();
  };

  const mutateDocument = async (
    document: DocumentOut,
    operation: "reindex" | "delete",
  ) => {
    if (!csrfToken || mutation) return;
    const mutationKey = `${document.id}:${operation}`;
    setMutation(mutationKey);
    setError(null);
    try {
      const accepted = await apiFetch<DocumentTaskAccepted>(
        operation === "reindex"
          ? `/api/documents/${encodeURIComponent(document.id)}/reindex`
          : `/api/documents/${encodeURIComponent(document.id)}`,
        { method: operation === "reindex" ? "POST" : "DELETE" },
        csrfToken,
      );
      trackTask(accepted, operation);
    } catch (caught) {
      setError(errorMessage(caught, operation === "reindex" ? "重新索引失败" : "删除失败"));
    } finally {
      setMutation(null);
    }
  };

  const logOut = async () => {
    try {
      await signOut();
    } catch (caught) {
      setError(errorMessage(caught, "退出登录失败"));
    }
  };

  if (!csrfToken) return null;

  const columns: ColumnsType<DocumentOut> = [
    {
      title: "文件名",
      dataIndex: "filename",
      key: "filename",
      render: (filename: string, document) => {
        const task = tasks[document.id];
        const taskError = task?.status === "failed"
          ? task.error ?? "Operation failed"
          : null;
        const documentError = document.status === "failed" && !taskError
          ? document.error ?? "Operation failed"
          : null;
        return (
          <div>
            <Typography.Text strong>{filename}</Typography.Text>
            {(taskError || documentError) && (
              <Alert
                type="error"
                showIcon
                message={taskError ?? documentError}
                style={{ marginTop: 8 }}
              />
            )}
          </div>
        );
      },
    },
    {
      title: "状态",
      key: "status",
      width: 160,
      render: (_, document) => {
        const task = tasks[document.id];
        const status = task && ACTIVE_TASK_STATUSES.has(task.status)
          ? task.status
          : task?.status === "failed"
            ? "failed"
            : document.status === "uploaded"
              ? "queued"
              : document.status;
        const presentation = STATUS_PRESENTATION[status] ?? {
          color: "default",
          label: status,
        };
        return (
          <Space direction="vertical" size={4}>
            <Tag color={presentation.color}>{presentation.label}</Tag>
            {ACTIVE_TASK_STATUSES.has(status) && (
              <Progress
                percent={status === "running" ? 60 : 20}
                status="active"
                showInfo={false}
                aria-label={`任务进度 ${document.filename}`}
              />
            )}
          </Space>
        );
      },
    },
    {
      title: "父块 / 子块",
      key: "chunks",
      width: 130,
      render: (_, document) => `${document.parent_count} / ${document.child_count}`,
    },
    {
      title: "操作",
      key: "actions",
      width: 300,
      render: (_, document) => {
        const task = tasks[document.id];
        const active = task ? ACTIVE_TASK_STATUSES.has(task.status) : false;
        const reindexMutation = mutation === `${document.id}:reindex`;
        const deleteMutation = mutation === `${document.id}:delete`;
        return (
          <Space wrap>
            {document.current_version_id && (
              <Button
                type="link"
                href={`/api/documents/${encodeURIComponent(document.id)}/content`}
                icon={<DownloadOutlined />}
                aria-label={`下载 ${document.filename}`}
              >
                下载
              </Button>
            )}
            <Popconfirm
              title={`重新索引 ${document.filename}？`}
              description="将基于当前已接受的版本重新建立索引。"
              okText="确认重新索引"
              cancelText="取消"
              disabled={active || !document.current_version_id || mutation !== null}
              okButtonProps={{ loading: reindexMutation }}
              onConfirm={() => mutateDocument(document, "reindex")}
            >
              <Button
                type="text"
                icon={<SyncOutlined />}
                disabled={active || !document.current_version_id || mutation !== null}
                aria-label={`重新索引 ${document.filename}`}
              >
                重新索引
              </Button>
            </Popconfirm>
            <Popconfirm
              title={`删除 ${document.filename}？`}
              description="删除任务完成后，该文档将不再可用。"
              okText="确认删除"
              cancelText="取消"
              disabled={active || mutation !== null}
              okButtonProps={{ danger: true, loading: deleteMutation }}
              onConfirm={() => mutateDocument(document, "delete")}
            >
              <Button
                type="text"
                danger
                icon={<DeleteOutlined />}
                disabled={active || mutation !== null}
                aria-label={`删除 ${document.filename}`}
              >
                删除
              </Button>
            </Popconfirm>
          </Space>
        );
      },
    },
  ];

  return (
    <Layout className="chat-shell">
      <Layout.Header className="chat-header">
        <Typography.Title level={3} className="chat-title">文档管理</Typography.Title>
        <Space>
          <Link to="/chat"><FileTextOutlined /> 返回对话</Link>
          <Link to="/monitoring"><MonitorOutlined /> 运行监控</Link>
          <Button type="text" icon={<LogoutOutlined />} onClick={() => void logOut()}>
            退出登录
          </Button>
        </Space>
      </Layout.Header>
      <Layout.Content style={{ width: "100%", maxWidth: 1280, margin: "0 auto", padding: 24 }}>
        <Space direction="vertical" size="large" style={{ width: "100%" }}>
          {error && (
            <Alert
              type="error"
              message={error}
              showIcon
              closable
              onClose={() => setError(null)}
            />
          )}
          <UploadPanel csrfToken={csrfToken} onAccepted={acceptUpload} />
          <Card title="文档列表">
            <Table<DocumentOut>
              rowKey="id"
              columns={columns}
              dataSource={documents}
              loading={loading}
              pagination={false}
              locale={{ emptyText: "暂无文档，请先上传。" }}
            />
          </Card>
        </Space>
      </Layout.Content>
    </Layout>
  );
}
