import {
  FileTextOutlined,
  LogoutOutlined,
  MessageOutlined,
  MonitorOutlined,
  ReloadOutlined,
} from "@ant-design/icons";
import {
  Alert,
  Button,
  Card,
  Col,
  Empty,
  Layout,
  Row,
  Select,
  Space,
  Spin,
  Statistic,
  Table,
  Tag,
  Timeline,
  Typography,
} from "antd";
import type { ColumnsType } from "antd/es/table";
import { useCallback, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import {
  getMonitoringRun,
  getMonitoringRuns,
  getMonitoringSummary,
} from "../api/monitoring";
import type {
  MonitoringRunDetail,
  MonitoringRunStatus,
  MonitoringRunSummary,
  MonitoringSummary,
} from "../api/monitoring";
import { useAuth } from "../auth/AuthProvider";

const STATUS_PRESENTATION: Record<
  MonitoringRunStatus,
  { color: string; label: string }
> = {
  running: { color: "processing", label: "运行中" },
  succeeded: { color: "success", label: "成功" },
  failed: { color: "error", label: "失败" },
  cancelled: { color: "default", label: "已取消" },
};

const numberFormatter = new Intl.NumberFormat("zh-CN", {
  maximumFractionDigits: 1,
});

function errorMessage(caught: unknown, fallback: string): string {
  return caught instanceof Error ? caught.message : fallback;
}

function formatDuration(milliseconds: number | null): string {
  return milliseconds === null ? "—" : `${numberFormatter.format(milliseconds)} ms`;
}

function formatDate(value: string | null): string {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString("zh-CN", { hour12: false });
}

function contentText(value: unknown): string {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "string") return value;
  return JSON.stringify(value, null, 2) ?? String(value);
}

function TextSection({ title, value }: { title: string; value: unknown }) {
  return (
    <section className="monitoring-text-section">
      <Typography.Title level={5}>{title}</Typography.Title>
      <pre className="monitoring-text">{contentText(value)}</pre>
    </section>
  );
}

export default function MonitoringPage() {
  const { csrfToken, signOut } = useAuth();
  const [summary, setSummary] = useState<MonitoringSummary | null>(null);
  const [runs, setRuns] = useState<MonitoringRunSummary[]>([]);
  const [status, setStatus] = useState<MonitoringRunStatus | undefined>();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [detail, setDetail] = useState<MonitoringRunDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState<string | null>(null);
  const dashboardRequest = useRef(0);
  const detailRequest = useRef(0);

  const loadDashboard = useCallback(async () => {
    const request = ++dashboardRequest.current;
    setLoading(true);
    setError(null);
    try {
      const [nextSummary, nextRuns] = await Promise.all([
        getMonitoringSummary(),
        getMonitoringRuns(status),
      ]);
      if (dashboardRequest.current !== request) return;
      setSummary(nextSummary);
      setRuns(nextRuns);
    } catch (caught) {
      if (dashboardRequest.current !== request) return;
      setSummary(null);
      setRuns([]);
      setError(errorMessage(caught, "无法加载运行监控"));
    } finally {
      if (dashboardRequest.current === request) setLoading(false);
    }
  }, [status]);

  useEffect(() => {
    void loadDashboard();
    return () => {
      dashboardRequest.current += 1;
    };
  }, [loadDashboard]);

  useEffect(() => () => {
    detailRequest.current += 1;
  }, []);

  const loadDetail = useCallback(async (runId: string) => {
    const request = ++detailRequest.current;
    setSelectedRunId(runId);
    setDetail(null);
    setDetailError(null);
    setDetailLoading(true);
    try {
      const nextDetail = await getMonitoringRun(runId);
      if (detailRequest.current !== request) return;
      setDetail(nextDetail);
    } catch (caught) {
      if (detailRequest.current !== request) return;
      setDetailError(errorMessage(caught, "无法加载运行详情"));
    } finally {
      if (detailRequest.current === request) setDetailLoading(false);
    }
  }, []);

  const changeStatus = (nextStatus?: MonitoringRunStatus) => {
    detailRequest.current += 1;
    setSelectedRunId(null);
    setDetail(null);
    setDetailError(null);
    setDetailLoading(false);
    setStatus(nextStatus);
  };

  const refresh = () => {
    void loadDashboard();
    if (selectedRunId) void loadDetail(selectedRunId);
  };

  const logOut = async () => {
    try {
      await signOut();
    } catch (caught) {
      setError(errorMessage(caught, "退出登录失败"));
    }
  };

  if (!csrfToken) return null;

  const summaryMetrics = summary
    ? [
        { title: "总请求", value: summary.total },
        { title: "成功", value: summary.succeeded },
        { title: "失败", value: summary.failed },
        { title: "已取消", value: summary.cancelled },
        { title: "运行中", value: summary.running },
        { title: "降级", value: summary.degraded },
        { title: "平均耗时", value: formatDuration(summary.average_duration_ms) },
        { title: "P95 耗时", value: formatDuration(summary.p95_duration_ms) },
      ]
    : [];

  const columns: ColumnsType<MonitoringRunSummary> = [
    {
      title: "问题",
      dataIndex: "question",
      key: "question",
      render: (question: string, run) => (
        <Button
          type="link"
          className="monitoring-question"
          aria-label={`查看详情 ${question} ${run.id}`}
          onClick={() => void loadDetail(run.id)}
        >
          {question}
        </Button>
      ),
    },
    {
      title: "状态",
      dataIndex: "status",
      key: "status",
      width: 150,
      render: (runStatus: MonitoringRunStatus, run) => (
        <Space size={4} wrap>
          <Tag color={STATUS_PRESENTATION[runStatus].color}>
            {STATUS_PRESENTATION[runStatus].label}
          </Tag>
          {run.degraded && <Tag color="warning">降级</Tag>}
        </Space>
      ),
    },
    {
      title: "开始时间",
      dataIndex: "created_at",
      key: "created_at",
      width: 190,
      render: formatDate,
    },
    {
      title: "耗时",
      dataIndex: "duration_ms",
      key: "duration_ms",
      width: 120,
      render: formatDuration,
    },
  ];

  return (
    <Layout className="chat-shell">
      <Layout.Header className="chat-header">
        <Typography.Title level={3} className="chat-title">
          <span aria-hidden="true"><MonitorOutlined /></span> 本地运行监控
        </Typography.Title>
        <Space wrap>
          <Link to="/chat" aria-label="返回对话"><span aria-hidden="true"><MessageOutlined /></span> <span className="header-label">返回对话</span></Link>
          <Link to="/documents" aria-label="文档管理"><span aria-hidden="true"><FileTextOutlined /></span> <span className="header-label">文档管理</span></Link>
          <Button
            icon={<ReloadOutlined />}
            loading={loading}
            onClick={refresh}
            aria-label="刷新"
          >
            刷新
          </Button>
          <Button type="text" icon={<LogoutOutlined />} onClick={() => void logOut()}>
            退出登录
          </Button>
        </Space>
      </Layout.Header>
      <Layout.Content className="monitoring-content">
        <Space direction="vertical" size="large" className="monitoring-stack">
          <Typography.Text type="secondary">最近 24 小时 · 仅手动刷新</Typography.Text>
          {error && (
            <Alert
              type="error"
              message={error}
              showIcon
              closable
              onClose={() => setError(null)}
            />
          )}
          <section aria-label="运行摘要">
            {loading && !summary ? (
              <div className="monitoring-loading"><Spin /></div>
            ) : summary ? (
              <Row gutter={[16, 16]}>
                {summaryMetrics.map((metric) => (
                  <Col xs={12} sm={8} lg={6} xl={3} key={metric.title}>
                    <Card className="monitoring-summary-card">
                      <Statistic title={metric.title} value={metric.value} />
                    </Card>
                  </Col>
                ))}
              </Row>
            ) : null}
          </section>
          <Card
            title="最近运行"
            extra={
              <Select<MonitoringRunStatus | undefined>
                aria-label="状态筛选"
                value={status}
                onChange={changeStatus}
                allowClear
                placeholder="全部状态"
                className="monitoring-status-filter"
                options={Object.entries(STATUS_PRESENTATION).map(([value, presentation]) => ({
                  value: value as MonitoringRunStatus,
                  label: presentation.label,
                }))}
              />
            }
          >
            <section aria-label="最近运行">
              <Table<MonitoringRunSummary>
                rowKey="id"
                columns={columns}
                dataSource={runs}
                loading={loading}
                pagination={false}
                locale={{ emptyText: "暂无运行记录。" }}
              />
            </section>
          </Card>
          <section aria-label="运行详情">
            <Card title="运行详情">
              {detailLoading ? (
                <div className="monitoring-loading"><Spin /></div>
              ) : detailError ? (
                <Alert type="error" message={detailError} showIcon />
              ) : detail ? (
                <Space direction="vertical" size="middle" className="monitoring-stack">
                  <Space wrap>
                    <Typography.Text code>{detail.id}</Typography.Text>
                    <Tag color={STATUS_PRESENTATION[detail.status].color}>
                      {STATUS_PRESENTATION[detail.status].label}
                    </Tag>
                    {detail.degraded && <Tag color="warning">降级</Tag>}
                  </Space>
                  <TextSection title="问题" value={detail.question} />
                  <TextSection title="回答" value={detail.answer} />
                  <TextSection title="检索查询" value={detail.retrieval_query} />
                  <section className="monitoring-text-section">
                    <Typography.Title level={5}>提示词</Typography.Title>
                    {detail.prompt_messages.length > 0 ? (
                      detail.prompt_messages.map((message, index) => (
                        <Card size="small" title={message.type} key={`${index}-${message.type}`}>
                          <pre className="monitoring-text">{contentText(message.content)}</pre>
                        </Card>
                      ))
                    ) : (
                      <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="暂无提示词" />
                    )}
                  </section>
                  <TextSection title="检索文本" value={detail.retrieved_text} />
                  {detail.error && (
                    <Alert type="error" message="运行错误" description={detail.error} showIcon />
                  )}
                  <section className="monitoring-text-section">
                    <Typography.Title level={5}>阶段时间线</Typography.Title>
                    {detail.events.length > 0 ? (
                      <Timeline
                        items={detail.events.map((event) => ({
                          key: event.id,
                          color: event.status === "failed" ? "red" : event.status === "succeeded" ? "green" : "blue",
                          children: (
                            <div className="monitoring-event">
                              <Space wrap>
                                <span>{event.stage}</span>
                                <Tag>{event.status}</Tag>
                                <Typography.Text type="secondary">
                                  {formatDate(event.started_at)} · {formatDuration(event.duration_ms)}
                                </Typography.Text>
                              </Space>
                              {Object.keys(event.details).length > 0 && (
                                <pre className="monitoring-text">{contentText(event.details)}</pre>
                              )}
                            </div>
                          ),
                        }))}
                      />
                    ) : (
                      <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="暂无阶段事件" />
                    )}
                  </section>
                </Space>
              ) : (
                <Empty description="选择最近运行查看详情。" />
              )}
            </Card>
          </section>
        </Space>
      </Layout.Content>
    </Layout>
  );
}
