import {
  DeleteOutlined,
  EditOutlined,
  FileTextOutlined,
  LogoutOutlined,
  PlusOutlined,
  SendOutlined,
  StopOutlined,
} from "@ant-design/icons";
import {
  Alert,
  Button,
  Input,
  Layout,
  Modal,
  Popconfirm,
  Space,
  Tag,
  Typography,
} from "antd";
import { useCallback, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import { Link, useNavigate, useParams } from "react-router-dom";
import { apiFetch } from "../api/client";
import type { ChatMessage, Conversation } from "../api/client";
import { streamChat } from "../api/client";
import { useAuth } from "../auth/AuthProvider";
import CitationList from "../components/CitationList";
import FeedbackButtons from "../components/FeedbackButtons";

const { Content, Header, Sider } = Layout;

export default function ChatPage() {
  const { id: conversationId } = useParams();
  const { csrfToken, signOut } = useAuth();
  const navigate = useNavigate();
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [question, setQuestion] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [loadingMessages, setLoadingMessages] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [renameTarget, setRenameTarget] = useState<Conversation | null>(null);
  const [renameTitle, setRenameTitle] = useState("");
  const [mutating, setMutating] = useState(false);
  const streamController = useRef<AbortController | null>(null);

  const loadConversations = useCallback(async () => {
    try {
      const result = await apiFetch<Conversation[]>("/api/sessions");
      setConversations(result);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "无法加载会话");
    }
  }, []);

  useEffect(() => {
    void loadConversations();
  }, [loadConversations]);

  useEffect(() => {
    let active = true;
    setMessages([]);
    setError(null);
    if (!conversationId) return;
    setLoadingMessages(true);
    apiFetch<ChatMessage[]>(`/api/sessions/${encodeURIComponent(conversationId)}/messages`)
      .then((result) => {
        if (active) setMessages(result);
      })
      .catch((caught: unknown) => {
        if (active) setError(caught instanceof Error ? caught.message : "无法加载消息");
      })
      .finally(() => {
        if (active) setLoadingMessages(false);
      });
    return () => {
      active = false;
    };
  }, [conversationId]);

  useEffect(() => () => streamController.current?.abort(), []);

  if (!csrfToken) return null;

  const createConversation = async () => {
    setMutating(true);
    setError(null);
    try {
      const created = await apiFetch<Conversation>(
        "/api/sessions",
        { method: "POST", body: JSON.stringify({}) },
        csrfToken,
      );
      setConversations((current) => [created, ...current]);
      navigate(`/chat/${created.id}`);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "无法创建会话");
    } finally {
      setMutating(false);
    }
  };

  const renameConversation = async () => {
    if (!renameTarget || !renameTitle.trim()) return;
    setMutating(true);
    try {
      const renamed = await apiFetch<Conversation>(
        `/api/sessions/${encodeURIComponent(renameTarget.id)}`,
        {
          method: "PATCH",
          body: JSON.stringify({ title: renameTitle.trim() }),
        },
        csrfToken,
      );
      setConversations((current) =>
        current.map((item) => (item.id === renamed.id ? renamed : item)),
      );
      setRenameTarget(null);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "无法重命名会话");
    } finally {
      setMutating(false);
    }
  };

  const deleteConversation = async (id: string) => {
    setMutating(true);
    try {
      await apiFetch<void>(
        `/api/sessions/${encodeURIComponent(id)}`,
        { method: "DELETE" },
        csrfToken,
      );
      setConversations((current) => current.filter((item) => item.id !== id));
      if (conversationId === id) navigate("/chat");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "无法删除会话");
    } finally {
      setMutating(false);
    }
  };

  const send = async () => {
    const trimmed = question.trim();
    if (!trimmed || streaming || loadingMessages) return;
    setQuestion("");
    setError(null);
    setStreaming(true);
    const now = new Date().toISOString();
    const temporaryId = `temp-${Date.now()}`;
    const optimisticAssistant: ChatMessage = {
      id: temporaryId,
      role: "assistant",
      content: "",
      citations: [],
      degraded: false,
      generation_id: null,
      created_at: now,
    };
    setMessages((current) => [
      ...current,
      {
        id: `${temporaryId}-question`,
        role: "user",
        content: trimmed,
        citations: [],
        degraded: false,
        generation_id: null,
        created_at: now,
      },
      optimisticAssistant,
    ]);

    const controller = new AbortController();
    streamController.current = controller;
    let answer = "";
    let assistantId = temporaryId;
    const updateAssistant = (update: Partial<ChatMessage>) => {
      setMessages((current) =>
        current.map((message) =>
          message.id === assistantId ? { ...message, ...update } : message,
        ),
      );
    };
    try {
      await streamChat(
        { conversation_id: conversationId, question: trimmed },
        {
          onStart: (event) => {
            updateAssistant({
              id: event.message_id,
              degraded: event.degraded,
              generation_id: event.generation_id,
            });
            assistantId = event.message_id;
          },
          onToken: (event) => {
            answer += event.content;
            updateAssistant({ content: answer });
          },
          onCitations: (event) => {
            updateAssistant({ citations: event.citations });
          },
          onDone: (event) => {
            void loadConversations();
            if (!conversationId) navigate(`/chat/${event.conversation_id}`);
          },
          onError: (event) => {
            updateAssistant({ error: event.message });
          },
        },
        controller.signal,
        csrfToken,
      );
    } catch (caught) {
      const stopped = caught instanceof DOMException && caught.name === "AbortError";
      const text = stopped
        ? "已停止，回答可能不完整"
        : caught instanceof Error
          ? caught.message
          : "对话请求失败";
      updateAssistant({ error: text });
    } finally {
      if (streamController.current === controller) {
        streamController.current = null;
        setStreaming(false);
      }
    }
  };

  const logOut = async () => {
    try {
      await signOut();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "退出登录失败");
    }
  };

  const activeTitle = conversations.find((item) => item.id === conversationId)?.title ?? "新对话";

  return (
    <Layout className="chat-shell">
      <Sider width={280} breakpoint="lg" collapsedWidth={0} className="chat-sider">
        <div className="brand">
          <h1>企业知识助手</h1>
          <Typography.Text type="secondary">基于企业知识库的回答</Typography.Text>
        </div>
        <div className="conversation-actions">
          <Button
            type="primary"
            icon={<PlusOutlined />}
            block
            loading={mutating}
            onClick={createConversation}
          >
            新建会话
          </Button>
        </div>
        <nav className="conversation-list" aria-label="会话列表">
          {conversations.map((conversation) => (
            <div
              className={`conversation-item ${conversation.id === conversationId ? "active" : ""}`}
              key={conversation.id}
            >
              <Link className="conversation-link" to={`/chat/${conversation.id}`}>
                {conversation.title}
              </Link>
              <Button
                type="text"
                size="small"
                aria-label={`重命名 ${conversation.title}`}
                icon={<EditOutlined />}
                onClick={() => {
                  setRenameTarget(conversation);
                  setRenameTitle(conversation.title);
                }}
              />
              <Popconfirm
                title="删除这个会话？"
                description="此操作无法撤销。"
                okText="删除"
                cancelText="取消"
                onConfirm={() => deleteConversation(conversation.id)}
              >
                <Button
                  type="text"
                  size="small"
                  danger
                  aria-label={`删除 ${conversation.title}`}
                  icon={<DeleteOutlined />}
                />
              </Popconfirm>
            </div>
          ))}
        </nav>
      </Sider>
      <Layout>
        <Header className="chat-header">
          <Typography.Title level={3}>{activeTitle}</Typography.Title>
          <Space>
            <Link to="/documents"><FileTextOutlined /> <span className="header-label">文档管理</span></Link>
            <Button type="text" icon={<LogoutOutlined />} onClick={() => void logOut()}>
              退出登录
            </Button>
          </Space>
        </Header>
        <Content className="chat-content">
          <div className="messages" aria-live="polite">
            {error && <Alert type="error" message={error} showIcon closable onClose={() => setError(null)} />}
            {!loadingMessages && messages.length === 0 && (
              <div className="empty-chat">输入问题，开始一次有引用的知识库对话。</div>
            )}
            {messages.map((message) => (
              <article className={`message ${message.role}`} key={message.id}>
                <div className="message-card">
                  {message.role === "assistant" && message.degraded && (
                    <Tag color="warning" className="degraded-badge">降级模式</Tag>
                  )}
                  {message.role === "assistant" ? (
                    <ReactMarkdown>{message.content}</ReactMarkdown>
                  ) : (
                    message.content
                  )}
                  {message.error && <Alert className="message-error" type="warning" message={message.error} />}
                  {message.role === "assistant" && (
                    <div className="message-tools">
                      <CitationList citations={message.citations} />
                      {!message.id.startsWith("temp-") && (
                        <FeedbackButtons messageId={message.id} csrfToken={csrfToken} />
                      )}
                    </div>
                  )}
                </div>
              </article>
            ))}
          </div>
          <div className="composer">
            <div className="composer-row">
              <Input.TextArea
                className="composer-input"
                aria-label="问题"
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                onPressEnter={(event) => {
                  if (!event.shiftKey) {
                    event.preventDefault();
                    void send();
                  }
                }}
                autoSize={{ minRows: 1, maxRows: 6 }}
                placeholder="输入问题，Enter 发送，Shift+Enter 换行"
                disabled={streaming || loadingMessages}
              />
              {streaming ? (
                <Button
                  aria-label="停止"
                  danger
                  icon={<StopOutlined />}
                  onClick={() => streamController.current?.abort()}
                >
                  停止
                </Button>
              ) : (
                <Button
                  aria-label="发送"
                  type="primary"
                  icon={<SendOutlined />}
                  disabled={!question.trim() || loadingMessages}
                  onClick={() => void send()}
                >
                  发送
                </Button>
              )}
            </div>
          </div>
        </Content>
      </Layout>
      <Modal
        title="重命名会话"
        open={renameTarget !== null}
        okText="保存"
        cancelText="取消"
        confirmLoading={mutating}
        okButtonProps={{ disabled: !renameTitle.trim() }}
        onOk={() => void renameConversation()}
        onCancel={() => setRenameTarget(null)}
      >
        <Input
          aria-label="会话标题"
          value={renameTitle}
          maxLength={200}
          onChange={(event) => setRenameTitle(event.target.value)}
        />
      </Modal>
    </Layout>
  );
}
