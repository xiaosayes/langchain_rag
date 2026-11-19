import { LockOutlined, UserOutlined } from "@ant-design/icons";
import { Alert, Button, Card, Form, Input, Typography } from "antd";
import { useState } from "react";
import { Navigate, useLocation, useNavigate } from "react-router-dom";
import { useAuth } from "../auth/AuthProvider";

interface LoginValues {
  username: string;
  password: string;
}

export default function LoginPage() {
  const { status, signIn } = useAuth();
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();

  if (status === "authenticated") return <Navigate to="/chat" replace />;

  const submit = async ({ username, password }: LoginValues) => {
    setSubmitting(true);
    setError(null);
    try {
      await signIn(username, password);
      const destination = (location.state as { from?: string } | null)?.from ?? "/chat";
      navigate(destination, { replace: true });
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "登录失败，请重试");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <main className="login-page">
      <Card className="login-card">
        <Typography.Title level={2}>登录</Typography.Title>
        <Typography.Paragraph type="secondary">企业知识助手</Typography.Paragraph>
        {error && <Alert type="error" message={error} showIcon className="login-error" />}
        <Form<LoginValues> layout="vertical" onFinish={submit} requiredMark={false}>
          <Form.Item
            name="username"
            label="用户名"
            rules={[{ required: true, message: "请输入用户名" }]}
          >
            <Input prefix={<UserOutlined />} autoComplete="username" />
          </Form.Item>
          <Form.Item
            name="password"
            label="密码"
            rules={[{ required: true, message: "请输入密码" }]}
          >
            <Input.Password prefix={<LockOutlined />} autoComplete="current-password" />
          </Form.Item>
          <Button aria-label="登录" type="primary" htmlType="submit" block loading={submitting}>
            登录
          </Button>
        </Form>
      </Card>
    </main>
  );
}
