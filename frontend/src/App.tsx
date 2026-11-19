import { Button, Result, Spin } from "antd";
import { Link, Navigate, Route, Routes, useLocation } from "react-router-dom";
import type { ReactNode } from "react";
import { AuthProvider, useAuth } from "./auth/AuthProvider";
import ChatPage from "./pages/ChatPage";
import LoginPage from "./pages/LoginPage";

function ProtectedRoute({ children }: { children: ReactNode }) {
  const { status } = useAuth();
  const location = useLocation();
  if (status === "loading") {
    return (
      <div className="page-loading" aria-label="正在验证登录状态">
        <Spin size="large" />
      </div>
    );
  }
  if (status === "unauthenticated") {
    return <Navigate to="/login" replace state={{ from: location.pathname }} />;
  }
  return children;
}

function DocumentsPlaceholder() {
  return (
    <Result
      status="info"
      title="文档管理"
      subTitle="文档管理页面将在下一阶段提供。"
      extra={<Button type="primary"><Link to="/chat">返回对话</Link></Button>}
    />
  );
}

export default function App() {
  return (
    <AuthProvider>
      <Routes>
        <Route path="/login" element={<LoginPage />} />
        <Route path="/chat" element={<ProtectedRoute><ChatPage /></ProtectedRoute>} />
        <Route path="/chat/:id" element={<ProtectedRoute><ChatPage /></ProtectedRoute>} />
        <Route
          path="/documents"
          element={<ProtectedRoute><DocumentsPlaceholder /></ProtectedRoute>}
        />
        <Route path="*" element={<Navigate to="/chat" replace />} />
      </Routes>
    </AuthProvider>
  );
}
