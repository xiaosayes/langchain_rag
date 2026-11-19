import { UploadOutlined } from "@ant-design/icons";
import { Alert, Button, Card, Radio, Space, Typography, Upload } from "antd";
import type { UploadFile, UploadProps } from "antd";
import { useState } from "react";
import { apiFetch } from "../api/client";

export type DocumentKind = "document" | "qa";

export interface UploadAccepted {
  document_id: string;
  version_id: string;
  task_id: string;
}

interface UploadPanelProps {
  csrfToken: string;
  onAccepted: (accepted: UploadAccepted) => void | Promise<void>;
}

const MAX_UPLOAD_BYTES = 50 * 1024 * 1024;
const ACCEPTED_EXTENSIONS = [".txt", ".docx", ".pdf", ".xlsx"];

export default function UploadPanel({ csrfToken, onAccepted }: UploadPanelProps) {
  const [kind, setKind] = useState<DocumentKind>("document");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const beforeUpload: NonNullable<UploadProps["beforeUpload"]> = (file) => {
    const extension = file.name.includes(".")
      ? `.${file.name.split(".").pop()?.toLowerCase()}`
      : "";
    if (!ACCEPTED_EXTENSIONS.includes(extension)) {
      setSelectedFile(null);
      setError("仅支持 TXT、DOCX、PDF、XLSX 文件");
      return Upload.LIST_IGNORE;
    }
    if (file.size > MAX_UPLOAD_BYTES) {
      setSelectedFile(null);
      setError("文件不能超过 50 MB");
      return Upload.LIST_IGNORE;
    }
    if (kind === "qa" && extension !== ".xlsx") {
      setSelectedFile(null);
      setError("问答对（QA）需上传 XLSX 文件");
      return Upload.LIST_IGNORE;
    }
    setError(null);
    setSelectedFile(file);
    return false;
  };

  const upload = async () => {
    if (!selectedFile || uploading) return;
    setUploading(true);
    setError(null);
    const body = new FormData();
    body.append("file", selectedFile);
    body.append("kind", kind);
    try {
      const accepted = await apiFetch<UploadAccepted>(
        "/api/documents/upload",
        { method: "POST", body },
        csrfToken,
      );
      setSelectedFile(null);
      await onAccepted(accepted);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "上传失败，请稍后重试");
    } finally {
      setUploading(false);
    }
  };

  const fileList: UploadFile[] = selectedFile
    ? [{ uid: "selected-file", name: selectedFile.name, size: selectedFile.size }]
    : [];

  return (
    <Card title="上传文档">
      <Space direction="vertical" size="middle" className="upload-panel">
        <Typography.Text type="secondary">
          支持 TXT、DOCX、PDF、XLSX，单个文件不超过 50 MB；问答对（QA）需使用 XLSX。
        </Typography.Text>
        <Radio.Group
          value={kind}
          disabled={uploading}
          onChange={(event) => setKind(event.target.value as DocumentKind)}
          options={[
            { value: "document", label: "普通文档" },
            { value: "qa", label: "问答对（QA）" },
          ]}
        />
        <Upload
          accept=".txt,.docx,.pdf,.xlsx"
          beforeUpload={beforeUpload}
          fileList={fileList}
          maxCount={1}
          disabled={uploading}
          onRemove={() => setSelectedFile(null)}
        >
          <Button icon={<UploadOutlined />} disabled={uploading}>选择文件</Button>
        </Upload>
        {error && <Alert type="error" message={error} showIcon />}
        <Button
          type="primary"
          loading={uploading}
          disabled={!selectedFile}
          onClick={() => void upload()}
        >
          上传文档
        </Button>
      </Space>
    </Card>
  );
}
