import { DislikeOutlined, LikeOutlined } from "@ant-design/icons";
import { Alert, Button, Input, Space } from "antd";
import { useState } from "react";
import { apiFetch } from "../api/client";

interface FeedbackButtonsProps {
  messageId: string;
  csrfToken: string;
}

export default function FeedbackButtons({ messageId, csrfToken }: FeedbackButtonsProps) {
  const [score, setScore] = useState<-1 | 1 | null>(null);
  const [comment, setComment] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async () => {
    if (score === null) return;
    setSubmitting(true);
    setError(null);
    try {
      await apiFetch(
        "/api/feedback",
        {
          method: "POST",
          body: JSON.stringify({
            message_id: messageId,
            score,
            comment: comment.trim() || null,
          }),
        },
        csrfToken,
      );
      setSubmitted(true);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "反馈提交失败");
    } finally {
      setSubmitting(false);
    }
  };

  if (submitted) return <Alert type="success" message="感谢反馈" showIcon />;
  return (
    <div className="feedback-controls">
      <Space wrap>
        <Button
          aria-label="有帮助"
          aria-pressed={score === 1}
          icon={<LikeOutlined />}
          type={score === 1 ? "primary" : "default"}
          onClick={() => setScore(1)}
        />
        <Button
          aria-label="没帮助"
          aria-pressed={score === -1}
          icon={<DislikeOutlined />}
          type={score === -1 ? "primary" : "default"}
          onClick={() => setScore(-1)}
        />
        <Button
          disabled={score === null}
          loading={submitting}
          onClick={submit}
        >
          提交反馈
        </Button>
      </Space>
      <Input.TextArea
        aria-label="反馈评论"
        value={comment}
        onChange={(event) => setComment(event.target.value)}
        maxLength={2000}
        rows={2}
        placeholder="可选：补充反馈评论"
      />
      {error && <Alert type="error" message={error} showIcon />}
    </div>
  );
}
