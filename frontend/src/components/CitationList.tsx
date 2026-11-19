import { Collapse, Typography } from "antd";
import type { Citation } from "../api/client";

export default function CitationList({ citations }: { citations: Citation[] }) {
  if (citations.length === 0) return null;
  return (
    <Collapse
      className="citation-list"
      ghost
      items={citations.map((citation, index) => ({
        key: `${citation.label}-${citation.document_id}-${index}`,
        label: `${citation.label} ${citation.document_id}`,
        children: (
          <div className="citation-content">
            <Typography.Paragraph>{citation.text}</Typography.Paragraph>
            <Typography.Text type="secondary">
              文档：{citation.document_id} · 段落：{citation.parent_id}
            </Typography.Text>
            <Collapse
              ghost
              items={[
                {
                  key: "metadata",
                  label: "引用元数据",
                  children: (
                    <pre className="citation-metadata">
                      {JSON.stringify(citation.metadata, null, 2)}
                    </pre>
                  ),
                },
              ]}
            />
          </div>
        ),
      }))}
    />
  );
}
