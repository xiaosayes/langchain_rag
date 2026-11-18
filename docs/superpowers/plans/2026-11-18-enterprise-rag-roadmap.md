# Enterprise RAG Delivery Roadmap

本路线图将已评审的设计拆成四个顺序交付阶段。每个阶段都形成可运行、可测试的软件，并在进入下一阶段前完成代码评审和验证。

## Phase 1：RAG 核心与 Mock 语料

详细计划：`docs/superpowers/plans/2026-11-18-enterprise-rag-core.md`

- 建立 Python 3.11 项目和类型化配置。
- 生成 100 个普通文档、1000 条知识 Q&A 和 100 条独立评测数据。
- 完成 TXT/DOCX/PDF/XLSX/Q&A 解析和扫描 PDF OCR。
- 完成父子切分、百炼模型适配、Qdrant、BM25、RRF、Rerank 和父块恢复。
- 完成整代索引构建、活动代切换和 CLI 问答。
- 验收：通过 CLI 可以重建语料、导入知识库并获得带引用的回答。

## Phase 2：Web 应用与知识库管理

计划文件：`docs/superpowers/plans/2026-11-18-enterprise-rag-web.md`，在 Phase 1 接口验收后编写。

- 增加 SQLite 任务队列、单 Worker、单管理员认证和安全会话。
- 增加文档、会话、聊天、反馈和健康检查 API。
- 增加 React 登录页、流式问答页和知识库管理页。
- 完成 SSE、CSRF、上传安全、断连取消和异常降级。
- 验收：单管理员可通过浏览器完成登录、上传、索引、问答、引用和反馈全流程。

## Phase 3：LangSmith 与效果评测

计划文件：`docs/superpowers/plans/2026-11-18-enterprise-rag-observability.md`，在 Phase 2 验收后编写。

- 增加完整问答 Trace、生产采样、本地反馈与 Run ID 关联。
- 增加 LangSmith Dataset、Feedback 和 Experiments。
- 实现 Recall、MRR、引用支持率、忠实度和拒答评估。
- 完成规则检查、LLM Judge、人工抽查数据和性能报告。
- 验收：生成可重复的质量报告，并能在 LangSmith 中查看 Trace 和反馈。

## Phase 4：Ubuntu 部署与运维

计划文件：`docs/superpowers/plans/2026-11-18-enterprise-rag-deployment.md`，在 Phase 3 验收后编写。

- 创建 `langchain_test` Conda 环境文件和精确依赖锁。
- 安装固定版本 Qdrant 独立二进制，配置专属端口和数据目录。
- 提供 Qdrant、FastAPI、Worker 启停及 systemd 模板。
- 提供备份、恢复、卸载、Nginx 示例和部署验证脚本。
- 验收：本地交付包检查通过；服务器最终状态以用户手动运行验证脚本的结果为准。

## 阶段依赖

```text
Phase 1 核心接口与 CLI
   → Phase 2 Web/API
   → Phase 3 线上观测与质量门禁
   → Phase 4 部署包与服务器验收
```

每个阶段都遵守测试先行、小步提交、实际命令验证和不得虚报服务器部署结果的约束。
