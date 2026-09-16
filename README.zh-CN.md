# AutoSales-Agent

[English](README.md) · [Demo 作品集](docs/demo/README.md)

> 一个面向新能源汽车销售咨询的多工具对话 Agent 工程化原型：结合本地 RAG、确定性计算、金融来源校验、持久化记忆、证据溯源与故障降级。

AutoSales-Agent 不是“所有问题都检索后让大模型回答”的普通聊天机器人，而是一个由单个 Agent 协调多种专用工具的应用原型。它会根据用户问题判断：该查本地产品资料、联网查实时信息、只接受官方金融来源、做确定性计算、使用图片上下文，还是直接回答。

## 项目解决的问题

新能源汽车销售咨询通常同时包含：

- 产品参数、充电、座舱、质保等稳定事实；
- 月供、年电费、落地价等需要精确计算的问题；
- 银行分期、竞品价格等实时且有来源要求的信息；
- 多轮对话中的城市、预算、购车时间和贷款偏好；
- 工具失败时不能凭空补全事实的安全边界。

核心原则：**LLM 负责语义理解、规划、参数理解和回答生成；检索、计算、来源校验与状态持久化交给专用组件。**

## 核心能力

| 模块 | 职责 | 典型场景 |
| --- | --- | --- |
| Local RAG | 检索本地产品资料 | 配置、续航、充电、质保、座舱 |
| Web Search | 获取非金融实时外部信息 | 竞品、新闻、市场信息 |
| Finance Search | 只接受可验证官方金融来源 | 银行车贷、分期方案、金融政策 |
| Calculator | 确定性计算 | 月供、年度电费、购车预算 |
| Parameter Parser | 归一化自然语言参数 | “首付三成”“36个月”“电价0.6元一度” |
| SQLite Memory | 持久化会话偏好 | 城市、预算、购车计划、关注车型 |
| Vision | 基于配置的视觉模型识别图片 | 车型页、充电页、座舱页 |
| JSON Graph | 轻量关系补充 | 用户需求 → 推荐配置关系 |

## 系统流程

```text
用户问题
  → 会话状态 / SQLite 记忆
  → Planner 选择 no tool / 单工具 / 多工具
  → 参数预检与工具执行
  → 有效结果转为 Evidence
  → 基于证据生成回答
  → 写入 Trace、故障事件与会话状态
```

其中 `planned_tools` 与 `executed_tools` 被明确区分。例如用户说“车价 30 万、首付 30%、贷款 3 年，帮我算月供”，Planner 应规划 `car_calculator`，但缺少利率时计算器不会执行；系统只追问利率。补齐后才执行确定性计算。

## 证据与故障处理

成功的工具结果会被统一转换为 Evidence，包含 `evidence_id`、来源类型、校验状态及必要元数据：

```text
Tool Result → Evidence ID → Answer Citation / Claim Mapping
```

失败、空结果或未经验证的金融信息不会进入 Evidence。若一个多工具请求中计算器成功、RAG 失败，系统保留计算结果，同时明确说明产品事实暂时无法验证，而不是假装两个工具都成功。

## 检索排序决策

冻结运行时策略：

```text
R0 = lexical_first
Top-K = 4
```

项目曾测试 R3（语义分数 0.7 + 词法分数 0.3）。它在 27 条开发检索集上看起来更好，但在冻结独立人工核验集上没有稳定泛化，因此没有上线。

| 冻结独立验证集（45 条正例 / 48 条总样本） | Hit@1 | Hit@4 | Recall@4 | MRR | 决策 |
| --- | ---: | ---: | ---: | ---: | --- |
| R0 lexical-first | 77.78% | 97.78% | 91.85% | 86.11% | 保留 |
| R3 语义加权融合 | 73.33% | 97.78% | 91.85% | 83.33% | 拒绝 |

R3 修复 2 条样本、破坏 4 条。结论不是“语义检索一定更差”，而是该固定融合策略在当前冻结验证集上没有稳定泛化。

## 已完成评测

| 评测层 | 范围 | 当前结果 |
| --- | --- | --- |
| 工具路由开发基线 | 38 条 | 38/38 |
| 计算器开发基线 | 8 条 | 8/8 |
| 独立行为集 | 60 条 | 56/56 确定性通过；4 条人工语义复核 |
| 真实 Agent E2E | 21 条 | 21/21；规划与执行均为 100% |
| Evidence | 21 条 E2E | Coverage 1.0；evidence gap 0 |
| 故障注入 | 25 条 | 25/25；unsafe fallback 0% |
| SQLite 记忆 | 15 条 | 15/15 |
| 图谱关系 | 3 条 | 3/3 |
| API | 18 条 TestClient + Uvicorn smoke | PASS |

以上是不同范围的评测结果，不能合并成一个“总体准确率”，也不代表生产环境保证。

## 快速开始

```bash
git clone https://github.com/<YOUR_GITHUB_USERNAME>/AutoSales-Agent.git
cd AutoSales-Agent
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

在 `.env` 中填写自己的模型服务配置。不要提交 `.env`，它已被 `.gitignore` 排除。

公开仓库不包含原产品 PDF、衍生知识库、向量库、SQLite 运行数据库或原始 Trace。请放入有授权的 PDF 后执行：

```bash
python app.py ingest
python app.py chat
```

持久化会话示例：

```bash
python app.py chat --session demo-session
```

FastAPI 服务：

```bash
uvicorn api.server:app --host 127.0.0.1 --port 8000
```

访问 `http://127.0.0.1:8000/docs` 查看 API 文档。

## 推荐演示顺序

1. RAG：询问“ES9 支持哪些快充标准？”
2. 缺参不乱算：输入“车价30万，首付30%，贷款3年，帮我算月供”，再补充“年利率3.5%”。
3. 持久记忆：使用固定 `--session` 输入预算，重启后问“你还记得我的预算吗？”。

完整 Demo 与脱敏 Trace 快照见：[docs/demo/README.md](docs/demo/README.md)。

## 项目边界

- 这是经过评测的应用原型，不是生产级金融顾问；
- 检索验证来自一个本地产品语料，不代表开放域泛化能力；
- Web 与 Finance 能力依赖上游网络服务；金融结果没有可验证官方来源时会拒绝采用；
- Vision 使用通用多模态模型，尚未完成独立校准的结构化抽取；
- 图谱是轻量 JSON 关系层，不是大规模知识图谱或 Neo4j；
- 生产化仍需认证、限流、监控、CI/CD 和数据治理。

## 文档入口

- [英文 README](README.md)
- [Demo 作品集](docs/demo/README.md)
- [最终架构说明](docs/final_architecture.md)
- [R0/R3 评测决策案例](docs/demo/evaluation_case_study.md)
- [最终评测报告](eval/report/final/FINAL_EVALUATION.md)
- [GitHub 上传指南（中文）](docs/GITHUB_UPLOAD_GUIDE_zh-CN.md)

## 项目定位

推荐在简历中写为：

> **AutoSales-Agent：面向新能源汽车销售咨询的多工具对话 Agent 工程化原型**

不要包装为“生产级智能销售系统”。更准确的说法是：实现并评测了具备 RAG、工具调用、确定性计算、证据溯源、故障降级、持久化记忆与 API 服务能力的多工具 Agent 原型。
