# 交互式对话记忆系统

双层记忆（**JSON Cards + 历史对话块检索**）+ Agentic RAG。支持跨会话回忆、分散信息整合与冲突指令处理；提供 **Web UI + API** 与 **命令行交互** 两种使用方式。

关键词：RAG / Memory Agent / Long-term Memory / JSON Cards / Vector Search / Conflict Resolution / FastAPI / ZhipuAI

## 功能概览
- **对话持久化**：每轮用户/助手消息落盘到 `memory_data/conversation.json`
- **两类记忆**
  - **Advanced JSON Cards**：结构化、相对稳定的核心事实（`memory_data/cards.json`）
  - **历史对话块检索**：按窗口分块，带上下文前缀摘要
- **两种检索方式**
  - **TF‑IDF（离线可用）**：生成/使用 `memory_data/index.json`（命令行 `/search`）
  - **向量检索（需智谱 Embeddings）**：生成/使用 `memory_data/vector_index.json`（Web/RAG 与命令行 `/search_vec`、`/ask`）
- **RAG 问答**：向量检索 TopK + JSON Cards + 智谱 Chat 生成答案（命令行 `/ask`，Web 的 `mode=rag`）

## 快速开始：Web UI（浏览器对话）

安装依赖：

```bash
pip install -r requirements.txt
```

设置环境变量（Web 无论 `rag/chat` 模式都需要智谱 Key）：

```bat
set ZHIPU_API_KEY=你的Key
```

启动 Web 服务：

```bash
python -m uvicorn web_server:app --host 127.0.0.1 --port 8000
```

打开浏览器：`http://127.0.0.1:8000/`

Web UI 说明：
- **模式**
  - `记忆增强（RAG）`：自动（按需）构建/更新向量索引，并展示本次引用的记忆块
  - `普通对话（Chat）`：不走检索，只做模型对话
- **TopK**：RAG 模式返回并展示的记忆块数量

相关 API（给二次开发用）：
- `POST /api/chat`：聊天（`mode=rag|chat`，`top_k` 控制检索条数）
- `POST /api/search`：仅向量检索
- `GET /api/cards`、`POST /api/cards`：读写 JSON Cards

## 快速开始：命令行（交互式）

命令行入口不依赖 Web：

```bash
python memory_app.py
```

常用命令：
- `/help`：显示帮助
- `/user <内容>`：追加用户发言
- `/assistant <内容>`：追加助手发言
- `/reindex [窗口大小]`：重建 **TF‑IDF** 索引（默认 20）
- `/search <查询>`：用 **TF‑IDF** 检索（不需要 `ZHIPU_API_KEY`）
- `/reindex_vec [窗口大小]`：重建 **向量** 索引（默认 20，需要 `ZHIPU_API_KEY`）
- `/search_vec <查询>`：向量检索（需要 `ZHIPU_API_KEY`）
- `/ask <问题>`：RAG 问答（需要 `ZHIPU_API_KEY`）
- `/card <json>`：写入 Advanced JSON Card
- `/facts`：查看所有 JSON Cards
- `/exit`：保存并退出

JSON Card 示例：

```
/card {"card_id":"travel","facts":{"护照过期":"2025-02","已预订":"1月15日飞东京"}}
```


## 上下文前缀（分块摘要）说明

索引阶段会为每个对话块生成上下文前缀，默认格式：

```
[上下文：时间 2026-02-02T10:00:00 ~ 2026-02-02T10:05:00；人物 用户、助手；意图 预订、确认]
```

若某轮对话提供 `context_tag`，则优先使用更精确的前缀（用于冲突测试场景）。

## 智谱（ZhipuAI）配置

必填：
- `ZHIPU_API_KEY`

可选（当你的账号需要不同模型名/地址时）：
- `ZHIPU_BASE_URL`（默认 `https://open.bigmodel.cn/api/paas/v4`）
- `ZHIPU_CHAT_MODEL`（默认 `glm-4`）
- `ZHIPU_EMBED_MODEL`（默认 `embedding-3`）

Windows CMD 示例（当前终端会话生效）：

```bat
set ZHIPU_API_KEY=你的Key
set ZHIPU_CHAT_MODEL=glm-4
set ZHIPU_EMBED_MODEL=embedding-3
```

## 常见错误：embedding 请求过大（Request Entity Too Large / 1210）

说明：一次性发送到 `/embeddings` 的内容过多或单条文本太长。

本项目已做保护性处理（分批 embedding + 过大自动拆分），你也可以通过环境变量进一步调小批次/长度上限：
- `EMBED_DOC_MAX_CHARS`：每个 chunk 参与 embedding 的最大字符数（默认 8000）
- `EMBED_DOC_MIN_CHARS`：当单条仍然过大时，自动继续截断时的最小字符数（默认 400）
- `EMBED_MAX_BATCH_TEXTS`：每个 embedding 请求最多包含多少条 input（默认 16）
- `EMBED_MAX_BATCH_CHARS`：每个 embedding 请求的字符预算（默认 20000）

也可以通过调小向量索引分块大小来降低单条 input 长度：
- `VEC_CHUNK_SIZE`：每个 chunk 包含的对话轮数（默认 20）