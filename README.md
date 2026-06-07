# RAG 知识库问答 Agent

基于 **LangChain + LangGraph + Qdrant + FastAPI** 的检索增强生成(RAG)问答系统。
上传文档 → 自动切块入向量库 → 用 ReAct Agent 检索 + 工具调用 + 多轮对话作答,并标注引用来源。

LLM 层走 **OpenAI 兼容接口**,可一键切换 OpenAI / 智谱 GLM / 通义千问等(只改 `.env` 里的 base_url + 模型名)。

## 技术栈

| 能力 | 选型 |
|------|------|
| 语言 | Python 3.12 |
| Agent 编排 | LangGraph(ReAct Agent + Function Calling + 多轮记忆) |
| RAG | LangChain:文档加载 → 递归切块 → 向量检索 |
| 向量数据库 | Qdrant(嵌入式本地模式,无需 Docker;维度自动探测) |
| LLM / Embedding | OpenAI 兼容(默认智谱 `glm-4-flash` + `embedding-3`,也支持 OpenAI / 通义) |
| 后端 | FastAPI + Uvicorn |

## 架构

```
data/ (你的文档)
   │  ingest.py: 加载 → 切块(800字/120重叠)
   ▼
Qdrant 向量库 (text-embedding-004 → 768维)
   ▲  similarity_search(k=4)
   │
LangGraph ReAct Agent ── tools: search_knowledge_base / calculator
   │  (模型自主决定是否检索 / 算数,checkpointer 存多轮记忆)
   ▼
FastAPI  /chat  /ingest  /health
```

## 快速开始

```bash
# 1. 装依赖(已建好 .venv)
source .venv/bin/activate
pip install -r requirements.txt

# 2. 配置 API Key
cp .env.example .env
# 编辑 .env,填入 OPENAI_API_KEY(可用智谱/通义/OpenAI 的 key)
# 智谱(免费 glm-4-flash):https://open.bigmodel.cn  → base_url 用 https://open.bigmodel.cn/api/paas/v4

# 3. 把文档放进 data/(已放了一个示例 sample-buildsom.md),然后入库
python -m app.ingest

# 4. 启动 API
bash scripts/run_api.sh
# 打开 http://127.0.0.1:8000/docs 交互测试
```

## 试一下

```bash
# 入库
curl -X POST http://127.0.0.1:8000/ingest

# 提问(会自动检索知识库并标注来源)
curl -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"BuildSOM 的 SOV 指标是什么意思?","thread_id":"u1"}'

# 多轮(同一 thread_id,带上下文)
curl -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"那它每天创建多少个采集任务?","thread_id":"u1"}'

# 工具调用(算数,会走 calculator 工具)
curl -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"96 乘以 8 等于多少?","thread_id":"u1"}'
```

## 简历可写的点

- 用 **LangGraph** 构建 ReAct Agent,实现 **Function Calling** 自主工具调用与多轮对话记忆。
- 基于 **Qdrant 向量数据库** + Embedding 实现 **RAG** 检索,递归分块 + 重叠保上下文,答案带引用溯源,向量维度自动探测以适配不同模型。
- LLM 层抽象为 **OpenAI 兼容接口**,一份代码可切换 OpenAI / 智谱 / 通义,降低供应商绑定。
- **FastAPI** 暴露 REST 接口,启动期校验配置,Swagger 自动文档。
