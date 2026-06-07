"""FastAPI 后端:把 Agent 暴露成 HTTP 接口。

接口:
  GET  /health           健康检查
  POST /ingest           触发把 data/ 下文档入库
  POST /chat             多轮对话问答(传 thread_id 维持上下文)
启动:uvicorn app.api:app --reload
文档:启动后访问 http://127.0.0.1:8000/docs (Swagger 自动生成)
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from app.agent import build_agent
from app.config import settings
from app.ingest import ingest

_agent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _agent
    settings.validate()          # 启动即校验 API Key,早失败早暴露
    _agent = build_agent()
    yield


app = FastAPI(title="RAG 知识库问答助手", version="1.0.0", lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default"   # 同一 thread_id = 同一会话,自动带历史


class ChatResponse(BaseModel):
    answer: str
    thread_id: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": settings.llm_model}


@app.post("/ingest")
def trigger_ingest() -> dict:
    n = ingest()
    return {"ingested_chunks": n}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    config = {"configurable": {"thread_id": req.thread_id}}
    result = _agent.invoke(
        {"messages": [{"role": "user", "content": req.message}]},
        config=config,
    )
    answer = result["messages"][-1].content
    return ChatResponse(answer=answer, thread_id=req.thread_id)
