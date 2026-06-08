"""LangGraph ReAct Agent:一个会"自己决定要不要查知识库"的多步推理 Agent。

和"裸 RAG"(每次都先检索再回答)不同,这里用 LangGraph 的 create_react_agent
构建一个能 **多步 reasoning→action 循环** 的 ReAct Agent:
  - 它有两个工具:search_knowledge_base(查知识库)和 calculator(算数)。
  - 模型自己判断:要查资料就调 search,要算数就调 calculator,寒暄就直接答。
  - 能链式调用:例如"每天多少任务?再除以8"→ 先 search 拿到 768,
    观察结果后再 calculator(768/8)→ 综合作答。这就是真正的 ReAct 循环。
  - 支持多轮对话(用 thread_id 区分会话,记忆存在 checkpointer 里)。
这对应 JD 里的 ReAct / Tool-using / Function Calling / 上下文记忆。

注:多步 ReAct 的稳定性依赖较强的 LLM。实测 glm-4-air / glm-4-plus 可稳定多步链式
调用;免费的 glm-4-flash 工具规划能力不足,只适合单步,故默认用 glm-4-air。
"""
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from app.config import settings
from app.vectorstore import get_vectorstore


@tool
def search_knowledge_base(query: str) -> str:
    """当需要回答与用户上传文档/知识库相关的问题时,用本工具检索资料。
    传入要查询的问题,返回最相关的若干文本片段。"""
    store = get_vectorstore()
    results = store.similarity_search(query, k=4)
    if not results:
        return "知识库里没有检索到相关内容。"
    blocks = []
    for i, doc in enumerate(results, 1):
        src = doc.metadata.get("source", "未知来源")
        page = doc.metadata.get("page")
        cite = f"{src}" + (f" 第{page+1}页" if isinstance(page, int) else "")
        blocks.append(f"[片段{i} | {cite}]\n{doc.page_content}")
    return "\n\n".join(blocks)


@tool
def calculator(expression: str) -> str:
    """做基础数学计算。传入一个 Python 算术表达式,如 '12*(3+4)'。"""
    allowed = set("0123456789+-*/.() ")
    if not set(expression) <= allowed:
        return "表达式包含不允许的字符,只支持数字和 + - * / ( )。"
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))  # noqa: S307 已做字符白名单
    except Exception as exc:  # noqa: BLE001
        return f"计算出错:{exc}"


SYSTEM_PROMPT = (
    "你是一个知识库问答助手,可以分多步使用工具来完成任务。\n"
    "工具:\n"
    "- search_knowledge_base(query):检索用户上传的文档,返回相关片段。\n"
    "- calculator(expression):计算数学表达式。\n\n"
    "工作方式(重要):\n"
    "1. 任何涉及文档内容的事实,都必须先调用 search_knowledge_base 获取,绝不凭空编造。\n"
    "2. 如果一个问题既需要文档里的事实、又需要对该事实做计算,你必须分两步:\n"
    "   先调用 search_knowledge_base 拿到具体数字,看到检索结果后,\n"
    "   再调用 calculator 用那个真实数字做计算。不要在没拿到数字前就调用 calculator。\n"
    "3. 每次调用工具后,先阅读返回结果,再决定下一步是继续调用工具还是给出最终答案。\n"
    "4. 最终回答要基于检索到的片段,并在末尾标注引用来源;检索不到就如实说明。"
)


def build_agent():
    llm_kwargs = {
        "model": settings.llm_model,
        "api_key": settings.openai_api_key,
        "temperature": 0,
    }
    if settings.openai_base_url:
        llm_kwargs["base_url"] = settings.openai_base_url
    llm = ChatOpenAI(**llm_kwargs)
    return create_react_agent(
        llm,
        tools=[search_knowledge_base, calculator],
        prompt=SYSTEM_PROMPT,
        checkpointer=MemorySaver(),  # 多轮对话记忆(按 thread_id 区分)
    )
