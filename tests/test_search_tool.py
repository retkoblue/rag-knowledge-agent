"""检索工具:用 mock 替身代替向量库,验证格式化与引用溯源逻辑(不联网)。"""
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

import app.agent as agent_mod


def _fake_store(docs):
    store = MagicMock()
    store.similarity_search.return_value = docs
    return store


def test_search_formats_results_with_citation():
    docs = [
        Document(page_content="BuildSOM 监测 8 个 AI 平台。",
                 metadata={"source": "doc.md", "page": 0}),
        Document(page_content="SOV 是声量份额。", metadata={"source": "doc.md"}),
    ]
    with patch.object(agent_mod, "get_vectorstore", return_value=_fake_store(docs)):
        out = agent_mod.search_knowledge_base.invoke({"query": "平台"})

    assert "BuildSOM 监测 8 个 AI 平台" in out
    assert "doc.md" in out          # 带来源
    assert "第1页" in out           # page=0 → 显示第1页(1-indexed)
    assert "片段1" in out and "片段2" in out


def test_search_handles_no_results():
    with patch.object(agent_mod, "get_vectorstore", return_value=_fake_store([])):
        out = agent_mod.search_knowledge_base.invoke({"query": "不存在"})
    assert "没有检索到" in out
