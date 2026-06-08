"""文档加载与切块:纯逻辑 + 文件系统,可真测(不联网)。"""
from pathlib import Path

from app.ingest import chunk_documents, load_documents


def test_load_documents_reads_txt_and_md(tmp_path: Path):
    (tmp_path / "a.txt").write_text("hello world", encoding="utf-8")
    (tmp_path / "b.md").write_text("# 标题\n内容", encoding="utf-8")
    (tmp_path / "ignore.png").write_bytes(b"\x89PNG")  # 非文本应被忽略

    docs = load_documents(tmp_path)
    contents = " ".join(d.page_content for d in docs)
    assert "hello world" in contents
    assert "内容" in contents


def test_chunking_splits_long_text(tmp_path: Path):
    long = "。".join(f"这是第{i}句话内容用于测试切块" for i in range(200))
    (tmp_path / "long.txt").write_text(long, encoding="utf-8")

    chunks = chunk_documents(load_documents(tmp_path))
    assert len(chunks) > 1  # 长文必被切成多块
    # 每块不应远超 chunk_size(800)+overlap 余量
    assert all(len(c.page_content) <= 1000 for c in chunks)


def test_empty_dir_returns_no_docs(tmp_path: Path):
    assert load_documents(tmp_path) == []
