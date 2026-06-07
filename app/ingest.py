"""文档入库:把 data/ 下的 PDF / txt / md 读出来 → 切块 → 写进向量库。

这是 RAG 的 "索引" 阶段。切块(chunking)很关键:块太大检索不精准,
太小会丢上下文。这里用 RecursiveCharacterTextSplitter,按段落→句子递归切。
"""
import sys
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.vectorstore import get_vectorstore

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_documents(data_dir: Path = DATA_DIR) -> list[Document]:
    docs: list[Document] = []
    for path in sorted(data_dir.rglob("*")):
        if path.suffix.lower() == ".pdf":
            docs.extend(PyPDFLoader(str(path)).load())
        elif path.suffix.lower() in {".txt", ".md"}:
            docs.extend(TextLoader(str(path), encoding="utf-8").load())
    return docs


def chunk_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,        # 每块约 800 字符
        chunk_overlap=120,     # 块间重叠,避免句子被切断丢上下文
        separators=["\n\n", "\n", "。", "！", "？", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def ingest() -> int:
    docs = load_documents()
    if not docs:
        print(f"data/ 目录下没有找到文档(支持 .pdf/.txt/.md)。先把文件放进 {DATA_DIR}")
        return 0
    chunks = chunk_documents(docs)
    store = get_vectorstore()
    store.add_documents(chunks)
    print(f"✅ 入库完成:{len(docs)} 个文档 → {len(chunks)} 个文本块,已写入向量库。")
    return len(chunks)


if __name__ == "__main__":
    sys.exit(0 if ingest() >= 0 else 1)
