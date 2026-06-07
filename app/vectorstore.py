"""向量库封装:用嵌入式 Qdrant(本地文件,无需 Docker / 单独服务)。

向量库做的事:把文本切块后转成向量(embedding)存起来,提问时把问题也转成向量,
按余弦相似度检索出最相关的几块文本 —— 这就是 RAG 里的 "Retrieval"。
"""
from functools import lru_cache

from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from app.config import settings


@lru_cache(maxsize=1)
def get_embeddings() -> OpenAIEmbeddings:
    kwargs = {
        "model": settings.embedding_model,
        "api_key": settings.openai_api_key,
        # 非 OpenAI 官方模型(如智谱 embedding-3)不在 tiktoken 词表里,
        # 关掉本地分词长度检查,避免 tiktoken 取不到编码而报错。
        "check_embedding_ctx_length": False,
    }
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url
    return OpenAIEmbeddings(**kwargs)


@lru_cache(maxsize=1)
def get_embedding_dim() -> int:
    """探测当前 embedding 模型的输出维度(不同平台不同:OpenAI small=1536,智谱 embedding-3=2048)。"""
    return len(get_embeddings().embed_query("dimension probe"))


@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    # path= 模式 = 本地嵌入式,数据落地到磁盘,进程内运行
    return QdrantClient(path=settings.qdrant_path)


def ensure_collection() -> None:
    """首次使用时创建 collection(相当于一张向量表)。"""
    client = get_client()
    existing = {c.name for c in client.get_collections().collections}
    if settings.collection_name not in existing:
        client.create_collection(
            collection_name=settings.collection_name,
            vectors_config=VectorParams(size=get_embedding_dim(), distance=Distance.COSINE),
        )


@lru_cache(maxsize=1)
def get_vectorstore() -> QdrantVectorStore:
    ensure_collection()
    return QdrantVectorStore(
        client=get_client(),
        collection_name=settings.collection_name,
        embedding=get_embeddings(),
        # 跳过构造时的额外验证 embedding 调用(我们已用 ensure_collection 保证维度一致),
        # 既省一次网络请求,也避免每次检索都重复触发。
        validate_embeddings=False,
        validate_collection_config=False,
    )
