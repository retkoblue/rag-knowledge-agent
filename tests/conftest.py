"""pytest 全局配置。

在任何 app 模块导入前注入一个假的 API Key,这样:
- config 的校验不会因缺 key 而炸;
- ChatOpenAI / OpenAIEmbeddings 能构造(构造不联网,联网才在调用时)。
真正需要网络的地方,各测试用 mock 替身,保证 CI 无 key 也能跑、不花钱、不联网。
"""
import os

os.environ.setdefault("OPENAI_API_KEY", "test-dummy-key-not-real")
os.environ.setdefault("OPENAI_BASE_URL", "https://example.invalid/v1")
os.environ.setdefault("QDRANT_PATH", "/tmp/rag-test-qdrant")
