"""集中读取环境变量配置。所有模块都从这里拿配置,避免散落各处。"""
import os
from dotenv import load_dotenv

load_dotenv()  # 读取项目根目录的 .env


class Settings:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    # 可选:自定义 OpenAI 兼容端点(如代理网关/中转),留空用官方默认
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    qdrant_path: str = os.getenv("QDRANT_PATH", "./qdrant_data")
    collection_name: str = os.getenv("COLLECTION_NAME", "knowledge_base")

    def validate(self) -> None:
        if not self.openai_api_key or self.openai_api_key.startswith("your-"):
            raise RuntimeError(
                "缺少 OPENAI_API_KEY。请在 .env 里填入你的 OpenAI API Key。"
            )


settings = Settings()
