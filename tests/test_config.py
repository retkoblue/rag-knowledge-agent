"""配置校验:validate() 应在缺 key / 占位 key 时抛错。"""
import pytest

from app.config import Settings


def test_validate_raises_on_empty_key():
    s = Settings()
    s.openai_api_key = ""
    with pytest.raises(RuntimeError):
        s.validate()


def test_validate_raises_on_placeholder_key():
    s = Settings()
    s.openai_api_key = "your-openai-api-key-here"
    with pytest.raises(RuntimeError):
        s.validate()


def test_validate_passes_with_real_key():
    s = Settings()
    s.openai_api_key = "sk-real-looking-key"
    s.validate()  # 不抛错即通过
