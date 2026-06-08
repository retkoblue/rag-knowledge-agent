"""calculator 工具:纯逻辑,可直接真测(不联网)。"""
from app.agent import calculator


def test_basic_arithmetic():
    assert calculator.invoke({"expression": "96*8"}) == "768"
    assert calculator.invoke({"expression": "768 / 8"}) == "96.0"
    assert calculator.invoke({"expression": "(3+4)*2"}) == "14"


def test_rejects_non_math_characters():
    # 字符白名单:含字母/下划线应被拒,防止 eval 注入
    out = calculator.invoke({"expression": "__import__('os').system('ls')"})
    assert "不允许" in out


def test_handles_eval_error_gracefully():
    out = calculator.invoke({"expression": "1/0"})
    assert "计算出错" in out
