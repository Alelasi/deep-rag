"""多轮对话一致性：相关追问借鉴 + 堆栈矛盾检测"""
from src.agents.dialog_context import (
    extract_stacks,
    extract_type_codes,
    is_related,
    build_prior_context,
    consistency_hint,
    find_contradiction,
)


def test_extract_stack_intj():
    text = "INTJ 的功能堆栈是 INTJ: Ni-Te-Fi-Se，主导为 Ni。"
    stacks = extract_stacks(text)
    assert stacks.get("INTJ") == "Ni-Te-Fi-Se"


def test_related_same_type():
    assert is_related(
        "INTJ的功能排序是什么？",
        "INTJ的主导功能是什么？",
        "INTJ: Ni-Te-Fi-Se",
    )


def test_build_prior_and_hint():
    turns = [
        {
            "q": "INTJ的主导功能是什么？",
            "a": "【直接回答】INTJ 主导功能是 Ni。完整堆栈 INTJ: Ni-Te-Fi-Se。",
        }
    ]
    prior = build_prior_context("INTJ的功能排序？", turns)
    assert "对话上下文" in prior
    assert "Ni-Te-Fi-Se" in prior
    hint = consistency_hint("INTJ的功能排序？", turns)
    assert "硬约束" in hint
    assert "Ni-Te-Fi-Se" in hint


def test_find_contradiction():
    turns = [{"q": "INTJ主导", "a": "INTJ: Ni-Te-Fi-Se"}]
    bad = "INTJ 的功能是 INTJ: Ni-Te-Si-Fe"
    contra = find_contradiction(bad, turns)
    assert contra is not None
    code, old, new = contra
    assert code == "INTJ"
    assert old == "Ni-Te-Fi-Se"
    assert new == "Ni-Te-Si-Fe"


def test_no_contradiction_same_stack():
    turns = [{"q": "INTJ主导", "a": "INTJ: Ni-Te-Fi-Se"}]
    good = "排序为 INTJ: Ni-Te-Fi-Se"
    assert find_contradiction(good, turns) is None


def test_type_codes():
    assert "INTJ" in extract_type_codes("请比较 INTJ 与 ENTJ")
