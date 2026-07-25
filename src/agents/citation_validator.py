"""引用验证器 — v2.9.1新增

Forced Citation 后置验证：
- 提取答案中的 [N] 引用标记
- 检查每个断言（句子）是否有引用
- 检查引用编号是否有效（对应实际文档）
- 返回验证结果（有效引用/未引用断言/无效引用/引用率）
"""
import re
import logging
from dataclasses import dataclass, field

log = logging.getLogger("deeprag")


@dataclass
class CitationValidation:
    """引用验证结果"""
    valid_citations: list[str] = field(default_factory=list)    # 有效引用编号
    orphan_claims: list[str] = field(default_factory=list)      # 未引用断言
    invalid_refs: list[str] = field(default_factory=list)       # 无效引用（编号越界）
    citation_rate: float = 0.0                                  # 引用率 = 有引用句子/总句子
    total_sentences: int = 0                                    # 总断言句子数
    total_cited: int = 0                                        # 有引用的句子数

    def to_dict(self) -> dict:
        return {
            "valid_citations": self.valid_citations,
            "orphan_claims": self.orphan_claims,
            "invalid_refs": self.invalid_refs,
            "citation_rate": self.citation_rate,
            "total_sentences": self.total_sentences,
            "total_cited": self.total_cited,
        }


class CitationValidator:
    """引用验证器

    用法：
        validator = CitationValidator()
        result = validator.validate(answer, num_docs=3)
        if result.orphan_claims:
            print(f"⚠️ {len(result.orphan_claims)} 条断言未标注引用")
    """

    # 过渡/总结性句子关键词（不需要引用）
    TRANSITION_WORDS = {
        "综上", "总之", "因此", "所以", "总结", "总的来说",
        "需要注意的是", "值得注意", "简而言之", "换言之",
        "In summary", "Therefore", "Thus", "Hence",
    }

    def validate(self, answer: str, num_docs: int) -> CitationValidation:
        """验证答案中的引用

        Args:
            answer: LLM生成的答案文本
            num_docs: 参考文档数量

        Returns:
            CitationValidation 验证结果
        """
        if not answer or not answer.strip():
            return CitationValidation()

        # 1. 按句子分割答案
        sentences = re.split(r'[。！？\n.]', answer)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        if not sentences:
            return CitationValidation()

        valid_citations = set()
        orphan_claims = []
        invalid_refs = set()
        total_cited = 0

        for sent in sentences:
            # 2. 提取每句的引用标记 [N]
            refs = re.findall(r'\[(\d+)\]', sent)

            if not refs:
                # 3. 无引用标记的句子 — 判断是否为过渡句
                is_transition = any(tw in sent for tw in self.TRANSITION_WORDS)
                if not is_transition:
                    # 检查是否为纯事实性断言（包含具体信息）
                    # 排除纯格式化文本（如 "---", "## 标题" 等）
                    if not sent.startswith('#') and not sent.startswith('---'):
                        orphan_claims.append(sent[:80])
            else:
                total_cited += 1
                # 4. 检查引用编号有效性
                for ref in refs:
                    n = int(ref)
                    if 1 <= n <= num_docs:
                        valid_citations.add(f"[{n}]")
                    else:
                        invalid_refs.add(f"[{n}]")

        total_factual = len(orphan_claims) + total_cited
        citation_rate = (total_cited / total_factual) if total_factual > 0 else 0.0

        return CitationValidation(
            valid_citations=sorted(valid_citations),
            orphan_claims=orphan_claims,
            invalid_refs=sorted(invalid_refs),
            citation_rate=round(citation_rate, 2),
            total_sentences=total_factual,
            total_cited=total_cited,
        )

    def format_warning(self, validation: CitationValidation) -> str:
        """生成警告文本（用于追加到答案末尾）"""
        if not validation.orphan_claims and not validation.invalid_refs:
            return ""

        parts = []
        if validation.orphan_claims:
            parts.append(
                f"⚠️ 答案中 {len(validation.orphan_claims)} 条断言未标注引用来源，请注意核实。"
            )
        if validation.invalid_refs:
            parts.append(
                f"⚠️ 答案中存在无效引用编号: {', '.join(validation.invalid_refs)}"
            )

        return "\n\n" + "\n".join(parts) if parts else ""


# 全局单例
_validator_instance: CitationValidator = None

def get_validator() -> CitationValidator:
    """获取全局验证器实例"""
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = CitationValidator()
    return _validator_instance
