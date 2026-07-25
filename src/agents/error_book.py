"""错题集系统 — 记录低质量回答，支持相似错题检索和修正提示

错题集用于记录 RAG Pipeline 中产生的低质量回答，自动分类错误类型，
并在后续检索中提供修正提示，避免重复犯错。

错误类型分类（按 Pipeline 阶段优先级）：
  1. knowledge_gap  — 知识库无匹配（retrieved_docs 为空，0条检索）
  2. grading_error  — 文档评分失败（relevant_count == 0，0条相关）
  3. hallucination  — 幻觉（hallucination_score > 0.3）
  4. fact_check_fail— 事实校验未通过（fact_check_passed == False）

存储格式：JSON 文件，每条记录包含：
  id, timestamp, question, answer, error_type, metrics, corrected, correction
"""
import json
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.config import DATA_DIR, EMBEDDING_MODEL, DEVICE

log = logging.getLogger("deeprag.error_book")

# 错题集默认存储路径
ERROR_BOOK_PATH = DATA_DIR / "error_book.json"

# 幻觉判定阈值
HALLUCINATION_THRESHOLD = 0.3


class ErrorBook:
    """错题集系统 — 记录、分类、检索历史错误

    用法示例::

        book = ErrorBook()
        book.record("什么是RAG？", "RAG是...", state)
        hint = book.get_correction_hint("什么是RAG？")
    """

    def __init__(self, storage_path: str = None):
        """初始化错题集

        Args:
            storage_path: JSON 存储路径，默认 data/error_book.json
        """
        self.storage_path = Path(storage_path) if storage_path else ERROR_BOOK_PATH
        self.records: list[dict] = []
        self._embedder = None
        self.load()

    # ------------------------------------------------------------------
    #  持久化
    # ------------------------------------------------------------------

    def load(self):
        """从 JSON 文件加载历史错题记录"""
        try:
            if self.storage_path.exists():
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    self.records = json.load(f)
                log.info("错题集加载成功: %d 条记录 (%s)", len(self.records), self.storage_path)
            else:
                self.records = []
                log.info("错题集文件不存在，初始化空记录: %s", self.storage_path)
        except (json.JSONDecodeError, OSError) as e:
            log.error("错题集加载失败: %s，初始化空记录", e)
            self.records = []

    def save(self):
        """将错题记录持久化到 JSON 文件"""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(self.records, f, ensure_ascii=False, indent=2)
            log.info("错题集保存成功: %d 条记录 (%s)", len(self.records), self.storage_path)
        except OSError as e:
            log.error("错题集保存失败: %s", e)

    # ------------------------------------------------------------------
    #  记录与分类
    # ------------------------------------------------------------------

    def record(self, question: str, answer: str, state: dict):
        """记录一条低质量回答

        Args:
            question: 用户原始提问
            answer:   系统生成的（低质量）回答
            state:    RAGState 状态字典，包含 retrieved_docs、relevant_count、
                      hallucination_score、fact_check_passed 等字段
        """
        error_type = self._classify_error(state)

        # 提取关键指标快照
        metrics = {
            "retrieved_count": len(state.get("retrieved_docs", [])),
            "relevant_count": state.get("relevant_count", 0),
            "hallucination_score": state.get("hallucination_score", 0.0),
            "fact_check_passed": state.get("fact_check_passed", False),
        }

        record = {
            "id": hashlib.md5(question.encode("utf-8")).hexdigest()[:16],
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "question": question,
            "answer": answer,
            "error_type": error_type,
            "metrics": metrics,
            "corrected": False,
            "correction": "",
        }

        self.records.append(record)
        self.save()
        log.warning(
            "记录错题 [id=%s, type=%s]: %s",
            record["id"], error_type, question[:80],
        )
        return record

    def _classify_error(self, state: dict) -> str:
        """根据 RAGState 自动分类错误类型

        按 Pipeline 阶段从早到晚判断，返回第一个命中的错误类型：
          knowledge_gap → grading_error → hallucination → fact_check_fail

        Args:
            state: RAGState 状态字典

        Returns:
            错误类型字符串
        """
        # 1. 知识库无匹配：检索阶段返回 0 条文档
        retrieved_docs = state.get("retrieved_docs", [])
        if len(retrieved_docs) == 0:
            return "knowledge_gap"

        # 2. 文档评分失败：检索到文档但没有一条相关
        relevant_count = state.get("relevant_count", 0)
        if relevant_count == 0:
            return "grading_error"

        # 3. 幻觉：生成阶段幻觉评分超过阈值
        hallucination_score = state.get("hallucination_score", 0.0)
        if hallucination_score > HALLUCINATION_THRESHOLD:
            return "hallucination"

        # 4. 事实校验未通过
        fact_check_passed = state.get("fact_check_passed", True)
        if not fact_check_passed:
            return "fact_check_fail"

        # 未识别到已知错误类型，标记为 unknown
        return "unknown"

    # ------------------------------------------------------------------
    #  相似错题检索
    # ------------------------------------------------------------------

    def _get_embedder(self):
        """延迟加载 SentenceTransformer 嵌入模型（与 Indexer 共享缓存）"""
        if self._embedder is None:
            from src.ui.model_cache import get_embedding_model
            self._embedder = get_embedding_model(EMBEDDING_MODEL, DEVICE)
        return self._embedder

    def get_similar_errors(self, question: str, top_k: int = 3) -> list[dict]:
        """用 embedding 余弦相似度查找历史错题

        Args:
            question: 查询问题
            top_k:    返回最相似的 top_k 条记录

        Returns:
            相似错题记录列表（按相似度降序），每条额外附带 similarity 字段
        """
        if not self.records:
            return []

        try:
            embedder = self._get_embedder()
            query_emb = embedder.encode([question])[0]

            # 收集历史问题的 embedding（缓存命中则复用）
            history_embs = []
            for r in self.records:
                if "_embedding" not in r:
                    r["_embedding"] = embedder.encode([r["question"]])[0].tolist()
                history_embs.append(r["_embedding"])

            # 计算余弦相似度
            import numpy as np
            query_vec = np.array(query_emb)
            history_vecs = np.array(history_embs)
            # 余弦相似度 = dot(a,b) / (||a|| * ||b||)
            norms = np.linalg.norm(history_vecs, axis=1) * np.linalg.norm(query_vec)
            norms[norms == 0] = 1e-10  # 避免除零
            similarities = history_vecs.dot(query_vec) / norms

            # 取 top_k
            top_indices = np.argsort(similarities)[::-1][:top_k]
            results = []
            for idx in top_indices:
                if similarities[idx] > 0:  # 只返回正相关的
                    record = self.records[idx].copy()
                    record["similarity"] = round(float(similarities[idx]), 4)
                    record.pop("_embedding", None)  # 返回时移除内部缓存字段
                    results.append(record)

            log.info("相似错题检索: query='%s...' → %d 条匹配", question[:40], len(results))
            return results

        except Exception as e:
            log.error("相似错题检索失败: %s", e)
            return []

    # ------------------------------------------------------------------
    #  修正提示
    # ------------------------------------------------------------------

    def get_correction_hint(self, question: str) -> str:
        """获取修正提示（基于历史相似错题）

        当存在已修正的相似错题时，返回修正内容作为提示；
        若无已修正记录，返回错误类型提示，帮助 Agent 在本轮检索中规避同类错误。

        Args:
            question: 当前用户提问

        Returns:
            修正提示字符串；无匹配时返回空字符串
        """
        similar = self.get_similar_errors(question, top_k=3)
        if not similar:
            return ""

        hints = []
        for rec in similar:
            error_type = rec.get("error_type", "unknown")
            similarity = rec.get("similarity", 0)

            # 优先使用已修正的错题
            if rec.get("corrected") and rec.get("correction"):
                hints.append(
                    f"[历史错题-已修正] 相似度={similarity:.2f} "
                    f"错误类型={error_type}\n"
                    f"  原问题: {rec['question'][:100]}\n"
                    f"  修正建议: {rec['correction']}"
                )
            else:
                # 未修正的错题，仅提示错误类型
                hints.append(
                    f"[历史错题-未修正] 相似度={similarity:.2f} "
                    f"错误类型={error_type}\n"
                    f"  原问题: {rec['question'][:100]}"
                )

        hint_text = "\n".join(hints)
        log.info("修正提示生成: %d 条历史错题匹配", len(hints))
        return hint_text

    # ------------------------------------------------------------------
    #  修正标记
    # ------------------------------------------------------------------

    def mark_corrected(self, record_id: str, correction: str):
        """标记某条错题为已修正，并记录修正内容

        Args:
            record_id:  记录 ID（hash(question) 的前16位）
            correction: 修正后的正确回答或建议
        """
        for r in self.records:
            if r["id"] == record_id:
                r["corrected"] = True
                r["correction"] = correction
                self.save()
                log.info("错题已标记修正 [id=%s]", record_id)
                return
        log.warning("未找到错题记录 [id=%s]，无法标记修正", record_id)
