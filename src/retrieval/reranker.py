"""Reranker重排序模块 — v2.8.3: API优先 + CPU降级

三种模式（按优先级自动选择）：
1. API模式：SiliconCloud /v1/rerank（bge-reranker-v2-m3），~200ms，不占本地VRAM
2. CPU模式：本地bge-reranker-v2-m3跑CPU，~500ms-1s，不占VRAM
3. 降级模式：跳过rerank，返回原始top-k

v2.8.2问题：GPU模式与7B LLM争抢VRAM导致互相驱逐，单次rerank从ms级飙到14-16s
v2.8.3修复：API模式走网络，CPU模式走CPU，均不占GPU显存
"""
import json
import logging
import os
import urllib.request

logger = logging.getLogger(__name__)


def _detect_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


import functools


@functools.lru_cache(maxsize=4)
def _get_cross_encoder(model_name: str, device: str):
    """缓存CrossEncoder模型实例（仅CPU模式使用）"""
    from sentence_transformers import CrossEncoder
    return CrossEncoder(model_name, device=device)


class Reranker:
    """文档重排序器 — API优先，CPU降级

    模式选择逻辑：
    1. 如果有 SILICONFLOW_API_KEY → API模式（~200ms）
    2. 否则尝试本地CPU加载 → CPU模式（~500ms-1s）
    3. 都不行 → 降级模式（跳过rerank）
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.api_key = os.getenv("SILICONFLOW_API_KEY", "")
        self.api_url = "https://api.siliconflow.cn/v1/rerank"
        self.mode = "none"
        self.model = None

        # 优先 API 模式（API用大模型，不受本地资源限制）
        if self.api_key:
            self.mode = "api"
            logger.info("[Reranker v2.8.3] API模式 (SiliconCloud bge-reranker-v2-m3)")
        else:
            # 降级到 CPU 本地模式（用小模型bge-reranker-base，速度快）
            try:
                self.device = _detect_device(device)
                # v2.8.3: 强制CPU，避免与7B LLM争抢VRAM
                if self.device == "cuda":
                    logger.info("[Reranker v2.8.3] 检测到GPU，但强制使用CPU避免VRAM竞争")
                    self.device = "cpu"
                self.model = _get_cross_encoder(model_name, self.device)
                self.mode = "cpu"
                logger.info(
                    "[Reranker v2.8.3] CPU模式: %s (device=%s)", model_name, self.device
                )
            except Exception as e:
                logger.warning(
                    "[Reranker v2.8.3] 模型加载失败，降级跳过rerank: %s", e
                )
                self.mode = "none"

    @staticmethod
    def _get_text(doc: dict) -> str:
        return doc.get("content") or doc.get("text") or ""

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """对候选文档重排序

        Args:
            query: 查询文本
            documents: 待重排的候选文档列表
            top_k: 返回的top文档数

        Returns:
            重排后的文档列表，每个文档附加"rerank_score"字段
        """
        if not documents:
            return []

        if self.mode == "none":
            logger.warning("[Reranker] 降级模式，跳过重排序")
            return documents[:top_k]

        try:
            if self.mode == "api":
                return self._rerank_api(query, documents, top_k)
            elif self.mode == "cpu":
                return self._rerank_local(query, documents, top_k)
        except Exception as e:
            logger.error("[Reranker] 重排序失败，返回原始文档: %s", e, exc_info=True)

        return documents[:top_k]

    def _rerank_api(
        self, query: str, documents: list[dict], top_k: int
    ) -> list[dict]:
        """API模式：调用SiliconCloud rerank接口

        延迟：~200ms（网络往返），不占本地VRAM
        """
        doc_texts = [self._get_text(doc) for doc in documents]

        payload = json.dumps({
            "model": "BAAI/bge-reranker-v2-m3",
            "query": query,
            "documents": doc_texts,
            "top_n": top_k,
            "return_documents": False,
        }).encode("utf-8")

        req = urllib.request.Request(
            self.api_url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())

        results = []
        for item in data.get("results", []):
            idx = item["index"]
            score = item["relevance_score"]
            new_doc = documents[idx].copy()
            new_doc["rerank_score"] = float(score)
            results.append(new_doc)

        logger.info(
            "[Reranker API] 输入%d篇, 输出%d篇, 最高分=%.4f, 耗时~200ms",
            len(documents), len(results),
            results[0]["rerank_score"] if results else 0.0,
        )
        return results

    def _rerank_local(
        self, query: str, documents: list[dict], top_k: int
    ) -> list[dict]:
        """CPU本地模式：CrossEncoder推理

        延迟：~500ms-1s（CPU上跑5-15篇文档），不占VRAM
        """
        if self.model is None:
            return documents[:top_k]

        pairs = [(query, self._get_text(doc)) for doc in documents]
        scores = self.model.predict(pairs, batch_size=32)

        scored_docs = list(zip(scores, documents))
        scored_docs.sort(key=lambda x: -float(x[0]))

        results = []
        for score, doc in scored_docs[:top_k]:
            new_doc = doc.copy()
            new_doc["rerank_score"] = float(score)
            results.append(new_doc)

        logger.info(
            "[Reranker CPU] 输入%d篇, 输出%d篇, 最高分=%.4f",
            len(documents), len(results),
            results[0]["rerank_score"] if results else 0.0,
        )
        return results
