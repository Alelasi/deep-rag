"""准确率评测 — 运行测试集，对比答案相似度"""
import json
import time
from pathlib import Path
from typing import List, Dict


class AccuracyTester:
    """准确率测试器"""

    def __init__(self, test_file: str = "data/test_set.json"):
        self.test_file = Path(test_file)
        self._embedder = None

    def _get_embedder(self):
        """懒加载 embedding 模型"""
        if self._embedder is None:
            from src.config import EMBEDDING_MODEL, DEVICE
            from src.ui.model_cache import get_embedding_model
            self._embedder = get_embedding_model(EMBEDDING_MODEL, DEVICE)
        return self._embedder

    def _compute_similarity(self, answer1: str, answer2: str) -> float:
        """计算两个答案的余弦相似度

        Args:
            answer1: 答案1
            answer2: 答案2

        Returns:
            相似度 (0-1)
        """
        if not answer1 or not answer2:
            return 0.0
        import numpy as np
        embedder = self._get_embedder()
        embeddings = embedder.encode([answer1, answer2])
        # 余弦相似度
        vec1 = embeddings[0]
        vec2 = embeddings[1]
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return float(similarity)

    def run_test_set(self, test_file: str = None, mode: str = "enhanced",
                     subset: str = "validation", limit: int = None) -> dict:
        """运行测试集

        Args:
            test_file: 测试集文件路径
            mode: RAG 模式（enhanced / agentic / naive）
            subset: 使用训练集还是验证集（train / validation）
            limit: 限制测试数量

        Returns:
            {total, passed, accuracy, avg_similarity, avg_time, failures}
        """
        test_file = test_file or str(self.test_file)
        if not Path(test_file).exists():
            return {"error": f"测试集文件不存在: {test_file}"}

        with open(test_file, "r", encoding="utf-8") as f:
            test_data = json.load(f)

        questions = test_data.get(subset, [])
        if limit:
            questions = questions[:limit]

        if not questions:
            return {"error": f"测试集 {subset} 为空"}

        from src.graph import query

        results = []
        passed = 0
        total_similarity = 0
        total_time = 0
        failures = []

        for i, item in enumerate(questions):
            q = item["question"]
            expected = item.get("expected_answer", "")

            try:
                start_time = time.time()
                state = query(q, mode=mode)
                elapsed = time.time() - start_time

                answer = state.get("answer", "")
                similarity = self._compute_similarity(answer, expected)
                total_similarity += similarity
                total_time += elapsed

                is_passed = similarity >= 0.5  # 相似度 >= 0.5 算通过
                if is_passed:
                    passed += 1
                else:
                    failures.append({
                        "question": q,
                        "expected": expected[:200],
                        "actual": answer[:200],
                        "similarity": round(similarity, 3),
                    })

                results.append({
                    "question": q,
                    "similarity": round(similarity, 3),
                    "time": round(elapsed, 2),
                    "passed": is_passed,
                })

                if (i + 1) % 10 == 0:
                    print(f"  进度: {i + 1}/{len(questions)}, 通过率: {passed / (i + 1) * 100:.1f}%")

            except Exception as e:
                failures.append({
                    "question": q,
                    "error": str(e),
                })

        total = len(questions)
        return {
            "total": total,
            "passed": passed,
            "accuracy": round(passed / total * 100, 1) if total > 0 else 0,
            "avg_similarity": round(total_similarity / total, 3) if total > 0 else 0,
            "avg_time": round(total_time / total, 2) if total > 0 else 0,
            "failures": failures[:20],  # 最多返回20个失败案例
            "mode": mode,
            "subset": subset,
        }


if __name__ == "__main__":
    tester = AccuracyTester()
    result = tester.run_test_set(subset="validation", limit=20)
    print(f"\n=== 测试结果 ===")
    print(f"总数: {result['total']}")
    print(f"通过: {result['passed']}")
    print(f"准确率: {result['accuracy']}%")
    print(f"平均相似度: {result['avg_similarity']}")
    print(f"平均耗时: {result['avg_time']}s")
    if result.get("failures"):
        print(f"\n失败案例 ({len(result['failures'])} 个):")
        for f in result["failures"][:5]:
            print(f"  Q: {f['question'][:50]}...")
            print(f"  相似度: {f.get('similarity', 'N/A')}")
