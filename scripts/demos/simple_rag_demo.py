"""
极简版RAG Agent演示 - 展示如何给本地LLM加装文件检索功能
"""
import os
import time
import httpx
from pathlib import Path

# ==================== 第1步：文件检索工具 ====================
class SimpleFileSearch:
    """简单的文件检索工具（不用向量数据库）"""

    def __init__(self, docs_dir: str):
        self.docs_dir = Path(docs_dir)
        self.documents = []

        # 加载所有文本文件
        for file_path in self.docs_dir.rglob("*.md"):
            try:
                content = file_path.read_text(encoding='utf-8')
                self.documents.append({
                    "path": str(file_path),
                    "content": content[:1000]  # 只取前1000字
                })
            except:
                pass

        print(f"✅ 加载了 {len(self.documents)} 个文档")

    def search(self, query: str, top_k: int = 3):
        """简单的关键词检索"""
        results = []
        for doc in self.documents:
            # 简单计算相关度：查询词在文档中出现的次数
            score = sum(1 for word in query.split() if word in doc["content"])
            if score > 0:
                results.append({
                    "path": doc["path"],
                    "content": doc["content"][:300],
                    "score": score
                })

        # 按分数排序
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]


# ==================== 第2步：LLM调用 ====================
class LocalLLM:
    """本地LLM客户端"""

    def __init__(self, base_url="http://localhost:11434/v1"):
        self.base_url = base_url
        self.client = httpx.Client(timeout=30.0)  # 复用连接

    def generate(self, prompt: str, max_tokens: int = 300):
        """调用LLM生成回答"""
        start = time.time()

        response = self.client.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": "google/gemma-4-e2b",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": max_tokens
            }
        )

        elapsed = time.time() - start

        if response.status_code == 200:
            answer = response.json()["choices"][0]["message"]["content"]
            return answer, elapsed
        else:
            return f"错误: {response.status_code}", elapsed


# ==================== 第3步：RAG Agent ====================
class SimpleRAGAgent:
    """简单的RAG Agent = LLM + 文件检索工具"""

    def __init__(self, docs_dir: str):
        self.file_search = SimpleFileSearch(docs_dir)
        self.llm = LocalLLM()

    def query(self, question: str, use_rag: bool = True):
        """回答问题（可选择是否使用RAG）"""

        if use_rag:
            # 第1步：检索文件
            retrieval_start = time.time()
            docs = self.file_search.search(question, top_k=3)
            retrieval_time = time.time() - retrieval_start

            # 第2步：构造上下文
            context = "\n\n".join([
                f"文档{i+1}（{doc['path']}）：\n{doc['content']}"
                for i, doc in enumerate(docs)
            ])

            # 第3步：生成回答
            prompt = f"""基于以下文档回答问题：

{context}

问题：{question}

要求：简洁专业，200字以内。"""

            answer, gen_time = self.llm.generate(prompt)

            return {
                "answer": answer,
                "retrieval_time": retrieval_time,
                "generation_time": gen_time,
                "total_time": retrieval_time + gen_time,
                "retrieved_docs": docs
            }

        else:
            # 纯LLM模式（无RAG）
            prompt = f"{question}\n\n要求：简洁专业，200字以内。"
            answer, gen_time = self.llm.generate(prompt)

            return {
                "answer": answer,
                "retrieval_time": 0.0,
                "generation_time": gen_time,
                "total_time": gen_time,
                "retrieved_docs": []
            }


# ==================== 测试 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("简单RAG Agent演示")
    print("=" * 60)

    # 创建Agent
    agent = SimpleRAGAgent("D:/文档/ai提问相关/工作/docs")

    question = "什么是RAG？"

    # 测试1：无RAG（纯LLM）
    print(f"\n【测试1】纯LLM模式")
    print(f"问题：{question}")
    result1 = agent.query(question, use_rag=False)
    print(f"\n回答：{result1['answer']}")
    print(f"生成耗时：{result1['generation_time']:.2f}s")
    print(f"总耗时：{result1['total_time']:.2f}s")

    # 测试2：RAG模式
    print(f"\n\n【测试2】RAG模式")
    print(f"问题：{question}")
    result2 = agent.query(question, use_rag=True)
    print(f"\n回答：{result2['answer']}")
    print(f"检索耗时：{result2['retrieval_time']:.3f}s")
    print(f"生成耗时：{result2['generation_time']:.2f}s")
    print(f"总耗时：{result2['total_time']:.2f}s")
    print(f"\n检索到的文档：")
    for i, doc in enumerate(result2['retrieved_docs'], 1):
        print(f"  {i}. {doc['path']} (相关度: {doc['score']})")
