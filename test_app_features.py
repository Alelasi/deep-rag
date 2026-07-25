"""前端功能自动化测试 — 测试队列、查询、缓存等所有功能"""
import os
import sys
import time

os.environ['no_proxy'] = 'localhost,127.0.0.1'
os.environ['NO_PROXY'] = 'localhost,127.0.0.1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
sys.path.insert(0, '.')

def test_1_dedup_queue():
    """测试1：队列去重"""
    print("[测试1] 队列去重...", end=" ")
    queue = []

    def add_to_queue(q, queue):
        """模拟添加到队列（带去重）"""
        if q.strip() and q.strip() not in [x["text"] for x in queue]:
            queue.append({"text": q.strip(), "selected": True})

    add_to_queue("INTJ的主导功能是什么？", queue)
    add_to_queue("INTJ的主导功能是什么？", queue)  # 重复
    add_to_queue("什么是MBTI？", queue)
    add_to_queue("什么是MBTI？", queue)  # 重复

    if len(queue) == 2:
        print(f"✅ 通过 (2个唯一问题)")
        return True
    else:
        print(f"❌ 失败 (期望2个，实际{len(queue)}个)")
        return False

def test_2_queue_select_deselect():
    """测试2：队列全选/全不选"""
    print("[测试2] 队列全选/全不选...", end=" ")
    queue = [
        {"text": "问题1", "selected": True},
        {"text": "问题2", "selected": False},
        {"text": "问题3", "selected": True},
    ]

    # 全选
    for q in queue:
        q["selected"] = True
    n_selected = sum(1 for q in queue if q["selected"])

    if n_selected != 3:
        print(f"❌ 全选失败 (期望3，实际{n_selected})")
        return False

    # 全不选
    for q in queue:
        q["selected"] = False
    n_selected = sum(1 for q in queue if q["selected"])

    if n_selected != 0:
        print(f"❌ 全不选失败 (期望0，实际{n_selected})")
        return False

    print("✅ 通过")
    return True

def test_3_queue_delete():
    """测试3：队列删除"""
    print("[测试3] 队列删除...", end=" ")
    queue = [
        {"text": "问题1", "selected": True},
        {"text": "问题2", "selected": False},
        {"text": "问题3", "selected": True},
    ]

    # 删除选中的
    queue = [q for q in queue if not q.get("selected", True)]

    if len(queue) == 1 and queue[0]["text"] == "问题2":
        print("✅ 通过")
        return True
    else:
        print(f"❌ 失败 (期望1个，实际{len(queue)}个)")
        return False

def test_4_queue_edit():
    """测试4：队列编辑"""
    print("[测试4] 队列编辑...", end=" ")
    queue = [
        {"text": "原始问题", "selected": True},
    ]

    # 编辑
    queue[0]["text"] = "修改后的问题"

    if queue[0]["text"] == "修改后的问题":
        print("✅ 通过")
        return True
    else:
        print(f"❌ 失败")
        return False

def test_5_qdrant_connection():
    """测试5：Qdrant 连接"""
    print("[测试5] Qdrant 连接...", end=" ")
    try:
        from src.retrieval.qdrant_retriever import get_qdrant_retriever
        retriever = get_qdrant_retriever()
        count = retriever.count()
        if count > 0:
            print(f"✅ 通过 ({count} 篇文档)")
            return True
        else:
            print(f"❌ 失败 (0 篇文档)")
            return False
    except Exception as e:
        print(f"❌ 失败 ({e})")
        return False

def test_6_embedding_performance():
    """测试6：Embedding 性能"""
    print("[测试6] Embedding 性能...", end=" ")
    try:
        from src.config import EMBEDDING_MODEL, DEVICE
        from src.ui.model_cache import get_embedding_model

        embedder = get_embedding_model(EMBEDDING_MODEL, DEVICE)

        # 首次加载
        t0 = time.time()
        emb1 = embedder.encode(["测试"])
        t1 = time.time()
        first_load = t1 - t0

        # 后续调用
        times = []
        for _ in range(3):
            t0 = time.time()
            embedder.encode(["测试"])
            t1 = time.time()
            times.append(t1 - t0)

        avg = sum(times) / len(times)

        if avg < 0.1:  # 平均小于100ms
            print(f"✅ 通过 (首次{first_load:.3f}s, 平均{avg:.3f}s)")
            return True
        else:
            print(f"❌ 失败 (平均{avg:.3f}s > 0.1s)")
            return False
    except Exception as e:
        print(f"❌ 失败 ({e})")
        return False

def test_7_llm_performance():
    """测试7：LLM API 性能"""
    print("[测试7] LLM API 性能...", end=" ")
    try:
        from src.config import get_llm

        llm = get_llm(temperature=0.3)
        times = []
        for _ in range(3):
            t0 = time.time()
            llm.invoke("你好")
            t1 = time.time()
            times.append(t1 - t0)

        avg = sum(times) / len(times)

        if avg < 10:  # 平均小于10秒
            print(f"✅ 通过 (平均{avg:.2f}s)")
            return True
        else:
            print(f"❌ 失败 (平均{avg:.2f}s > 10s)")
            return False
    except Exception as e:
        print(f"❌ 失败 ({e})")
        return False

def test_8_bm25_cache():
    """测试8：BM25 缓存"""
    print("[测试8] BM25 缓存...", end=" ")
    try:
        from src.retrieval.cache import get_cached_bm25
        from rank_bm25 import BM25Okapi
        import jieba

        # 首次构建
        docs = [{"content": f"测试文档{i}"} for i in range(100)]
        t0 = time.time()
        bm25 = get_cached_bm25("test_cache", lambda: BM25Okapi([list(jieba.cut(d["content"])) for d in docs]))
        t1 = time.time()
        first_build = t1 - t0

        # 缓存命中
        t2 = time.time()
        bm25_cached = get_cached_bm25("test_cache", lambda: BM25Okapi([list(jieba.cut(d["content"])) for d in docs]))
        t3 = time.time()
        cached = t3 - t2

        if cached < first_build * 0.1:  # 缓存应该快10倍以上
            print(f"✅ 通过 (首次{first_build:.3f}s, 缓存{cached:.3f}s)")
            return True
        else:
            print(f"❌ 失败 (首次{first_build:.3f}s, 缓存{cached:.3f}s)")
            return False
    except Exception as e:
        print(f"❌ 失败 ({e})")
        return False

def test_9_doc_cache():
    """测试9：文档缓存"""
    print("[测试9] 文档缓存...", end=" ")
    try:
        from src.retrieval.cache import get_cached_documents

        # 首次获取
        t0 = time.time()
        docs = get_cached_documents("test_doc_cache", lambda: [{"content": f"doc{i}"} for i in range(100)])
        t1 = time.time()
        first_fetch = t1 - t0

        # 缓存命中
        t2 = time.time()
        docs_cached = get_cached_documents("test_doc_cache", lambda: [{"content": f"doc{i}"} for i in range(100)])
        t3 = time.time()
        cached = t3 - t2

        if len(docs) == 100 and cached < first_fetch * 0.1:
            print(f"✅ 通过 (首次{first_fetch:.3f}s, 缓存{cached:.3f}s)")
            return True
        else:
            print(f"❌ 失败")
            return False
    except Exception as e:
        print(f"❌ 失败 ({e})")
        return False

def test_10_vector_search():
    """测试10：向量检索"""
    print("[测试10] 向量检索...", end=" ")
    try:
        from src.retrieval.qdrant_retriever import get_qdrant_retriever
        from src.config import EMBEDDING_MODEL, DEVICE
        from src.ui.model_cache import get_embedding_model

        retriever = get_qdrant_retriever()
        embedder = get_embedding_model(EMBEDDING_MODEL, DEVICE)

        qemb = embedder.encode(["INTJ功能排序"]).tolist()[0]
        results = retriever.search(qemb, top_k=3)

        if len(results) == 3:
            print(f"✅ 通过 (返回{len(results)}条)")
            return True
        else:
            print(f"❌ 失败 (返回{len(results)}条)")
            return False
    except Exception as e:
        print(f"❌ 失败 ({e})")
        return False

def test_11_qdrant_restart():
    """测试11：Qdrant 重启持久化"""
    print("[测试11] Qdrant 重启持久化...", end=" ")
    try:
        from src.retrieval.qdrant_retriever import get_qdrant_retriever
        import subprocess

        retriever = get_qdrant_retriever()
        count_before = retriever.count()

        # 重启 Qdrant
        docker = r"C:\Program Files\Docker\Docker\resources\bin\docker.exe"
        subprocess.run([docker, "restart", "qdrant"], capture_output=True)
        time.sleep(10)

        # 验证
        retriever2 = get_qdrant_retriever()
        count_after = retriever2.count()

        if count_before == count_after and count_after > 0:
            print(f"✅ 通过 (重启前{count_before}, 重启后{count_after})")
            return True
        else:
            print(f"❌ 失败 (重启前{count_before}, 重启后{count_after})")
            return False
    except Exception as e:
        print(f"❌ 失败 ({e})")
        return False

def main():
    print("=" * 60)
    print("前端功能自动化测试")
    print("=" * 60)

    results = []
    results.append(("队列去重", test_1_dedup_queue()))
    results.append(("队列全选/全不选", test_2_queue_select_deselect()))
    results.append(("队列删除", test_3_queue_delete()))
    results.append(("队列编辑", test_4_queue_edit()))
    results.append(("Qdrant 连接", test_5_qdrant_connection()))
    results.append(("Embedding 性能", test_6_embedding_performance()))
    results.append(("LLM API 性能", test_7_llm_performance()))
    results.append(("BM25 缓存", test_8_bm25_cache()))
    results.append(("文档缓存", test_9_doc_cache()))
    results.append(("向量检索", test_10_vector_search()))
    results.append(("Qdrant 重启持久化", test_11_qdrant_restart()))

    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    for name, ok in results:
        status = "✅ 通过" if ok else "❌ 失败"
        print(f"  {name}: {status}")
    print(f"\n总计: {passed}/{total} 通过")

    if passed == total:
        print("\n🎉 所有测试通过！")
    else:
        print("\n⚠️ 部分测试失败，需要修复。")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
