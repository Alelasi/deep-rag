"""Qdrant 稳定性测试 — 模拟之前 ChromaDB 损坏的所有场景

测试项：
1. 基础读写
2. 重启持久化（ChromaDB 损坏的主因）
3. 大批量写入
4. 并发写入
5. 强制杀进程后恢复
6. 查询期间中断
7. 集合删除重建
"""
import os
import sys
import time
import hashlib
import signal
import subprocess
import threading

os.environ['no_proxy'] = 'localhost,127.0.0.1'
os.environ['NO_PROXY'] = 'localhost,127.0.0.1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

sys.path.insert(0, '.')

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
TEST_COLLECTION = "stability_test"
VECTOR_SIZE = 768

DOCKER = r"C:\Program Files\Docker\Docker\resources\bin\docker.exe"

def get_client():
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

def docker_restart():
    """重启 Qdrant Docker 容器"""
    subprocess.run([DOCKER, "restart", "qdrant"], capture_output=True)

def gen_random_vectors(n, dim=VECTOR_SIZE):
    """生成随机向量"""
    import numpy as np
    return np.random.randn(n, dim).tolist()

def gen_random_docs(n):
    """生成随机文档"""
    return [f"测试文档{i} 内容{'A' * 100}" for i in range(n)]

def cleanup():
    """清理测试集合"""
    try:
        client = get_client()
        collections = [c.name for c in client.get_collections().collections]
        if TEST_COLLECTION in collections:
            client.delete_collection(TEST_COLLECTION)
    except Exception:
        pass

def test_1_basic_rw():
    """测试1：基础读写"""
    print("\n[测试1] 基础读写...", end=" ")
    client = get_client()
    cleanup()

    # 创建集合
    client.create_collection(
        collection_name=TEST_COLLECTION,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )

    # 写入
    docs = gen_random_docs(10)
    vectors = gen_random_vectors(10)
    points = [
        PointStruct(id=i, vector=vectors[i], payload={"content": docs[i], "doc_id": f"doc_{i}"})
        for i in range(10)
    ]
    client.upsert(collection_name=TEST_COLLECTION, points=points)

    # 读取
    info = client.get_collection(TEST_COLLECTION)
    assert info.points_count == 10, f"Expected 10, got {info.points_count}"

    # 查询
    results = client.query_points(
        collection_name=TEST_COLLECTION,
        query=vectors[0],
        limit=3,
        with_payload=True,
    )
    assert len(results.points) == 3, f"Expected 3 results, got {len(results.points)}"

    print("✅ 通过")
    return True

def test_2_restart_persistence():
    """测试2：重启持久化（ChromaDB 损坏的主因）"""
    print("\n[测试2] 重启持久化...", end=" ")
    client = get_client()

    # 写入 1000 条数据
    docs = gen_random_docs(1000)
    vectors = gen_random_vectors(1000)
    points = [
        PointStruct(id=i + 1000, vector=vectors[i], payload={"content": docs[i], "doc_id": f"doc_{i}"})
        for i in range(1000)
    ]
    client.upsert(collection_name=TEST_COLLECTION, points=points)

    info = client.get_collection(TEST_COLLECTION)
    count_before = info.points_count
    print(f"写入 {count_before} 条, ", end="")

    # 重启 Qdrant Docker
    docker_restart()
    time.sleep(10)  # 等待重启

    # 验证
    client2 = get_client()
    info2 = client2.get_collection(TEST_COLLECTION)
    count_after = info2.points_count

    if count_after == count_before:
        # 查询验证
        results = client2.query_points(
            collection_name=TEST_COLLECTION,
            query=vectors[0],
            limit=3,
            with_payload=True,
        )
        if len(results.points) == 3:
            print(f"✅ 通过 (重启后 {count_after} 条, 查询正常)")
            return True
        else:
            print(f"❌ 失败 (查询返回 {len(results.points)} 条)")
            return False
    else:
        print(f"❌ 失败 (重启前 {count_before}, 重启后 {count_after})")
        return False

def test_3_large_batch():
    """测试3：大批量写入（6000+ 条）"""
    print("\n[测试3] 大批量写入 (6000条)...", end=" ")
    client = get_client()

    # 写入 6000 条
    batch_size = 100
    total = 6000
    t0 = time.time()

    for i in range(0, total, batch_size):
        batch_docs = gen_random_docs(batch_size)
        batch_vectors = gen_random_vectors(batch_size)
        points = [
            PointStruct(
                id=2000 + i + j,  # 整数 ID
                vector=batch_vectors[j],
                payload={"content": batch_docs[j], "doc_id": f"large_{i+j}"}
            )
            for j in range(batch_size)
        ]
        client.upsert(collection_name=TEST_COLLECTION, points=points)

    elapsed = time.time() - t0
    info = client.get_collection(TEST_COLLECTION)
    count = info.points_count

    if count >= total:
        print(f"✅ 通过 ({count} 条, {elapsed:.1f}s)")
        return True
    else:
        print(f"❌ 失败 (期望 {total}, 实际 {count})")
        return False

def test_4_concurrent_write():
    """测试4：并发写入"""
    print("\n[测试4] 并发写入...", end=" ")
    client = get_client()

    errors = []
    results = []

    def write_batch(thread_id, n):
        try:
            c = get_client()
            for i in range(n):
                docs = gen_random_docs(10)
                vectors = gen_random_vectors(10)
                points = [
                    PointStruct(
                        id=10000 + thread_id * 1000 + i * 10 + j,  # 整数 ID
                        vector=vectors[j],
                        payload={"content": docs[j], "doc_id": f"thread_{thread_id}_{i}_{j}"}
                    )
                    for j in range(10)
                ]
                c.upsert(collection_name=TEST_COLLECTION, points=points)
            results.append(thread_id)
        except Exception as e:
            errors.append((thread_id, str(e)))

    # 启动 5 个并发线程
    threads = []
    for t in range(5):
        thread = threading.Thread(target=write_batch, args=(t, 5))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    if not errors:
        print(f"✅ 通过 ({len(results)} 个线程完成)")
        return True
    else:
        print(f"❌ 失败 ({len(errors)} 个错误: {errors[:2]})")
        return False

def test_5_kill_recovery():
    """测试5：强制杀进程后恢复"""
    print("\n[测试5] 强制杀进程后恢复...", end=" ")

    # 记录当前数据量
    client = get_client()
    info_before = client.get_collection(TEST_COLLECTION)
    count_before = info_before.points_count

    # 启动一个写入进程
    write_script = '''
import os, sys, time
os.environ['no_proxy'] = 'localhost,127.0.0.1'
os.environ['NO_PROXY'] = 'localhost,127.0.0.1'
sys.path.insert(0, '.')
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
import numpy as np

client = QdrantClient(host='localhost', port=6333)
for i in range(100):
    vectors = np.random.randn(10, 768).tolist()
    points = [
        PointStruct(
            id=50000 + i * 10 + j,
            vector=vectors[j],
            payload={"content": f"kill test {i}_{j}", "doc_id": f"kill_test_{i}_{j}"}
        )
        for j in range(10)
    ]
    client.upsert(collection_name='stability_test', points=points)
    time.sleep(0.01)
'''
    proc = subprocess.Popen([sys.executable, '-c', write_script])

    # 写入一部分后强制杀掉
    time.sleep(0.5)
    proc.kill()
    proc.wait()

    # 验证数据完整性
    time.sleep(2)
    client2 = get_client()
    info_after = client2.get_collection(TEST_COLLECTION)
    count_after = info_after.points_count

    # 查询验证
    try:
        import numpy as np
        qvec = np.random.randn(768).tolist()
        results = client2.query_points(
            collection_name=TEST_COLLECTION,
            query=qvec,
            limit=3,
            with_payload=True,
        )
        query_ok = len(results.points) == 3
    except Exception:
        query_ok = False

    if query_ok and count_after >= count_before:
        print(f"✅ 通过 (杀进程前 {count_before}, 杀进程后 {count_after}, 查询正常)")
        return True
    else:
        print(f"❌ 失败 (杀进程前 {count_before}, 杀进程后 {count_after}, 查询 {query_ok})")
        return False

def test_6_delete_recreate():
    """测试6：删除集合后重建"""
    print("\n[测试6] 删除集合后重建...", end=" ")
    client = get_client()

    # 删除
    client.delete_collection(TEST_COLLECTION)

    # 重建
    client.create_collection(
        collection_name=TEST_COLLECTION,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )

    # 写入
    docs = gen_random_docs(50)
    vectors = gen_random_vectors(50)
    points = [
        PointStruct(id=i, vector=vectors[i], payload={"content": docs[i], "doc_id": f"recreate_{i}"})
        for i in range(50)
    ]
    client.upsert(collection_name=TEST_COLLECTION, points=points)

    # 验证
    info = client.get_collection(TEST_COLLECTION)
    if info.points_count == 50:
        print(f"✅ 通过 (重建后 {info.points_count} 条)")
        return True
    else:
        print(f"❌ 失败 (期望 50, 实际 {info.points_count})")
        return False

def test_7_query_during_write():
    """测试7：写入期间查询"""
    print("\n[测试7] 写入期间查询...", end=" ")

    errors = []

    def writer():
        try:
            c = get_client()
            import numpy as np
            for i in range(50):
                vectors = np.random.randn(10, 768).tolist()
                points = [
                    PointStruct(
                        id=60000 + i * 10 + j,
                        vector=vectors[j],
                        payload={"content": f"concurrent {i}_{j}", "doc_id": f"concurrent_{i}_{j}"}
                    )
                    for j in range(10)
                ]
                c.upsert(collection_name=TEST_COLLECTION, points=points)
                time.sleep(0.01)
        except Exception as e:
            errors.append(f"writer: {e}")

    def reader():
        try:
            c = get_client()
            import numpy as np
            for i in range(50):
                qvec = np.random.randn(768).tolist()
                results = c.query_points(
                    collection_name=TEST_COLLECTION,
                    query=qvec,
                    limit=3,
                    with_payload=True,
                )
                time.sleep(0.01)
        except Exception as e:
            errors.append(f"reader: {e}")

    # 同时启动写入和查询
    writer_thread = threading.Thread(target=writer)
    reader_thread = threading.Thread(target=reader)
    writer_thread.start()
    reader_thread.start()
    writer_thread.join()
    reader_thread.join()

    if not errors:
        print("✅ 通过 (并发读写无错误)")
        return True
    else:
        print(f"❌ 失败 ({len(errors)} 个错误: {errors[:2]})")
        return False

def main():
    print("=" * 60)
    print("Qdrant 稳定性测试")
    print("=" * 60)

    results = []
    results.append(("基础读写", test_1_basic_rw()))
    results.append(("重启持久化", test_2_restart_persistence()))
    results.append(("大批量写入", test_3_large_batch()))
    results.append(("并发写入", test_4_concurrent_write()))
    results.append(("强制杀进程恢复", test_5_kill_recovery()))
    results.append(("删除重建", test_6_delete_recreate()))
    results.append(("写入期间查询", test_7_query_during_write()))

    # 清理
    cleanup()

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
        print("\n🎉 所有测试通过！Qdrant 稳定性验证完成。")
    else:
        print("\n⚠️ 部分测试失败，需要进一步调查。")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
