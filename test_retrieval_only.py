"""DeepRAG 检索正确性批量测试（不消耗 LLM 配额）。

1) 单元测试 MemoryBackend `in` 运算符修复（此前导致 100% 生成失败）。
2) 检索电池：用与真实管线相同的自动路由 _auto_route_collection，
   对 30 个跨库问题做混合检索（本地 embedding + BM25），验证向量库
   返回的相关文档是否命中主题（term_coverage + 人工可见片段）。
3) 生成冒烟：仅 1 题真实调用 query()，优雅捕获 429 以反映配额状态。
"""
import sys, os, time, json, re

def load_env(path=".env"):
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

load_env()
import logging
logging.basicConfig(level=logging.WARNING)

from src.retrieval.cache import TTLCache, MemoryBackend
from src.pipeline.caches import get_enhanced_retriever
from src.rag.guards import _auto_route_collection


def _doc_content(d):
    if isinstance(d, dict):
        return d.get("content") or d.get("text") or d.get("page_content") or ""
    return getattr(d, "content", "") or getattr(d, "page_content", "")


def _doc_source(d):
    if isinstance(d, dict):
        return d.get("source") or (d.get("metadata") or {}).get("source") or ""
    return getattr(d, "source", "") or ""

# ---------- 1) 缓存修复单元测试 ----------
def test_cache_contains():
    # 真实触发此前崩溃的代码路径：TTLCache.set 内部的 `key not in self._backend`
    c = TTLCache(ttl=60, max_size=4)
    c.set("a", 1)   # 首次写入
    c.set("a", 2)   # 二次写入会执行 `key not in self._backend`（原崩溃点）
    assert c.get("a") == 2, "TTLCache.set 未正确覆盖"
    assert "a" in c._backend, "MemoryBackend.__contains__ 未生效"
    print("[UNIT] TTLCache.set 的 `in` 运算符（MemoryBackend）修复 OK")

test_cache_contains()

# ---------- 工具：中文相关性 ----------
def char_bigrams(text):
    text = re.sub(r"\s+", "", text or "")
    if len(text) < 2:
        return set([text]) if text else set()
    return set(text[i:i+2] for i in range(len(text) - 1))

def term_coverage(query, doc):
    """问题中的字符 bigram 在文档里出现的比例（0~1），越高越相关。"""
    q = char_bigrams(query)
    if not q:
        return 0.0
    d = char_bigrams(doc)
    if not d:
        return 0.0
    return len(q & d) / len(q)

# ---------- 2) 检索电池 ----------
QUESTIONS = [
    ("proj_psychology", "心理学上如何应对持续的焦虑情绪？给我几条实用建议"),
    ("proj_psychology", "抑郁情绪和抑郁症的主要区别是什么？什么情况需要专业帮助"),
    ("proj_psychology", "亲密关系中常见的沟通误区有哪些？如何改善"),
    ("proj_psychology", "拖延症背后的心理机制是什么？有什么可行的克服方法"),
    ("proj_work", "如何在工作中保持长期的高效和专注？"),
    ("proj_work", "远程办公时如何避免工作和生活边界模糊？"),
    ("proj_work", "职业倦怠（burnout）的早期信号有哪些？怎么缓解"),
    ("proj_work", "如何向上级有效地汇报项目进展和争取资源？"),
    ("proj_thesis", "写学术论文时，文献综述部分应该包含哪些内容？"),
    ("proj_thesis", "定性研究和定量研究的核心区别是什么？各自适合什么场景"),
    ("proj_thesis", "如何设计研究的理论框架和假设？"),
    ("proj_social", "在陌生社交场合如何自然地开启话题并维持对话？"),
    ("proj_social", "和朋友发生矛盾时，有哪些建设性的化解方式？"),
    ("proj_social", "如何识别并远离消耗型的人际关系？"),
    ("proj_ideas", "当完全没有灵感时，有哪些方法可以刺激创意产生？"),
    ("proj_ideas", "如何判断一个新点子是否值得投入去做？"),
    ("proj_ideas", "头脑风暴时有哪些常见的陷阱需要避免？"),
    ("proj_tools", "有哪些适合个人知识管理的工具组合推荐？"),
    ("proj_tools", "如何用自动化工具减少重复性事务工作？"),
    ("proj_tools", "做读书笔记和卡片笔记（Zettelkasten）有什么好用的软件？"),
    ("proj_assistant", "如何用 AI 助手规划一天的日程并提升执行力？"),
    ("proj_assistant", "把任务拆解成可执行步骤有哪些实用框架？"),
    ("proj_assistant", "如何让个人助手帮我长期跟踪一个目标的实现进度？"),
    ("proj_worklog", "坚持写工作日志对个人成长有哪些实际好处？"),
    ("proj_worklog", "工作日志应该记录哪些关键信息才最有价值？"),
    ("proj_api_router", "设计 RESTful API 路由时应该遵循哪些最佳实践？"),
    ("proj_api_router", "API 网关和路由层各自承担什么职责？"),
    ("proj_zhesi", "如何平衡理性思考与感性体验在日常生活中的作用？"),
    ("proj_zhesi", "面对不确定性时，哲学上有哪些应对不确定性的思想资源？"),
    ("proj_zhesi", "意义感缺失时，可以从哪些角度重新构建生活的意义？"),
]

print(f"\n开始检索电池，共 {len(QUESTIONS)} 题（仅检索，不调用 LLM）...\n")
rows = []
for i, (asked_col, q) in enumerate(QUESTIONS, 1):
    t0 = time.time()
    rec = {"idx": i, "asked": asked_col, "q": q, "routed": None,
           "retrieved": 0, "top_coverage": 0.0, "top_src": "", "top_snip": "", "err": None}
    try:
        routed, notes = _auto_route_collection(q, asked_col)
        rec["routed"] = routed
        retriever = get_enhanced_retriever(routed)
        res = retriever.retrieve(q, top_k=5, mode="simple")
        docs = res.get("results", []) or []
        rec["retrieved"] = len(docs)
        if docs:
            top = docs[0]
            content = _doc_content(top)
            rec["top_coverage"] = round(term_coverage(q, content), 3)
            rec["top_src"] = _doc_source(top)
            rec["top_snip"] = content[:90].replace("\n", " ")
        dt = round(time.time() - t0, 2)
        flag = "OK " if rec["retrieved"] > 0 else "EMPTY"
        print(f"[{i:02d}/{len(QUESTIONS)}] {flag} asked={asked_col:15s} "
              f"routed={rec['routed']:15s} ret={rec['retrieved']:2d} "
              f"cov={rec['top_coverage']:.2f} {dt:5.1f}s")
        print(f"        Q: {q}")
        print(f"        top1[{rec['top_src']}]: {rec['top_snip']}")
    except Exception as e:
        rec["err"] = f"{type(e).__name__}: {str(e)[:160]}"
        print(f"[{i:02d}/{len(QUESTIONS)}] EXCEPTION {rec['err']}")
    rows.append(rec)

ok_ret = [r for r in rows if r["retrieved"] > 0]
avg_cov = round(sum(r["top_coverage"] for r in ok_ret) / max(1, len(ok_ret)), 3)
print("\n" + "=" * 64)
print("检索电池汇总")
print("=" * 64)
print(f"总题数:           {len(rows)}")
print(f"成功检索(>0):     {len(ok_ret)}  ({round(100*len(ok_ret)/len(rows),1)}%)")
print(f"平均 top1 主题命中率(term_coverage): {avg_cov}")
print(f"自动路由与请求一致率: "
      f"{round(100*sum(1 for r in rows if r['routed']==r['asked'])/len(rows),1)}%")
# 路由变化明细
chg = [(r['idx'], r['asked'], r['routed']) for r in rows if r['routed'] != r['asked']]
if chg:
    print("路由被改写:", chg)

# ---------- 3) 生成冒烟（1 题，捕获 429）----------
print("\n" + "=" * 64)
print("生成冒烟（1 题真实 LLM，验证 MemoryBackend 修复 + 配额状态）")
print("=" * 64)
from src.graph import query
try:
    r = query("用一句话解释什么是认知行为疗法（CBT）", collection_name="proj_psychology", max_retries=0)
    print("生成成功:", (r.get("answer") or "")[:200])
    print("citations:", len(r.get("citations", []) or []), "| no_knowledge:", r.get("no_knowledge"))
except Exception as e:
    msg = str(e)
    if "429" in msg or "rate" in msg.lower() or "quota" in msg.lower():
        print("!! 当前 Cerebras 免费额度（每小时请求数）已耗尽 -> 429。")
        print("   这是免费套餐的外部限制，不是代码缺陷；等配额窗口重置后可正常生成。")
    else:
        print("生成异常:", type(e).__name__, msg[:300])

with open("test_retrieval_results.json", "w", encoding="utf-8") as f:
    json.dump({"rows": rows, "avg_coverage": avg_cov,
                "ok_retrieval": len(ok_ret)}, f, ensure_ascii=False, indent=2)
print("\n明细已写入 test_retrieval_results.json")
