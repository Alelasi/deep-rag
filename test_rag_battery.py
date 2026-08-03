"""DeepRAG 批量问答测试：跨 10 个知识库各提问，验证成功率和正确性。

输出：
  - 控制台打印逐题结果与汇总指标
  - test_rag_results.json 留存全部明细
"""
import sys, os, time, json, re

# ---- 0. 加载 .env（不依赖 dotenv）----
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

if not os.getenv("CEREBRAS_API_KEY"):
    print("!! CEREBRAS_API_KEY 未加载，请检查 .env")
    sys.exit(1)

import logging
logging.basicConfig(level=logging.WARNING)  # 压低噪声

from src.graph import query

# ---- 1. 测试问题（覆盖全部 10 个 collection + 拒识类）----
QUESTIONS = [
    # proj_psychology
    ("proj_psychology", "心理学上如何应对持续的焦虑情绪？给我几条实用建议"),
    ("proj_psychology", "抑郁情绪和抑郁症的主要区别是什么？什么情况下需要寻求专业帮助"),
    ("proj_psychology", "亲密关系中常见的沟通误区有哪些？如何改善"),
    ("proj_psychology", "拖延症背后的心理机制是什么？有什么可行的克服方法"),
    # proj_work
    ("proj_work", "如何在工作中保持长期的高效和专注？"),
    ("proj_work", "远程办公时如何避免工作和生活边界模糊？"),
    ("proj_work", "职业倦怠（burnout）的早期信号有哪些？怎么缓解"),
    ("proj_work", "如何向上级有效地汇报项目进展和争取资源？"),
    # proj_thesis
    ("proj_thesis", "写一篇学术论文时，文献综述部分应该包含哪些内容？"),
    ("proj_thesis", "定性研究和定量研究的核心区别是什么？各自适合什么场景"),
    ("proj_thesis", "如何设计研究的理论框架和假设？"),
    # proj_social
    ("proj_social", "在陌生社交场合如何自然地开启话题并维持对话？"),
    ("proj_social", "和朋友发生矛盾时，有哪些建设性的化解方式？"),
    ("proj_social", "如何识别并远离消耗型的人际关系？"),
    # proj_ideas
    ("proj_ideas", "当完全没有灵感时，有哪些方法可以刺激创意产生？"),
    ("proj_ideas", "如何判断一个新点子是否值得投入去做？"),
    ("proj_ideas", "头脑风暴时有哪些常见的陷阱需要避免？"),
    # proj_tools
    ("proj_tools", "有哪些适合个人知识管理的工具组合推荐？"),
    ("proj_tools", "如何用自动化工具减少重复性事务工作？"),
    ("proj_tools", "做读书笔记和卡片笔记（Zettelkasten）有什么好用的软件？"),
    # proj_assistant
    ("proj_assistant", "如何用 AI 助手规划一天的日程并提升执行力？"),
    ("proj_assistant", "把任务拆解成可执行步骤有哪些实用框架？"),
    ("proj_assistant", "如何让个人助手帮我长期跟踪一个目标的实现进度？"),
    # proj_worklog
    ("proj_worklog", "坚持写工作日志对个人成长有哪些实际好处？"),
    ("proj_worklog", "工作日志应该记录哪些关键信息才最有价值？"),
    # proj_api_router
    ("proj_api_router", "设计 RESTful API 路由时应该遵循哪些最佳实践？"),
    ("proj_api_router", "API 网关和路由层各自承担什么职责？"),
    # proj_zhesi
    ("proj_zhesi", "如何平衡理性思考与感性体验在日常生活中的作用？"),
    ("proj_zhesi", "面对不确定性时，哲学上有哪些应对不确定性的思想资源？"),
    ("proj_zhesi", "意义感缺失时，可以从哪些角度重新构建生活的意义？"),
    # 拒识/闲聊类（验证守卫，不应消耗 LLM 或应明确拒答）
    ("default", "你是谁？你能做什么？"),
    ("default", "今天上海天气怎么样？"),  # 实时类 → 应走 web 或拒答
    ("default", "请用一句话证明你是一个会飞的大象"),  # 荒诞 → 应拒识
]

# ---- 2. 简易中文分词（按字符 bigram，用于接地性度量）----
def char_ngrams(text, n=2):
    text = re.sub(r"\s+", "", text or "")
    if len(text) < n:
        return set([text]) if text else set()
    return set(text[i:i+n] for i in range(len(text)-n+1))

def grounded_overlap(answer, docs, top_n=8):
    """答案与检索文档的字符 bigram 重叠比例，粗略衡量是否被检索内容支撑。"""
    a = char_ngrams(answer)
    if not a:
        return 0.0, 0
    d = set()
    for doc in docs[:top_n]:
        if isinstance(doc, dict):
            txt = doc.get("content") or doc.get("text") or doc.get("page_content") or ""
        else:
            txt = str(doc)
        d |= char_ngrams(txt)
    if not d:
        return 0.0, 0
    return len(a & d) / len(a | d), len(a & d)

# ---- 3. 执行 ----
results = []
print(f"开始批量测试，共 {len(QUESTIONS)} 题...\n")
for i, (col, q) in enumerate(QUESTIONS, 1):
    t0 = time.time()
    rec = {"idx": i, "collection": col, "question": q, "ok": False,
           "error": None, "answer": "", "answer_len": 0,
           "citations": 0, "retrieved": 0, "graded": 0,
           "hallucination_score": None, "no_knowledge": None,
           "grounded_jaccard": None, "overlap_ngrams": None, "latency": None}
    try:
        r = query(q, collection_name=col, max_retries=1)
        dt = round(time.time() - t0, 2)
        rec["latency"] = dt
        rec["ok"] = True
        rec["answer"] = (r.get("answer") or "").strip()
        rec["answer_len"] = len(rec["answer"])
        rec["citations"] = len(r.get("citations", []) or [])
        docs = r.get("retrieved_docs", []) or r.get("graded_docs", []) or []
        rec["retrieved"] = len(docs)
        rec["graded"] = len(r.get("graded_docs", []) or [])
        rec["hallucination_score"] = r.get("hallucination_score")
        rec["no_knowledge"] = r.get("no_knowledge")
        jac, ov = grounded_overlap(rec["answer"], docs)
        rec["grounded_jaccard"] = round(jac, 3)
        rec["overlap_ngrams"] = ov
        # 成功判定：有非空答案且答案长度合理
        rec["ok"] = bool(rec["answer"]) and rec["answer_len"] > 20
        mark = "OK " if rec["ok"] else "FAIL"
        print(f"[{i:02d}/{len(QUESTIONS)}] {mark} {col:16s} lat={dt:5.1f}s "
              f"ans={rec['answer_len']:4d}  cit={rec['citations']:2d}  "
              f"ret={rec['retrieved']:2d}  grd={rec['grounded_jaccard']}  "
              f"nk={rec['no_knowledge']}")
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        print(f"[{i:02d}/{len(QUESTIONS)}] EXCEPTION {col:16s} {rec['error']}")
    results.append(rec)

# ---- 4. 汇总 ----
ok = [r for r in results if r["ok"]]
fail = [r for r in results if not r["ok"]]
real = [r for r in results if r["collection"] != "default"]  # 排除拒识类
real_ok = [r for r in real if r["ok"]]
with_cit = [r for r in results if r["citations"] > 0]
avg_lat = round(sum(r["latency"] or 0 for r in results) / max(1, len(results)), 2)
avg_grd = round(sum(r["grounded_jaccard"] or 0 for r in ok) / max(1, len(ok)), 3)

print("\n" + "=" * 60)
print("汇总")
print("=" * 60)
print(f"总题数:           {len(results)}")
print(f"成功(有有效答案): {len(ok)}  ({round(100*len(ok)/len(results),1)}%)")
print(f"失败/异常:        {len(fail)}")
print(f"知识库类题成功:   {len(real_ok)}/{len(real)}")
print(f"带引用(citations>0): {len(with_cit)}")
print(f"平均耗时:         {avg_lat}s")
print(f"平均接地性(Jaccard): {avg_grd}")
print("\n失败/异常题:")
for r in fail:
    print(f"  - [{r['idx']}] {r['collection']} :: {r['question'][:30]} -> {r['error']}")

# ---- 5. 抽查 6 题完整答案供人工核对 ----
print("\n" + "=" * 60)
print("抽查完整答案（供人工判断正确性）")
print("=" * 60)
spot = [1, 8, 14, 19, 25, 31]  # 心理学/工作/社交/工具/哲思/拒识
by_idx = {r["idx"]: r for r in results}
for idx in spot:
    r = by_idx.get(idx)
    if not r:
        continue
    print(f"\n--- [{idx}] {r['collection']} | {r['question']}")
    ans = r["answer"] or f"(无答案 / 拒识: {r['no_knowledge']} / err: {r['error']})"
    print(ans[:900])

with open("test_rag_results.json", "w", encoding="utf-8") as f:
    json.dump({"summary": {"total": len(results), "ok": len(ok), "fail": len(fail),
                            "avg_latency": avg_lat, "avg_grounded_jaccard": avg_grd},
                "results": results}, f, ensure_ascii=False, indent=2)
print("\n明细已写入 test_rag_results.json")
