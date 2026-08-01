#!/usr/bin/env python3
"""DeepRAG 性能基准测试 — L4 层级

测量指标：
1. 响应时间：P50/P90/P99
2. 召回率：Hit@K
3. Token 使用量

用法：
    python scripts/benchmark_test.py
    python scripts/benchmark_test.py --rounds 5
"""
import argparse, json, time, sys, statistics
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

class Color:
    RESET="\033[0m";BOLD="\033[1m";RED="\033[91m";GREEN="\033[92m";YELLOW="\033[93m"
    BLUE="\033[94m";MAGENTA="\033[95m";CYAN="\033[96m";GRAY="\033[90m"

def colorize(t,c): return f"{c}{t}{Color.RESET}"

BENCHMARK_QUESTIONS = [
    {"id":1,"category":"MBTI","question":"INTJ的主导功能是什么？","expected_keywords":["Ni","内向直觉"]},
    {"id":3,"category":"MBTI","question":"INTJ和INFJ的核心区别是什么？","expected_keywords":["Te","Fe"]},
    {"id":5,"category":"MBTI","question":"INTJ在压力下会表现出哪些特征？","expected_keywords":["Se","劣势功能"]},
    {"id":9,"category":"MBTI","question":"ISTJ的性格特点是什么？","expected_keywords":["Si","责任感"]},
    {"id":21,"category":"RAG","question":"什么是RAG？","expected_keywords":["检索增强生成"]},
    {"id":23,"category":"RAG","question":"什么是混合检索？","expected_keywords":["BM25","向量"]},
    {"id":27,"category":"RAG","question":"什么是重排序（Reranking）？","expected_keywords":["精排","Cross-Encoder"]},
    {"id":33,"category":"RAG","question":"什么是Embedding模型？","expected_keywords":["向量化","bge"]},
    {"id":41,"category":"LLM","question":"什么是Transformer？","expected_keywords":["注意力机制"]},
    {"id":43,"category":"LLM","question":"什么是KV Cache？","expected_keywords":["Key","Value","缓存"]},
    {"id":46,"category":"LLM","question":"什么是Temperature参数？","expected_keywords":["随机性","采样"]},
    {"id":50,"category":"LLM","question":"什么是RLHF？","expected_keywords":["人类反馈","强化学习"]},
    {"id":61,"category":"Agent","question":"什么是Function Calling？","expected_keywords":["工具调用"]},
    {"id":62,"category":"Agent","question":"什么是MCP协议？","expected_keywords":["Model Context Protocol"]},
    {"id":66,"category":"Agent","question":"什么是ReAct模式？","expected_keywords":["推理","行动"]},
    {"id":76,"category":"Agent","question":"什么是SSE？","expected_keywords":["Server-Sent Events"]},
    {"id":81,"category":"Engineering","question":"如何优化LLM推理速度？","expected_keywords":["量化","KV Cache"]},
    {"id":83,"category":"Engineering","question":"什么是模型路由？","expected_keywords":["任务类型","选择模型"]},
    {"id":89,"category":"Engineering","question":"什么是降级策略？","expected_keywords":["备用方案","兜底"]},
    {"id":93,"category":"Engineering","question":"什么是缓存？","expected_keywords":["存储","复用"]},
]

def calculate_percentiles(latencies):
    if not latencies: return {"p50":0,"p90":0,"p99":0,"mean":0,"min":0,"max":0}
    s=sorted(latencies); n=len(s)
    def pct(p):
        if n==1: return round(s[0],3)
        k=(n-1)*p/100; f=int(k); c=f+1
        return round(s[f]+(k-f)*(s[c]-s[f]),3) if c<n else round(s[-1],3)
    return {"p50":pct(50),"p90":pct(90),"p99":pct(99),"mean":round(statistics.mean(latencies),3),"min":round(min(latencies),3),"max":round(max(latencies),3)}

def calculate_hit_at_k(answer,expected_keywords,k=5):
    if not expected_keywords: return {"hit_at_k":0,"hit_count":0,"total":0,"hits":[]}
    top_k=expected_keywords[:k]; al=answer.lower(); hits=[kw for kw in top_k if kw.lower() in al]
    return {"hit_at_k":round(len(hits)/len(top_k),4),"hit_count":len(hits),"total":len(top_k),"hits":hits}

def estimate_tokens(text):
    cn=sum(1 for c in text if '\u4e00'<=c<='\u9fff'); return int(cn*1.5+(len(text)-cn)/4)

def _try_import_rag():
    try:
        from src.graph import query as q; return q
    except: return None

def call_rag(question):
    rag_query=_try_import_rag()
    if rag_query is None: raise ImportError("src.graph.query 不可用")
    result=rag_query(question)
    if isinstance(result,dict):
        answer=result.get("answer",str(result)); sources=result.get("citations",result.get("sources",[]))
    else: answer=str(result); sources=[]
    return {"answer":answer,"sources":sources if isinstance(sources,list) else [],"error":None,"tokens_in":estimate_tokens(question),"tokens_out":estimate_tokens(answer)}

def simulate_rag(question):
    import random; time.sleep(random.uniform(0.1,0.3))
    answer=f"这是对「{question[:20]}...」的模拟回答。"
    return {"answer":answer,"sources":[],"error":None,"tokens_in":estimate_tokens(question),"tokens_out":estimate_tokens(answer)}

def run_single_benchmark(q_data,rounds=3):
    question=q_data["question"]; kws=q_data.get("expected_keywords",[])
    latencies=[];answers=[];errors=[];tin=0;tout=0;sim=False
    rag_ok=_try_import_rag() is not None
    if not rag_ok: sim=True
    for i in range(rounds):
        start=time.time()
        try:
            r=call_rag(question) if rag_ok else simulate_rag(question)
            elapsed=round(time.time()-start,3); latencies.append(elapsed); answers.append(r["answer"])
            tin+=r["tokens_in"]; tout+=r["tokens_out"]; errors.append(None)
        except Exception as e:
            elapsed=round(time.time()-start,3); latencies.append(elapsed); answers.append(f"[ERROR] {e}")
            errors.append(str(e)); rag_ok=False; sim=True
    pct=calculate_percentiles(latencies)
    valid=[a for a,e in zip(answers,errors) if e is None]
    hit=calculate_hit_at_k(valid[-1],kws,5) if valid else {"hit_at_k":0,"hit_count":0,"total":0,"hits":[]}
    return {"id":q_data["id"],"category":q_data["category"],"question":question,"latencies":latencies,"percentiles":pct,"hit_at_k":hit,"tokens_in_avg":round(tin/max(rounds,1)),"tokens_out_avg":round(tout/max(rounds,1)),"errors":[e for e in errors if e],"simulation":sim}

def run_benchmark(rounds=3):
    print(f"\n{'='*70}\n  DeepRAG 性能基准测试 (L4)\n  题目:{len(BENCHMARK_QUESTIONS)} 轮数:{rounds}\n{'='*70}")
    results=[]; start=time.time()
    for i,q in enumerate(BENCHMARK_QUESTIONS):
        print(f"  [{i+1}/{len(BENCHMARK_QUESTIONS)}] {q['category']} {q['question'][:30]}")
        r=run_single_benchmark(q,rounds); results.append(r)
        p=r["percentiles"]; h=r["hit_at_k"]
        print(f"    P50={p['p50']:.3f}s P90={p['p90']:.3f}s Hit@K={h['hit_at_k']:.2%}")
    total=time.time()-start
    all_lat=[];all_hit=[];err=0;all_sim=True
    for r in results:
        all_lat.extend(r["latencies"]); all_hit.append(r["hit_at_k"]["hit_at_k"]); err+=len(r["errors"])
        if not r["simulation"]: all_sim=False
    return {"timestamp":datetime.now().isoformat(),"config":{"questions":len(BENCHMARK_QUESTIONS),"rounds":rounds,"total_runs":len(BENCHMARK_QUESTIONS)*rounds,"simulation":all_sim},"summary":{"total_time":round(total,2),"latency":calculate_percentiles(all_lat),"avg_hit_at_k":round(statistics.mean(all_hit),4) if all_hit else 0,"total_errors":err,"error_rate":round(err/(len(BENCHMARK_QUESTIONS)*rounds),4)},"per_question":results}

def save_report(report):
    d=Path(__file__).parent.parent/"tests"/"reports"; d.mkdir(parents=True,exist_ok=True)
    ts=datetime.now().strftime("%Y%m%d_%H%M%S"); p=d/f"benchmark_{ts}.json"
    with open(p,"w",encoding="utf-8") as f: json.dump(report,f,ensure_ascii=False,indent=2)
    print(f"  报告: {p}")

def print_console_summary(report):
    s=report["summary"]; l=s["latency"]
    print(f"\n{'='*70}\n  基准测试汇总\n{'='*70}")
    print(f"  P50={l['p50']:.3f}s P90={l['p90']:.3f}s P99={l['p99']:.3f}s Mean={l['mean']:.3f}s")
    print(f"  Hit@K={s['avg_hit_at_k']:.2%} Errors={s['total_errors']} Rate={s['error_rate']:.2%}")
    if report["config"]["simulation"]: print("  [注意] 模拟模式")
    print(f"  总耗时: {s['total_time']:.1f}s")

def main():
    parser=argparse.ArgumentParser(description="DeepRAG 性能基准测试")
    parser.add_argument("--rounds",type=int,default=3,help="每题运行轮数")
    args=parser.parse_args()
    if args.rounds<1: print("错误: --rounds>=1"); sys.exit(1)
    report=run_benchmark(rounds=args.rounds); print_console_summary(report); save_report(report)
    sys.exit(1 if report["summary"]["total_errors"]>0 else 0)

if __name__=="__main__": main()
