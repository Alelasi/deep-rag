"""测试不同LM Studio模型的意图分类能力"""
import httpx
import json
import time
from typing import Dict, Any

def test_model(model_name: str, query: str) -> Dict[str, Any]:
    """测试单个模型"""
    
    # 简化的Prompt（更容易让模型输出JSON）
    prompt = f"""Classify this query into intent categories.

Query: {query}

L1 Intent (choose one): Knowledge Query, Realtime Query, Mixed, Refusal
L2 Intent: Concept, API Usage, Code Example, Best Practice, Error Debug, etc.

Return only JSON:
{{"intent_l1": "Knowledge Query", "intent_l2": "Concept", "confidence": 0.95}}"""

    start = time.time()
    try:
        resp = httpx.post(
            "http://localhost:11434/v1/chat/completions",
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 150,
                "temperature": 0.1
            },
            timeout=30.0
        )
        elapsed = time.time() - start
        
        if resp.status_code != 200:
            return {
                "success": False,
                "error": f"HTTP {resp.status_code}",
                "elapsed": elapsed
            }
        
        result = resp.json()
        message = result["choices"][0]["message"]
        
        # 尝试从content或reasoning_content提取
        content = message.get("content", "") or message.get("reasoning_content", "")
        
        return {
            "success": True,
            "content": content,
            "tokens": result["usage"],
            "elapsed": elapsed
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "elapsed": time.time() - start
        }

if __name__ == "__main__":
    # 测试样本
    test_queries = [
        "LangChain是什么？",
        "create_agent怎么用？",
        "LangChain最新版本是多少？"
    ]
    
    # 等待用户加载模型
    print("=" * 60)
    print("LM Studio 模型测试工具")
    print("=" * 60)
    print()
    print("请在LM Studio中加载模型，然后按Enter继续...")
    input()
    
    # 获取当前加载的模型
    try:
        resp = httpx.get("http://localhost:11434/v1/models", timeout=5.0)
        models = resp.json()["data"]
        print(f"\n可用模型数量: {len(models)}")
        print()
    except:
        print("无法获取模型列表，继续测试...")
    
    # 让用户选择要测试的模型
    print("请输入要测试的模型名（或按Enter使用当前加载的模型）:")
    model_input = input().strip()
    
    if not model_input:
        print("\n尝试自动检测当前模型...")
        # 发送一个测试请求看看用什么模型
        test_resp = httpx.post(
            "http://localhost:11434/v1/chat/completions",
            json={
                "model": "local-model",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 5
            },
            timeout=10.0
        )
        current_model = test_resp.json()["model"]
        print(f"检测到当前模型: {current_model}")
        model_name = current_model
    else:
        model_name = model_input
    
    print()
    print("=" * 60)
    print(f"测试模型: {model_name}")
    print("=" * 60)
    
    # 测试每个查询
    for i, query in enumerate(test_queries, 1):
        print(f"\n[测试 {i}/{len(test_queries)}] {query}")
        print("-" * 60)
        
        result = test_model(model_name, query)
        
        if result["success"]:
            print(f"✅ 成功 | 耗时: {result['elapsed']:.2f}s")
            print(f"Tokens: {result['tokens']}")
            print(f"\n输出内容:")
            print(result["content"][:300])
            if len(result["content"]) > 300:
                print(f"... (共 {len(result['content'])} 字符)")
        else:
            print(f"❌ 失败 | 耗时: {result['elapsed']:.2f}s")
            print(f"错误: {result['error']}")
    
    print()
    print("=" * 60)
    print("测试完成！")
