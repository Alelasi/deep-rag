"""
测试模型路由器集成

运行：python tests/test_model_router_integration.py
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import os

# 设置测试环境变量
os.environ["MODEL_CANDIDATES"] = "anthropic:claude-sonnet-4,openai:gpt-4o-mini"
os.environ["ANTHROPIC_API_KEY"] = "sk-test-anthropic"
os.environ["OPENAI_API_KEY"] = "sk-test-openai"
os.environ["CIRCUIT_BREAKER_FAILURE_THRESHOLD"] = "2"
os.environ["CIRCUIT_BREAKER_OPEN_DURATION_SEC"] = "30"

from src.llm.model_router_wrapper import get_routed_llm, parse_candidates

print("="*60)
print("测试 1：解析候选模型配置")
print("="*60)

candidates = parse_candidates("anthropic:claude-sonnet-4,openai:gpt-4o-mini")
print(f"✅ 解析成功：{len(candidates)} 个候选")
for c in candidates:
    print(f"  - {c.id} ({c.provider}) - 优先级 {c.priority}")

print("\n" + "="*60)
print("测试 2：创建路由 LLM")
print("="*60)

try:
    llm = get_routed_llm(temperature=0.3)
    print("✅ 路由 LLM 创建成功")
    print(f"  - 候选模型：{[c.id for c in llm.router.candidates]}")
    print(f"  - 熔断阈值：{llm.router.circuit_breakers[llm.router.candidates[0].id].failure_threshold} 次")
    print(f"  - 熔断时长：{llm.router.circuit_breakers[llm.router.candidates[0].id].open_duration_sec} 秒")
    print(f"  - Temperature：{llm.temperature}")
except Exception as e:
    print(f"❌ 失败：{e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("测试 3：检查 LangChain 兼容性")
print("="*60)

if 'llm' in locals():
    try:
        # 检查是否有必需的方法
        assert hasattr(llm, '_generate'), "缺少 _generate 方法"
        assert hasattr(llm, '_llm_type'), "缺少 _llm_type 属性"
        print(f"✅ LangChain 兼容性检查通过")
        print(f"  - LLM Type：{llm._llm_type}")
    except AssertionError as e:
        print(f"❌ 兼容性检查失败：{e}")
else:
    print("⏭️  跳过（llm 未创建成功）")

print("\n" + "="*60)
print("测试 4：集成到 config.get_llm()")
print("="*60)

os.environ["ENABLE_MODEL_ROUTING"] = "true"

try:
    from src.config import get_llm

    llm = get_llm(temperature=0.5)
    print("✅ config.get_llm() 集成成功")
    print(f"  - 返回类型：{type(llm).__name__}")
    print(f"  - 是否为 RoutedLLM：{type(llm).__name__ == 'RoutedLLM'}")

    if hasattr(llm, 'router'):
        print(f"  - 候选模型：{[c.id for c in llm.router.candidates]}")
except Exception as e:
    print(f"❌ 集成失败：{e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("✅ 所有测试完成")
print("="*60)
