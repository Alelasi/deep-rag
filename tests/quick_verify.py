#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速验证 Multi-Agent 优化功能
"""

import sys
import time

def test_imports():
    """测试导入"""
    print("1. 测试导入...")
    try:
        from src.agents.coordination import MessageBus, SmartRouter, RouteStrategy, RetryHandler
        from src.evaluation.multi_agent_metrics import MultiAgentMetrics
        print("   ✅ 所有模块导入成功")
        return True
    except Exception as e:
        print(f"   ❌ 导入失败: {e}")
        return False

def test_message_bus():
    """测试消息总线"""
    print("\n2. 测试 MessageBus...")
    try:
        from src.agents.coordination import MessageBus

        bus = MessageBus()
        received = []

        def handler(data):
            received.append(data)

        bus.subscribe("test", handler)
        count = bus.publish("test", {"msg": "hello"})

        assert count == 1
        assert len(received) == 1
        assert received[0]["msg"] == "hello"

        bus.clear()
        print("   ✅ MessageBus 测试通过")
        return True
    except Exception as e:
        print(f"   ❌ MessageBus 测试失败: {e}")
        return False

def test_smart_router():
    """测试智能路由"""
    print("\n3. 测试 SmartRouter...")
    try:
        from src.agents.coordination import SmartRouter, RouteStrategy

        router = SmartRouter()

        # 高置信度
        strategy, _ = router.route("query", 0.9)
        assert strategy == RouteStrategy.FAST

        # 中等置信度
        strategy, _ = router.route("query", 0.6)
        assert strategy == RouteStrategy.STANDARD

        # 低置信度
        strategy, _ = router.route("query", 0.3)
        assert strategy == RouteStrategy.ENHANCED

        router.clear()
        print("   ✅ SmartRouter 测试通过")
        return True
    except Exception as e:
        print(f"   ❌ SmartRouter 测试失败: {e}")
        return False

def test_retry_handler():
    """测试重试机制"""
    print("\n4. 测试 RetryHandler...")
    try:
        from src.agents.coordination import RetryHandler

        handler = RetryHandler(max_retries=2, backoff_factor=0.1)

        # 成功执行
        result = handler.execute_with_retry(lambda: "success", "TestAgent")
        assert result == "success"

        handler.clear()
        print("   ✅ RetryHandler 测试通过")
        return True
    except Exception as e:
        print(f"   ❌ RetryHandler 测试失败: {e}")
        return False

def test_multi_agent_metrics():
    """测试评测系统"""
    print("\n5. 测试 MultiAgentMetrics...")
    try:
        from src.evaluation.multi_agent_metrics import MultiAgentMetrics

        metrics = MultiAgentMetrics()

        # 记录数据
        start = time.time()
        metrics.collaboration.log_agent_execution("Agent1", start, start + 0.5, True)
        metrics.communication.log_token_usage("Agent1", "gpt-3.5-turbo", 100, 50)
        metrics.decision.log_routing_decision("query", "fast", "fast", 0.9)

        # 生成报告
        report = metrics.generate_report()
        assert "Multi-Agent 评测报告" in report
        assert "协作效率" in report

        metrics.clear()
        print("   ✅ MultiAgentMetrics 测试通过")
        return True
    except Exception as e:
        print(f"   ❌ MultiAgentMetrics 测试失败: {e}")
        return False

def main():
    """运行所有验证"""
    print("=" * 60)
    print("Multi-Agent 优化功能快速验证")
    print("=" * 60)

    results = [
        test_imports(),
        test_message_bus(),
        test_smart_router(),
        test_retry_handler(),
        test_multi_agent_metrics()
    ]

    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"✅ 所有测试通过！({passed}/{total})")
        print("=" * 60)
        return 0
    else:
        print(f"❌ 部分测试失败！({passed}/{total})")
        print("=" * 60)
        return 1

if __name__ == "__main__":
    sys.exit(main())
