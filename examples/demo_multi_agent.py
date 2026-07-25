#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi-Agent 优化功能演示

展示如何使用：
1. MessageBus（消息总线）
2. SmartRouter（智能路由）
3. RetryHandler（重试机制）
4. MultiAgentMetrics（评测系统）
"""

import time
from src.agents.coordination import MessageBus, SmartRouter, RouteStrategy, RetryHandler
from src.evaluation.multi_agent_metrics import MultiAgentMetrics


def demo_message_bus():
    """演示消息总线"""
    print("\n" + "=" * 80)
    print("【演示1：消息总线（Message Bus）】")
    print("=" * 80)

    bus = MessageBus()

    # 定义订阅者
    def query_handler(data):
        print(f"  📨 QueryAgent received: {data}")

    def grader_handler(data):
        print(f"  📨 GraderAgent received: {data}")

    # 订阅事件
    bus.subscribe("query_analyzed", query_handler)
    bus.subscribe("query_analyzed", grader_handler)

    # 发布事件
    print("\n1. 发布事件...")
    count = bus.publish("query_analyzed", {"query": "What is RAG?", "intent": "question"})
    print(f"   ✅ 通知了 {count} 个订阅者")

    # 共享状态
    print("\n2. 共享状态...")
    bus.set_shared_state("current_query", "What is RAG?")
    bus.set_shared_state("confidence", 0.85)

    print(f"   Query: {bus.get_shared_state('current_query')}")
    print(f"   Confidence: {bus.get_shared_state('confidence')}")

    # 查看指标
    print("\n3. 消息总线指标...")
    metrics = bus.get_metrics()
    print(f"   总消息数: {metrics['total_messages']}")
    print(f"   总订阅者数: {metrics['total_subscribers']}")
    print(f"   事件列表: {metrics['events']}")

    bus.clear()


def demo_smart_router():
    """演示智能路由"""
    print("\n" + "=" * 80)
    print("【演示2：智能路由（Smart Router）】")
    print("=" * 80)

    router = SmartRouter()

    # 测试不同置信度的路由
    test_cases = [
        ("Simple question", 0.95),
        ("Complex query", 0.6),
        ("Ambiguous question", 0.3)
    ]

    for query, confidence in test_cases:
        strategy, decision = router.route(query, confidence)
        print(f"\n查询: {query}")
        print(f"置信度: {confidence}")
        print(f"策略: {strategy.value}")
        print(f"使用Agent: {', '.join(decision['agents'])}")

    # 降级策略
    print("\n降级策略演示:")
    fallback, agents = router.get_fallback_strategy(RouteStrategy.ENHANCED)
    print(f"  ENHANCED → {fallback.value}")
    fallback, agents = router.get_fallback_strategy(RouteStrategy.STANDARD)
    print(f"  STANDARD → {fallback.value}")
    fallback, agents = router.get_fallback_strategy(RouteStrategy.FAST)
    print(f"  FAST → {fallback.value}")

    # 路由指标
    print("\n路由指标:")
    metrics = router.get_routing_metrics()
    print(f"  总路由次数: {metrics['total_routes']}")
    print(f"  策略分布: {metrics['strategy_distribution']}")
    print(f"  平均置信度: {metrics['avg_confidence']:.2f}")

    router.clear()


def demo_retry_handler():
    """演示重试机制"""
    print("\n" + "=" * 80)
    print("【演示3：重试机制（Retry Handler）】")
    print("=" * 80)

    handler = RetryHandler(max_retries=3, backoff_factor=0.5)

    # 场景1：立即成功
    print("\n场景1：立即成功（无需重试）")
    def success_func():
        return "success"

    result = handler.execute_with_retry(success_func, "SuccessAgent")
    print(f"  结果: {result}")

    # 场景2：重试后成功
    print("\n场景2：重试2次后成功")
    call_count = [0]
    def retry_then_success():
        call_count[0] += 1
        if call_count[0] < 3:
            raise Exception("temporary error")
        return "success after retries"

    result = handler.execute_with_retry(retry_then_success, "RetryAgent")
    print(f"  结果: {result}")

    # 场景3：达到最大重试次数
    print("\n场景3：达到最大重试次数（失败）")
    def always_fail():
        raise Exception("persistent error")

    try:
        handler.execute_with_retry(always_fail, "FailAgent")
    except Exception as e:
        print(f"  ❌ 最终失败: {e}")

    # 重试指标
    print("\n重试指标:")
    metrics = handler.get_metrics()
    print(f"  总失败次数: {metrics['total_failures']}")
    print(f"  各Agent失败次数: {metrics['failures_by_agent']}")

    handler.clear()


def demo_multi_agent_metrics():
    """演示Multi-Agent评测系统"""
    print("\n" + "=" * 80)
    print("【演示4：Multi-Agent评测系统】")
    print("=" * 80)

    metrics = MultiAgentMetrics()

    # 模拟完整的Multi-Agent执行流程
    print("\n正在执行Multi-Agent流程...")

    # 1. QueryAgent
    start_time = time.time()
    time.sleep(0.1)  # 模拟执行
    end_time = time.time()
    metrics.collaboration.log_agent_execution("QueryAgent", start_time, end_time, True)
    metrics.communication.log_token_usage("QueryAgent", "gpt-3.5-turbo", 200, 100)
    metrics.communication.log_api_call("QueryAgent", "OpenAI", end_time - start_time, True)
    metrics.decision.log_agent_decision("QueryAgent", "analyze_query", True, 0.95)

    # 2. GraderAgent（并行开始）
    start_time = time.time() - 0.05  # 稍微提前开始（模拟并行）
    time.sleep(0.15)
    end_time = time.time()
    metrics.collaboration.log_agent_execution("GraderAgent", start_time, end_time, True)
    metrics.communication.log_token_usage("GraderAgent", "gpt-3.5-turbo", 300, 150)
    metrics.communication.log_api_call("GraderAgent", "OpenAI", end_time - start_time, True)
    metrics.decision.log_agent_decision("GraderAgent", "grade_docs", True, 0.88)

    # 3. GeneratorAgent
    start_time = time.time()
    time.sleep(0.2)
    end_time = time.time()
    metrics.collaboration.log_agent_execution("GeneratorAgent", start_time, end_time, True)
    metrics.communication.log_token_usage("GeneratorAgent", "gpt-3.5-turbo", 500, 300)
    metrics.communication.log_api_call("GeneratorAgent", "OpenAI", end_time - start_time, True)
    metrics.decision.log_agent_decision("GeneratorAgent", "generate_answer", True, 0.90)

    # 4. FactCheckerAgent（并行验证）
    start_time = time.time() - 0.1
    time.sleep(0.12)
    end_time = time.time()
    metrics.collaboration.log_agent_execution("FactCheckerAgent", start_time, end_time, True)
    metrics.communication.log_token_usage("FactCheckerAgent", "gpt-3.5-turbo", 150, 50)
    metrics.communication.log_api_call("FactCheckerAgent", "OpenAI", end_time - start_time, True)
    metrics.decision.log_agent_decision("FactCheckerAgent", "check_facts", True, 0.93)

    # 5. ConflictDetectorAgent（并行验证）
    start_time = time.time() - 0.1
    time.sleep(0.1)
    end_time = time.time()
    metrics.collaboration.log_agent_execution("ConflictDetectorAgent", start_time, end_time, True)
    metrics.communication.log_token_usage("ConflictDetectorAgent", "gpt-3.5-turbo", 100, 50)
    metrics.communication.log_api_call("ConflictDetectorAgent", "OpenAI", end_time - start_time, True)
    metrics.decision.log_agent_decision("ConflictDetectorAgent", "detect_conflicts", True, 0.85)

    # 路由决策
    metrics.decision.log_routing_decision("What is RAG?", "standard", "standard", 0.85)

    # 生成完整报告
    print("\n" + metrics.generate_report())

    # 获取结构化指标
    all_metrics = metrics.get_all_metrics()
    print("\n【结构化指标】")
    print(f"协作效率 - 并行度: {all_metrics['collaboration']['parallelism']}x")
    print(f"通信成本 - 总Token: {all_metrics['communication']['total_tokens']:,}")
    print(f"通信成本 - 成本: ${all_metrics['communication']['cost_usd']}")
    print(f"决策准确率 - 路由准确率: {all_metrics['decision']['routing_accuracy'] * 100:.0f}%")

    metrics.clear()


def main():
    """运行所有演示"""
    print("\n" + "🚀" * 40)
    print("Multi-Agent 优化功能完整演示")
    print("🚀" * 40)

    demo_message_bus()
    demo_smart_router()
    demo_retry_handler()
    demo_multi_agent_metrics()

    print("\n" + "=" * 80)
    print("✅ 所有演示完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
