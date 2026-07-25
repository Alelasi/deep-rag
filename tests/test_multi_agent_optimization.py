#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi-Agent 优化功能测试

测试覆盖：
1. MessageBus（消息总线）
2. SmartRouter（智能路由）
3. RetryHandler（重试机制）
4. MultiAgentMetrics（评测系统）
"""

import pytest
import time
from src.agents.coordination import MessageBus, SmartRouter, RouteStrategy, RetryHandler, CircuitState
from src.evaluation.multi_agent_metrics import (
    CollaborationMetrics,
    CommunicationMetrics,
    DecisionMetrics,
    MultiAgentMetrics
)


# ===== MessageBus 测试 =====

def test_message_bus_publish_subscribe():
    """测试发布/订阅机制"""
    bus = MessageBus()
    received_data = []

    def handler(data):
        received_data.append(data)

    # 订阅事件
    bus.subscribe("test_event", handler)

    # 发布事件
    count = bus.publish("test_event", {"message": "hello"})

    assert count == 1
    assert len(received_data) == 1
    assert received_data[0]["message"] == "hello"

    bus.clear()


def test_message_bus_multiple_subscribers():
    """测试多个订阅者"""
    bus = MessageBus()
    received_count = [0, 0]

    def handler1(data):
        received_count[0] += 1

    def handler2(data):
        received_count[1] += 1

    bus.subscribe("test_event", handler1)
    bus.subscribe("test_event", handler2)

    count = bus.publish("test_event", {"test": True})

    assert count == 2
    assert received_count == [1, 1]

    bus.clear()


def test_message_bus_shared_state():
    """测试共享状态管理"""
    bus = MessageBus()

    # 设置状态
    bus.set_shared_state("key1", "value1")
    bus.set_shared_state("key2", 42)

    # 获取状态
    assert bus.get_shared_state("key1") == "value1"
    assert bus.get_shared_state("key2") == 42
    assert bus.get_shared_state("key3", "default") == "default"

    bus.clear()


def test_message_bus_message_history():
    """测试消息历史"""
    bus = MessageBus()

    bus.publish("event1", {"data": 1})
    bus.publish("event2", {"data": 2})
    bus.publish("event1", {"data": 3})

    # 获取所有历史
    all_history = bus.get_message_history()
    assert len(all_history) == 3

    # 获取特定事件历史
    event1_history = bus.get_message_history("event1")
    assert len(event1_history) == 2

    bus.clear()


def test_message_bus_metrics():
    """测试消息总线指标"""
    bus = MessageBus()

    bus.subscribe("event1", lambda d: None)
    bus.subscribe("event2", lambda d: None)
    bus.subscribe("event2", lambda d: None)

    bus.publish("event1", {})
    bus.publish("event2", {})

    metrics = bus.get_metrics()

    assert metrics["total_messages"] == 2
    assert metrics["total_subscribers"] == 3
    assert "event1" in metrics["events"]
    assert "event2" in metrics["events"]

    bus.clear()


# ===== SmartRouter 测试 =====

def test_smart_router_high_confidence():
    """测试高置信度路由（快速流程）"""
    router = SmartRouter()

    strategy, decision = router.route("simple query", confidence=0.9)

    assert strategy == RouteStrategy.FAST
    assert decision["confidence"] == 0.9
    assert "QueryAgent" in decision["agents"]
    assert "GeneratorAgent" in decision["agents"]

    router.clear()


def test_smart_router_medium_confidence():
    """测试中等置信度路由（标准流程）"""
    router = SmartRouter()

    strategy, decision = router.route("medium query", confidence=0.65)

    assert strategy == RouteStrategy.STANDARD
    assert decision["confidence"] == 0.65
    assert len(decision["agents"]) == 5  # 5个Agent

    router.clear()


def test_smart_router_low_confidence():
    """测试低置信度路由（增强流程）"""
    router = SmartRouter()

    strategy, decision = router.route("complex query", confidence=0.3)

    assert strategy == RouteStrategy.ENHANCED
    assert decision["confidence"] == 0.3
    assert len(decision["agents"]) > 5  # 增强流程有更多Agent

    router.clear()


def test_smart_router_fallback_decision():
    """测试降级决策"""
    router = SmartRouter()

    # 测试不同策略的降级
    fallback_strategy, agents = router.get_fallback_strategy(RouteStrategy.ENHANCED)
    assert fallback_strategy == RouteStrategy.STANDARD

    fallback_strategy, agents = router.get_fallback_strategy(RouteStrategy.STANDARD)
    assert fallback_strategy == RouteStrategy.FAST

    fallback_strategy, agents = router.get_fallback_strategy(RouteStrategy.FAST)
    assert fallback_strategy == RouteStrategy.FALLBACK

    router.clear()


def test_smart_router_should_fallback():
    """测试是否需要降级"""
    router = SmartRouter()

    # 重试次数超过阈值
    assert router.should_fallback("TestAgent", Exception("test"), retry_count=3) is True

    # 关键错误
    assert router.should_fallback("TestAgent", Exception("timeout"), retry_count=1) is True
    assert router.should_fallback("TestAgent", Exception("rate limit"), retry_count=1) is True

    # 正常错误
    assert router.should_fallback("TestAgent", Exception("normal error"), retry_count=1) is False

    router.clear()


def test_smart_router_metrics():
    """测试路由指标"""
    router = SmartRouter()

    router.route("query1", 0.9)
    router.route("query2", 0.6)
    router.route("query3", 0.3)

    metrics = router.get_routing_metrics()

    assert metrics["total_routes"] == 3
    assert "fast" in metrics["strategy_distribution"]
    assert "standard" in metrics["strategy_distribution"]
    assert "enhanced" in metrics["strategy_distribution"]
    assert 0 < metrics["avg_confidence"] < 1

    router.clear()


# ===== RetryHandler 测试 =====

def test_retry_handler_success_no_retry():
    """测试成功执行（无需重试）"""
    handler = RetryHandler(max_retries=3)

    call_count = [0]

    def success_func():
        call_count[0] += 1
        return "success"

    result = handler.execute_with_retry(success_func, "TestAgent")

    assert result == "success"
    assert call_count[0] == 1  # 只调用1次

    handler.clear()


def test_retry_handler_success_after_retries():
    """测试重试后成功"""
    handler = RetryHandler(max_retries=3, backoff_factor=0.1)  # 快速退避用于测试

    call_count = [0]

    def retry_then_success():
        call_count[0] += 1
        if call_count[0] < 3:
            raise Exception("temporary error")
        return "success"

    result = handler.execute_with_retry(retry_then_success, "TestAgent")

    assert result == "success"
    assert call_count[0] == 3  # 调用3次

    handler.clear()


def test_retry_handler_failure_after_max_retries():
    """测试达到最大重试次数后失败"""
    handler = RetryHandler(max_retries=2, backoff_factor=0.1)

    def always_fail():
        raise Exception("persistent error")

    with pytest.raises(Exception, match="persistent error"):
        handler.execute_with_retry(always_fail, "TestAgent")

    handler.clear()


def test_retry_handler_circuit_breaker():
    """测试熔断器"""
    handler = RetryHandler(
        max_retries=0,  # 不重试
        circuit_breaker_threshold=3
    )

    def failing_func():
        raise Exception("error")

    # 前3次失败
    for _ in range(3):
        try:
            handler.execute_with_retry(failing_func, "TestAgent")
        except:
            pass

    # 熔断器应该打开
    assert handler.is_circuit_open("TestAgent") is True

    # 再次调用应该直接拒绝
    with pytest.raises(Exception, match="Circuit breaker is OPEN"):
        handler.execute_with_retry(failing_func, "TestAgent")

    handler.clear()


def test_retry_handler_metrics():
    """测试重试指标"""
    handler = RetryHandler(max_retries=0)

    def failing_func():
        raise Exception("error")

    # 记录2次失败
    for _ in range(2):
        try:
            handler.execute_with_retry(failing_func, "TestAgent1")
        except:
            pass

    try:
        handler.execute_with_retry(failing_func, "TestAgent2")
    except:
        pass

    metrics = handler.get_metrics()

    assert metrics["total_failures"] == 3
    assert metrics["failures_by_agent"]["TestAgent1"] == 2
    assert metrics["failures_by_agent"]["TestAgent2"] == 1

    handler.clear()


# ===== CollaborationMetrics 测试 =====

def test_collaboration_metrics_basic():
    """测试协作效率指标（基础）"""
    metrics = CollaborationMetrics()

    # 模拟3个Agent执行
    start = time.time()
    metrics.log_agent_execution("Agent1", start, start + 0.5, True)
    metrics.log_agent_execution("Agent2", start + 0.1, start + 0.6, True)
    metrics.log_agent_execution("Agent3", start + 0.2, start + 0.8, True)

    result = metrics.calculate_metrics()

    assert result["total_agents_called"] == 3
    assert result["successful_agents"] == 3
    assert 1.4 <= result["serial_time"] <= 1.7  # 0.5 + 0.5 + 0.5 (允许浮点误差)
    assert 0.7 <= result["total_time"] <= 0.9  # 从0到0.8 (允许浮点误差)
    assert result["parallelism"] > 1.0  # 串行/总耗时 > 1（有并行）


def test_collaboration_metrics_parallelism():
    """测试并行度计算"""
    metrics = CollaborationMetrics()

    # 完全串行（并行度 = 1.0）
    start = time.time()
    metrics.log_agent_execution("Agent1", start, start + 1.0, True)
    metrics.log_agent_execution("Agent2", start + 1.0, start + 2.0, True)

    result = metrics.calculate_metrics()

    assert result["parallelism"] == 1.0  # 完全串行


# ===== CommunicationMetrics 测试 =====

def test_communication_metrics_token_usage():
    """测试Token使用统计"""
    metrics = CommunicationMetrics()

    metrics.log_token_usage("Agent1", "gpt-3.5-turbo", 100, 50)
    metrics.log_token_usage("Agent2", "gpt-3.5-turbo", 200, 100)

    result = metrics.calculate_metrics()

    assert result["total_tokens"] == 450  # 100+50+200+100
    assert result["input_tokens"] == 300
    assert result["output_tokens"] == 150
    assert result["tokens_by_agent"]["Agent1"] == 150
    assert result["tokens_by_agent"]["Agent2"] == 300


def test_communication_metrics_cost_calculation():
    """测试成本计算"""
    metrics = CommunicationMetrics()

    # 1000 input tokens + 1000 output tokens
    metrics.log_token_usage("Agent1", "gpt-3.5-turbo", 1000, 1000)

    result = metrics.calculate_metrics()

    # gpt-3.5-turbo: input $0.001/1K, output $0.002/1K
    expected_cost = (1000 / 1000 * 0.001) + (1000 / 1000 * 0.002)  # $0.003
    assert result["cost_usd"] == expected_cost


def test_communication_metrics_api_calls():
    """测试API调用统计"""
    metrics = CommunicationMetrics()

    metrics.log_api_call("Agent1", "OpenAI", 0.5, True)
    metrics.log_api_call("Agent2", "OpenAI", 0.3, True)
    metrics.log_api_call("Agent3", "OpenAI", 1.0, False)

    result = metrics.calculate_metrics()

    assert result["api_calls"] == 3
    assert result["successful_api_calls"] == 2
    assert result["avg_response_time"] == 0.6  # (0.5+0.3+1.0)/3


# ===== DecisionMetrics 测试 =====

def test_decision_metrics_routing_accuracy():
    """测试路由准确率"""
    metrics = DecisionMetrics()

    metrics.log_routing_decision("query1", "fast", "fast", 0.9)
    metrics.log_routing_decision("query2", "standard", "standard", 0.6)
    metrics.log_routing_decision("query3", "fast", "enhanced", 0.3)  # 错误

    result = metrics.calculate_metrics()

    assert result["routing_accuracy"] == 0.67  # 2/3
    assert result["routing_decisions"] == 3


def test_decision_metrics_agent_accuracy():
    """测试Agent决策准确率"""
    metrics = DecisionMetrics()

    metrics.log_agent_decision("Agent1", "decision1", True, 0.9)
    metrics.log_agent_decision("Agent1", "decision2", True, 0.8)
    metrics.log_agent_decision("Agent1", "decision3", False, 0.5)

    metrics.log_agent_decision("Agent2", "decision1", True, 0.95)

    result = metrics.calculate_metrics()

    assert result["agent_decision_accuracy"]["Agent1"] == 0.67  # 2/3
    assert result["agent_decision_accuracy"]["Agent2"] == 1.0   # 1/1
    assert result["agent_decisions"]["Agent1"] == 3
    assert result["agent_decisions"]["Agent2"] == 1


# ===== MultiAgentMetrics 测试 =====

def test_multi_agent_metrics_integration():
    """测试综合评测系统"""
    metrics = MultiAgentMetrics()

    # 记录协作数据
    start = time.time()
    metrics.collaboration.log_agent_execution("Agent1", start, start + 0.5, True)
    metrics.collaboration.log_agent_execution("Agent2", start + 0.1, start + 0.6, True)

    # 记录通信数据
    metrics.communication.log_token_usage("Agent1", "gpt-3.5-turbo", 100, 50)
    metrics.communication.log_api_call("Agent1", "OpenAI", 0.5, True)

    # 记录决策数据
    metrics.decision.log_routing_decision("query1", "fast", "fast", 0.9)
    metrics.decision.log_agent_decision("Agent1", "decision1", True, 0.9)

    # 获取所有指标
    all_metrics = metrics.get_all_metrics()

    assert "collaboration" in all_metrics
    assert "communication" in all_metrics
    assert "decision" in all_metrics

    # 生成报告
    report = metrics.generate_report()

    assert "Multi-Agent 评测报告" in report
    assert "协作效率" in report
    assert "通信成本" in report
    assert "决策准确率" in report


def test_multi_agent_metrics_report_format():
    """测试报告格式"""
    metrics = MultiAgentMetrics()

    # 添加一些模拟数据
    start = time.time()
    metrics.collaboration.log_agent_execution("Agent1", start, start + 1.0, True)
    metrics.communication.log_token_usage("Agent1", "gpt-3.5-turbo", 200, 100)
    metrics.decision.log_routing_decision("query1", "fast", "fast", 0.9)

    report = metrics.generate_report()

    # 检查报告包含关键信息
    assert "并行度" in report
    assert "Token" in report
    assert "准确率" in report
    assert "Agent1" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
