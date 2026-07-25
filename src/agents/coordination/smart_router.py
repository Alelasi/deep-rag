#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
智能路由器（Smart Router）- 动态决策路由

功能：
1. 根据查询置信度选择路由策略
2. 失败降级（Fallback）
3. 路由决策追踪
"""

from typing import Dict, List, Tuple
from enum import Enum


class RouteStrategy(Enum):
    """路由策略枚举"""
    FAST = "fast"           # 快速流程（跳过部分验证）
    STANDARD = "standard"   # 标准流程（完整5个Agent）
    ENHANCED = "enhanced"   # 增强流程（增加验证步骤）
    FALLBACK = "fallback"   # 降级流程（最简单）


class SmartRouter:
    """智能路由器"""

    def __init__(self):
        # 路由决策历史
        self._routing_history: List[Dict] = []

        # 置信度阈值
        self.high_confidence_threshold = 0.8
        self.low_confidence_threshold = 0.5

    def route(self, query: str, confidence: float, context: Dict = None) -> Tuple[RouteStrategy, Dict]:
        """
        根据查询和置信度决定路由策略

        Args:
            query: 用户查询
            confidence: 置信度（0-1）
            context: 额外上下文（可选）

        Returns:
            (策略, 决策详情)
        """
        context = context or {}

        # 决策逻辑
        if confidence >= self.high_confidence_threshold:
            strategy = RouteStrategy.FAST
            reason = f"High confidence ({confidence:.2f}) - use fast route"
            agents = ["QueryAgent", "GeneratorAgent", "FactCheckerAgent"]

        elif confidence >= self.low_confidence_threshold:
            strategy = RouteStrategy.STANDARD
            reason = f"Medium confidence ({confidence:.2f}) - use standard route"
            agents = ["QueryAgent", "GraderAgent", "GeneratorAgent", "FactCheckerAgent", "ConflictDetectorAgent"]

        else:
            strategy = RouteStrategy.ENHANCED
            reason = f"Low confidence ({confidence:.2f}) - use enhanced route with extra validation"
            agents = ["QueryAgent", "GraderAgent", "GeneratorAgent", "FactCheckerAgent", "ConflictDetectorAgent", "RerankerAgent"]

        # 记录决策
        decision = {
            "query": query[:50] + "..." if len(query) > 50 else query,
            "confidence": confidence,
            "strategy": strategy.value,
            "reason": reason,
            "agents": agents,
            "context": context
        }

        self._routing_history.append(decision)

        return strategy, decision

    def should_fallback(self, agent_name: str, error: Exception, retry_count: int) -> bool:
        """
        判断是否需要降级

        Args:
            agent_name: Agent名称
            error: 错误异常
            retry_count: 已重试次数

        Returns:
            True 表示需要降级
        """
        # 降级条件：
        # 1. 重试次数 >= 3
        # 2. 特定错误类型（API超时、模型不可用等）
        if retry_count >= 3:
            return True

        error_str = str(error).lower()
        critical_errors = ["timeout", "unavailable", "rate limit", "quota exceeded"]

        return any(err in error_str for err in critical_errors)

    def get_fallback_strategy(self, original_strategy: RouteStrategy) -> Tuple[RouteStrategy, List[str]]:
        """
        获取降级策略

        Args:
            original_strategy: 原始策略

        Returns:
            (降级策略, Agent列表)
        """
        if original_strategy == RouteStrategy.ENHANCED:
            # 增强 → 标准
            return RouteStrategy.STANDARD, ["QueryAgent", "GraderAgent", "GeneratorAgent", "FactCheckerAgent", "ConflictDetectorAgent"]

        elif original_strategy == RouteStrategy.STANDARD:
            # 标准 → 快速
            return RouteStrategy.FAST, ["QueryAgent", "GeneratorAgent", "FactCheckerAgent"]

        elif original_strategy == RouteStrategy.FAST:
            # 快速 → 降级（最简单）
            return RouteStrategy.FALLBACK, ["GeneratorAgent"]

        else:
            # 已经是降级，无法再降
            return RouteStrategy.FALLBACK, ["GeneratorAgent"]

    def get_routing_metrics(self) -> Dict:
        """
        获取路由指标

        Returns:
            {
                "total_routes": 总路由次数,
                "strategy_distribution": {策略: 次数},
                "avg_confidence": 平均置信度
            }
        """
        if not self._routing_history:
            return {
                "total_routes": 0,
                "strategy_distribution": {},
                "avg_confidence": 0.0
            }

        strategy_distribution = {}
        total_confidence = 0.0

        for decision in self._routing_history:
            strategy = decision["strategy"]
            strategy_distribution[strategy] = strategy_distribution.get(strategy, 0) + 1
            total_confidence += decision["confidence"]

        return {
            "total_routes": len(self._routing_history),
            "strategy_distribution": strategy_distribution,
            "avg_confidence": total_confidence / len(self._routing_history)
        }

    def get_routing_history(self) -> List[Dict]:
        """获取路由历史"""
        return self._routing_history.copy()

    def clear(self):
        """清空历史（用于测试）"""
        self._routing_history.clear()
