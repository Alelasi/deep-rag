#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi-Agent 评测系统 - 协作效率、通信成本、决策准确率

功能：
1. 协作效率指标（并行度、耗时、消息传递）
2. 通信成本指标（Token消耗、API调用、成本）
3. 决策准确率指标（路由准确率、Agent准确率）
"""

from typing import Dict, List, Any
from datetime import datetime


class CollaborationMetrics:
    """协作效率指标"""

    def __init__(self):
        self.execution_logs: List[Dict] = []

    def log_agent_execution(
        self,
        agent_name: str,
        start_time: float,
        end_time: float,
        success: bool,
        metadata: Dict = None
    ):
        """
        记录 Agent 执行日志

        Args:
            agent_name: Agent名称
            start_time: 开始时间（timestamp）
            end_time: 结束时间（timestamp）
            success: 是否成功
            metadata: 额外元数据
        """
        self.execution_logs.append({
            "agent": agent_name,
            "start_time": start_time,
            "end_time": end_time,
            "duration": end_time - start_time,
            "success": success,
            "metadata": metadata or {}
        })

    def calculate_metrics(self) -> Dict:
        """
        计算协作效率指标

        Returns:
            {
                "total_agents_called": Agent调用总数,
                "successful_agents": 成功的Agent数,
                "total_time": 总耗时（秒）,
                "serial_time": 串行耗时（秒）,
                "parallelism": 并行度,
                "avg_agent_time": 平均Agent耗时（秒）,
                "agent_times": {agent: 耗时},
            }
        """
        if not self.execution_logs:
            return {
                "total_agents_called": 0,
                "successful_agents": 0,
                "total_time": 0.0,
                "serial_time": 0.0,
                "parallelism": 0.0,
                "avg_agent_time": 0.0,
                "agent_times": {},
            }

        # 总调用数
        total_agents = len(self.execution_logs)
        successful_agents = sum(1 for log in self.execution_logs if log["success"])

        # 串行耗时（所有Agent耗时之和）
        serial_time = sum(log["duration"] for log in self.execution_logs)

        # 总耗时（最早开始到最晚结束）
        start_times = [log["start_time"] for log in self.execution_logs]
        end_times = [log["end_time"] for log in self.execution_logs]
        total_time = max(end_times) - min(start_times)

        # 并行度（串行耗时 / 总耗时）
        parallelism = serial_time / total_time if total_time > 0 else 1.0

        # 平均耗时
        avg_time = serial_time / total_agents if total_agents > 0 else 0.0

        # 各Agent耗时
        agent_times = {}
        for log in self.execution_logs:
            agent = log["agent"]
            if agent not in agent_times:
                agent_times[agent] = 0.0
            agent_times[agent] += log["duration"]

        return {
            "total_agents_called": total_agents,
            "successful_agents": successful_agents,
            "total_time": round(total_time, 2),
            "serial_time": round(serial_time, 2),
            "parallelism": round(parallelism, 2),
            "avg_agent_time": round(avg_time, 2),
            "agent_times": {k: round(v, 2) for k, v in agent_times.items()},
        }


class CommunicationMetrics:
    """通信成本指标"""

    def __init__(self):
        self.token_logs: List[Dict] = []
        self.api_call_logs: List[Dict] = []

        # 价格配置（每1K tokens的价格，美元）
        self.pricing = {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-3.5-turbo": {"input": 0.001, "output": 0.002},
            "claude-3": {"input": 0.015, "output": 0.075},
        }

    def log_token_usage(
        self,
        agent_name: str,
        model: str,
        input_tokens: int,
        output_tokens: int
    ):
        """
        记录 Token 使用

        Args:
            agent_name: Agent名称
            model: 模型名称
            input_tokens: 输入Token数
            output_tokens: 输出Token数
        """
        self.token_logs.append({
            "agent": agent_name,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "timestamp": datetime.now().isoformat()
        })

    def log_api_call(
        self,
        agent_name: str,
        api_name: str,
        response_time: float,
        success: bool
    ):
        """
        记录 API 调用

        Args:
            agent_name: Agent名称
            api_name: API名称
            response_time: 响应时间（秒）
            success: 是否成功
        """
        self.api_call_logs.append({
            "agent": agent_name,
            "api": api_name,
            "response_time": response_time,
            "success": success,
            "timestamp": datetime.now().isoformat()
        })

    def calculate_metrics(self) -> Dict:
        """
        计算通信成本指标

        Returns:
            {
                "total_tokens": 总Token数,
                "input_tokens": 输入Token数,
                "output_tokens": 输出Token数,
                "tokens_by_agent": {agent: tokens},
                "api_calls": API调用次数,
                "successful_api_calls": 成功的API调用数,
                "avg_response_time": 平均响应时间（秒）,
                "cost_usd": 总成本（美元）,
            }
        """
        if not self.token_logs:
            return {
                "total_tokens": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "tokens_by_agent": {},
                "api_calls": 0,
                "successful_api_calls": 0,
                "avg_response_time": 0.0,
                "cost_usd": 0.0,
            }

        # Token 统计
        total_input = sum(log["input_tokens"] for log in self.token_logs)
        total_output = sum(log["output_tokens"] for log in self.token_logs)
        total_tokens = total_input + total_output

        # 各Agent Token消耗
        tokens_by_agent = {}
        for log in self.token_logs:
            agent = log["agent"]
            if agent not in tokens_by_agent:
                tokens_by_agent[agent] = 0
            tokens_by_agent[agent] += log["input_tokens"] + log["output_tokens"]

        # API 统计
        api_calls = len(self.api_call_logs)
        successful_calls = sum(1 for log in self.api_call_logs if log["success"])

        # 平均响应时间
        if self.api_call_logs:
            avg_response_time = sum(log["response_time"] for log in self.api_call_logs) / len(self.api_call_logs)
        else:
            avg_response_time = 0.0

        # 成本计算（简化版，假设使用 gpt-3.5-turbo）
        model_pricing = self.pricing.get("gpt-3.5-turbo", {"input": 0.001, "output": 0.002})
        cost = (total_input / 1000 * model_pricing["input"]) + (total_output / 1000 * model_pricing["output"])

        return {
            "total_tokens": total_tokens,
            "input_tokens": total_input,
            "output_tokens": total_output,
            "tokens_by_agent": tokens_by_agent,
            "api_calls": api_calls,
            "successful_api_calls": successful_calls,
            "avg_response_time": round(avg_response_time, 2),
            "cost_usd": round(cost, 4),
        }


class DecisionMetrics:
    """决策准确率指标"""

    def __init__(self):
        self.routing_logs: List[Dict] = []
        self.agent_decision_logs: List[Dict] = []

    def log_routing_decision(
        self,
        query: str,
        predicted_strategy: str,
        actual_strategy: str = None,
        confidence: float = None
    ):
        """
        记录路由决策

        Args:
            query: 查询
            predicted_strategy: 预测的策略
            actual_strategy: 实际应该使用的策略（可选，用于计算准确率）
            confidence: 置信度
        """
        self.routing_logs.append({
            "query": query[:50] + "..." if len(query) > 50 else query,
            "predicted": predicted_strategy,
            "actual": actual_strategy,
            "confidence": confidence,
            "correct": predicted_strategy == actual_strategy if actual_strategy else None
        })

    def log_agent_decision(
        self,
        agent_name: str,
        decision: str,
        correct: bool,
        confidence: float = None
    ):
        """
        记录 Agent 决策

        Args:
            agent_name: Agent名称
            decision: 决策内容
            correct: 是否正确
            confidence: 置信度
        """
        self.agent_decision_logs.append({
            "agent": agent_name,
            "decision": decision,
            "correct": correct,
            "confidence": confidence
        })

    def calculate_metrics(self) -> Dict:
        """
        计算决策准确率指标

        Returns:
            {
                "routing_accuracy": 路由准确率,
                "routing_decisions": 路由决策总数,
                "agent_decision_accuracy": {agent: 准确率},
                "agent_decisions": {agent: 决策总数},
                "avg_confidence": 平均置信度,
            }
        """
        # 路由准确率
        routing_decisions = len(self.routing_logs)
        correct_routings = sum(1 for log in self.routing_logs if log.get("correct") is True)
        routing_accuracy = correct_routings / routing_decisions if routing_decisions > 0 else 0.0

        # Agent 决策准确率
        agent_accuracy = {}
        agent_decisions = {}

        for log in self.agent_decision_logs:
            agent = log["agent"]
            if agent not in agent_accuracy:
                agent_accuracy[agent] = []
                agent_decisions[agent] = 0

            agent_accuracy[agent].append(1 if log["correct"] else 0)
            agent_decisions[agent] += 1

        # 计算各Agent平均准确率
        agent_accuracy = {
            agent: sum(scores) / len(scores) if scores else 0.0
            for agent, scores in agent_accuracy.items()
        }

        # 平均置信度
        all_confidences = [
            log["confidence"]
            for log in self.routing_logs + self.agent_decision_logs
            if log.get("confidence") is not None
        ]
        avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0

        return {
            "routing_accuracy": round(routing_accuracy, 2),
            "routing_decisions": routing_decisions,
            "agent_decision_accuracy": {k: round(v, 2) for k, v in agent_accuracy.items()},
            "agent_decisions": agent_decisions,
            "avg_confidence": round(avg_confidence, 2),
        }


class MultiAgentMetrics:
    """Multi-Agent 综合评测系统"""

    def __init__(self):
        self.collaboration = CollaborationMetrics()
        self.communication = CommunicationMetrics()
        self.decision = DecisionMetrics()

    def generate_report(self) -> str:
        """
        生成完整的评测报告

        Returns:
            格式化的报告字符串
        """
        collab_metrics = self.collaboration.calculate_metrics()
        comm_metrics = self.communication.calculate_metrics()
        decision_metrics = self.decision.calculate_metrics()

        report = []
        report.append("=" * 80)
        report.append("Multi-Agent 评测报告")
        report.append("=" * 80)
        report.append("")

        # 协作效率
        report.append("【协作效率】")
        report.append(f"  总调用Agent数: {collab_metrics['total_agents_called']}")
        report.append(f"  成功Agent数: {collab_metrics['successful_agents']}")
        report.append(f"  总耗时: {collab_metrics['total_time']}秒")
        report.append(f"  串行耗时: {collab_metrics['serial_time']}秒")
        report.append(f"  并行度: {collab_metrics['parallelism']}x ⭐")
        report.append(f"  平均Agent耗时: {collab_metrics['avg_agent_time']}秒")
        report.append("")

        # 通信成本
        report.append("【通信成本】")
        report.append(f"  总Token消耗: {comm_metrics['total_tokens']:,}")
        report.append(f"  输入Token: {comm_metrics['input_tokens']:,}")
        report.append(f"  输出Token: {comm_metrics['output_tokens']:,}")
        report.append(f"  API调用次数: {comm_metrics['api_calls']}")
        report.append(f"  成功调用: {comm_metrics['successful_api_calls']}")
        report.append(f"  平均响应时间: {comm_metrics['avg_response_time']}秒")
        report.append(f"  成本: ${comm_metrics['cost_usd']}")
        report.append("")

        # 决策准确率
        report.append("【决策准确率】")
        report.append(f"  路由准确率: {decision_metrics['routing_accuracy'] * 100:.0f}% ✅")
        report.append(f"  路由决策数: {decision_metrics['routing_decisions']}")
        report.append(f"  平均置信度: {decision_metrics['avg_confidence']}")
        report.append("")

        # 各Agent表现
        report.append("【各Agent表现】")
        for agent, time_spent in collab_metrics['agent_times'].items():
            tokens = comm_metrics['tokens_by_agent'].get(agent, 0)
            accuracy = decision_metrics['agent_decision_accuracy'].get(agent, 0.0)
            report.append(f"  {agent}: {accuracy * 100:.0f}% (耗时 {time_spent}秒, Token {tokens})")

        report.append("")
        report.append("=" * 80)

        return "\n".join(report)

    def get_all_metrics(self) -> Dict:
        """获取所有指标（结构化）"""
        return {
            "collaboration": self.collaboration.calculate_metrics(),
            "communication": self.communication.calculate_metrics(),
            "decision": self.decision.calculate_metrics(),
        }

    def clear(self):
        """清空所有数据（用于测试）"""
        self.collaboration.execution_logs.clear()
        self.communication.token_logs.clear()
        self.communication.api_call_logs.clear()
        self.decision.routing_logs.clear()
        self.decision.agent_decision_logs.clear()
