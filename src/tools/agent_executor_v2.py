"""
Agent执行器 v2.0 - 企业级增强版
集成5大核心能力：规划/记忆/日志/Prompt管理/重试
"""

from typing import Dict, List, Any, Optional
import httpx
import json
from .tool_registry import get_registry
from .modules import (
    TaskPlanner,
    ConversationMemory,
    StructuredLogger,
    PromptManager,
    RetryManager,
    RetryConfig
)


try:
    from src.logging_config import get_logger
except Exception:
    import logging

    def get_logger(n):  # type: ignore
        return logging.getLogger(n)

logger = get_logger(__name__)

class AgentExecutorV2:
    """
    Agent执行器 v2.0（企业级增强版）

    🆕 v2.0 新增能力：
    1. ✅ 任务规划（Planning）- 支持多步骤复杂查询
    2. ✅ 记忆管理（Memory）- 支持多轮对话上下文
    3. ✅ 结构化日志（Logging）- JSON格式性能追踪
    4. ✅ Prompt版本管理 - 热切换/A/B测试
    5. ✅ 智能重试（Retry）- 指数退避算法

    原有能力：
    - ReAct循环：思考 → 行动 → 观察 → 继续
    - Function Calling：自动调用工具
    - 白名单沙箱：只允许search_database
    - 提示词约束：强制安全规则
    """

    def __init__(
        self,
        llm_base_url: str = "http://localhost:11434/v1/chat/completions",
        model: str = "gemma-3-4b-it",
        max_iterations: int = 5,
        enable_planning: bool = True,
        enable_memory: bool = True,
        enable_logging: bool = True,
        log_file: Optional[str] = "logs/agent_execution.log",
        prompt_version: str = "v1"
    ):
        """
        初始化Agent执行器

        Args:
            llm_base_url: LLM API地址
            model: 模型名称
            max_iterations: 最大迭代次数
            enable_planning: 是否启用任务规划
            enable_memory: 是否启用记忆管理
            enable_logging: 是否启用结构化日志
            log_file: 日志文件路径
            prompt_version: Prompt版本（v1/v2/latest）
        """
        self.llm_base_url = llm_base_url
        self.model = model
        self.max_iterations = max_iterations
        self.registry = get_registry()

        # 🆕 新增模块
        self.planner = TaskPlanner(llm_base_url, model) if enable_planning else None
        self.memory = ConversationMemory() if enable_memory else None
        self.logger = StructuredLogger(log_file=log_file, console_output=False) if enable_logging else None
        self.prompt_manager = PromptManager()
        self.retry_manager = RetryManager(RetryConfig(max_retries=3, base_delay=1.0))
        self.prompt_version = prompt_version

        # 统计
        self.stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "total_planning_calls": 0,
            "complex_tasks": 0
        }

    def run(
        self,
        user_query: str,
        verbose: bool = True,
        use_memory: bool = True
    ) -> Dict[str, Any]:
        """
        执行Agent（增强版）

        Args:
            user_query: 用户问题
            verbose: 是否打印详细日志
            use_memory: 是否使用记忆上下文

        Returns:
            {
                "success": bool,
                "answer": str,
                "iterations": int,
                "history": List,
                "planning_result": Dict,  # 🆕 规划结果
                "memory_context": str  # 🆕 使用的上下文
            }
        """
        self.stats["total_queries"] += 1

        if self.logger:
            self.logger.start_timer("total")

        try:
            # 🆕 步骤1: 任务规划
            planning_result = None
            if self.planner:
                if verbose:
                    logger.info(f"\n{'='*60}")
                    logger.info("📋 步骤1: 任务规划")
                    logger.info(f"{'='*60}")

                self.stats["total_planning_calls"] += 1
                planning_result = self.planner.plan(user_query, verbose=verbose)

                if planning_result["is_complex"]:
                    self.stats["complex_tasks"] += 1

            # 🆕 步骤2: 获取记忆上下文
            memory_context = ""
            if self.memory and use_memory:
                memory_context = self.memory.get_context(k=3)
                if memory_context and verbose:
                    logger.info(f"\n{'='*60}")
                    logger.info("🧠 步骤2: 记忆上下文")
                    logger.info(f"{'='*60}")
                    logger.info(memory_context[:300] + "..." if len(memory_context) > 300 else memory_context)

            # 🆕 步骤3: 加载Prompt（版本化）
            system_prompt = self._build_system_prompt_v2(memory_context)

            # 步骤4: ReAct循环
            if verbose:
                logger.info(f"\n{'='*60}")
                logger.info("🤖 步骤3: ReAct执行")
                logger.info(f"{'='*60}")

            result = self._run_react_loop(
                user_query=user_query,
                system_prompt=system_prompt,
                verbose=verbose
            )

            # 🆕 步骤5: 保存到记忆
            if self.memory and result["success"]:
                self.memory.add(
                    query=user_query,
                    result=result["answer"],
                    metadata={"iterations": result["iterations"]}
                )

            # 🆕 步骤6: 记录日志
            if self.logger:
                total_latency = self.logger.stop_timer("total")
                self.logger.log_agent_complete(
                    success=result["success"],
                    iterations=result["iterations"],
                    total_latency_ms=total_latency,
                    answer=result.get("answer"),
                    error=result.get("error")
                )

            # 统计
            if result["success"]:
                self.stats["successful_queries"] += 1
            else:
                self.stats["failed_queries"] += 1

            # 🆕 添加新字段到返回结果
            result["planning_result"] = planning_result
            result["memory_context"] = memory_context

            return result

        except Exception as e:
            if self.logger:
                self.logger.log_error("agent_execution_error", str(e))

            self.stats["failed_queries"] += 1

            return {
                "success": False,
                "error": str(e),
                "iterations": 0,
                "history": []
            }

    def _build_system_prompt_v2(self, memory_context: str = "") -> str:
        """
        🆕 构建System Prompt v2（使用Prompt管理器）

        Args:
            memory_context: 记忆上下文

        Returns:
            完整的System Prompt
        """
        try:
            # 从Prompt管理器加载
            tools_desc = self.registry.get_tool_schemas()

            base_prompt = self.prompt_manager.load_prompt(
                prompt_name="system_prompt",
                version=self.prompt_version,
                variables={"tools_desc": json.dumps(tools_desc, ensure_ascii=False, indent=2)}
            )

            # 🆕 添加记忆上下文
            if memory_context:
                base_prompt += f"\n\n# 历史对话上下文\n{memory_context}\n"
                base_prompt += "注意：用户可能会引用之前的对话内容（如'刚才那个'），请结合上下文理解。\n"

            return base_prompt

        except FileNotFoundError:
            # 降级到硬编码版本
            return self._build_system_prompt_fallback(memory_context)

    def _build_system_prompt_fallback(self, memory_context: str = "") -> str:
        """回退版本（硬编码Prompt）"""
        tools_desc = self.registry.get_tool_schemas()

        prompt = f"""你是一个数据分析助手，使用ReAct模式工作。

# 核心规则（必须遵守）
1. 你只能执行**只读查询**，绝对禁止任何写操作
2. 你只有1个工具：search_database（只读SQL）
3. 任何INSERT/UPDATE/DELETE请求都必须拒绝

# ReAct工作流程
每一轮你必须输出JSON格式：

{{"thought": "我的思考...", "action": "search_database", "action_input": {{"sql_query": "SELECT ..."}}}}

或者完成时输出：

{{"thought": "任务完成", "final_answer": "最终答案"}}

# 可用工具
{json.dumps(tools_desc, ensure_ascii=False, indent=2)}

# 安全约束（强制）
- 只允许SELECT查询
- 禁止INSERT/UPDATE/DELETE/DROP等任何写操作
"""

        if memory_context:
            prompt += f"\n\n# 历史对话上下文\n{memory_context}\n"

        return prompt

    def _run_react_loop(
        self,
        user_query: str,
        system_prompt: str,
        verbose: bool
    ) -> Dict[str, Any]:
        """
        ReAct循环（核心逻辑）

        Returns:
            {
                "success": bool,
                "answer": str,
                "iterations": int,
                "history": List
            }
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]

        history = []

        for iteration in range(self.max_iterations):
            if verbose:
                logger.info(f"\n{'='*60}")
                logger.info(f"迭代 {iteration + 1}/{self.max_iterations}")
                logger.info(f"{'='*60}")

            if self.logger:
                self.logger.start_timer(f"iteration_{iteration}")

            try:
                # 🆕 带重试的LLM调用
                response = self.retry_manager.retry_with_backoff(
                    self._call_llm,
                    messages
                )

                agent_output = self._parse_agent_output(response)

                if verbose:
                    logger.info(f"思考: {agent_output.get('thought', 'N/A')}")

                # 记录历史
                step_record = {
                    "iteration": iteration + 1,
                    "thought": agent_output.get("thought"),
                    "action": agent_output.get("action"),
                    "observation": None
                }
                history.append(step_record)

                # 检查是否完成
                if "final_answer" in agent_output:
                    final_answer = agent_output["final_answer"]

                    # 后处理
                    if isinstance(final_answer, str) and final_answer.strip().startswith('{'):
                        try:
                            data = json.loads(final_answer)
                            if 'final_answer' in data:
                                final_answer = data['final_answer']
                        except:
                            pass

                    if verbose:
                        logger.info(f"✅ 完成: {final_answer}")

                    # 🆕 记录日志
                    if self.logger:
                        step_latency = self.logger.stop_timer(f"iteration_{iteration}")
                        self.logger.log_agent_step(
                            iteration=iteration + 1,
                            thought=agent_output.get("thought"),
                            action="final_answer",
                            observation=final_answer,
                            latency_ms=step_latency
                        )

                    return {
                        "success": True,
                        "answer": final_answer,
                        "iterations": iteration + 1,
                        "history": history
                    }

                # 执行工具
                action = agent_output.get("action")
                action_input = agent_output.get("action_input", {})

                if verbose:
                    logger.info(f"行动: {action}")
                    logger.info(f"参数: {json.dumps(action_input, ensure_ascii=False)}")

                # 白名单检查
                if action != "search_database":
                    observation = f"❌ 错误：工具 [{action}] 不在白名单中"
                else:
                    try:
                        # 🆕 带重试的工具调用
                        observation = self.retry_manager.retry_with_backoff(
                            self.registry.execute,
                            action,
                            **action_input
                        )
                        observation = json.dumps(observation, ensure_ascii=False)
                    except Exception as e:
                        observation = f"❌ 执行失败: {str(e)}"

                if verbose:
                    logger.info(f"观察: {observation[:200]}...")

                # 更新历史
                history[-1]["observation"] = observation

                # 🆕 记录日志
                if self.logger:
                    step_latency = self.logger.stop_timer(f"iteration_{iteration}")
                    self.logger.log_agent_step(
                        iteration=iteration + 1,
                        thought=agent_output.get("thought"),
                        action=action,
                        observation=observation,
                        latency_ms=step_latency
                    )

                # 添加观察到消息
                messages.append({
                    "role": "assistant",
                    "content": json.dumps(agent_output, ensure_ascii=False)
                })
                messages.append({
                    "role": "user",
                    "content": f"观察结果：{observation}"
                })

            except Exception as e:
                if verbose:
                    logger.error(f"❌ 迭代失败: {str(e)}")

                if self.logger:
                    self.logger.log_error("iteration_error", str(e), {"iteration": iteration + 1})

                return {
                    "success": False,
                    "error": str(e),
                    "iterations": iteration + 1,
                    "history": history
                }

        # 超过最大迭代次数
        return {
            "success": False,
            "error": f"超过最大迭代次数 ({self.max_iterations})",
            "iterations": self.max_iterations,
            "history": history
        }

    def _call_llm(self, messages: List[Dict]) -> str:
        """调用LLM"""
        client = httpx.Client(timeout=30.0)
        response = client.post(
            self.llm_base_url,
            json={
                "model": self.model,
                "messages": messages,
                "max_tokens": 200,
                "temperature": 0.1
            }
        )

        if response.status_code != 200:
            raise Exception(f"LLM调用失败: HTTP {response.status_code}")

        result = response.json()
        message = result["choices"][0]["message"]
        content = message.get("content", "") or message.get("reasoning_content", "")

        if not content:
            raise Exception("LLM返回空内容")

        return content

    def _parse_agent_output(self, content: str) -> Dict:
        """解析Agent输出（JSON格式）"""
        start_idx = content.find('{')
        if start_idx == -1:
            return {
                "thought": "未找到JSON格式",
                "final_answer": content
            }

        # 括号配对
        brace_count = 0
        in_string = False
        escape_next = False

        for i in range(start_idx, len(content)):
            char = content[i]

            if escape_next:
                escape_next = False
                continue
            if char == '\\':
                escape_next = True
                continue
            if char == '"':
                in_string = not in_string
                continue

            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_str = content[start_idx:i+1]
                        try:
                            return json.loads(json_str)
                        except Exception as e:
                            raise Exception(f"解析JSON失败: {str(e)}\nJSON: {json_str[:200]}")

        return {
            "thought": "JSON不完整",
            "final_answer": content
        }

    def get_stats(self) -> Dict[str, Any]:
        """🆕 获取统计信息"""
        stats = self.stats.copy()

        if self.retry_manager:
            stats["retry_stats"] = self.retry_manager.get_stats()

        if self.memory:
            stats["memory_size"] = len(self.memory.history)

        return stats

    def clear_memory(self):
        """🆕 清空记忆"""
        if self.memory:
            self.memory.clear()


# ============================================================================
# 使用示例
# ============================================================================

def demo_agent_v2():
    """演示Agent v2.0"""
    from .builtin_tools import register_builtin_tools
    from .modules import init_default_prompts

    # 初始化
    logger.info("初始化Agent v2.0...")
    register_builtin_tools()

    # 初始化默认Prompt
    prompt_manager = PromptManager()
    if not prompt_manager.list_versions("system_prompt"):
        init_default_prompts(prompt_manager)

    # 创建Agent v2.0
    agent = AgentExecutorV2(
        model="gemma-3-4b-it",
        max_iterations=5,
        enable_planning=True,
        enable_memory=True,
        enable_logging=True,
        log_file="logs/agent_v2_demo.log",
        prompt_version="v1"
    )

    # 测试案例（多轮对话）
    test_cases = [
        "查询所有用户",
        "年龄最大的是谁？",  # 依赖上下文
        "刚才那个用户的年龄是多少？",  # 依赖记忆
    ]

    for i, query in enumerate(test_cases, 1):
        logger.info(f"\n{'#'*60}")
        logger.info(f"查询 {i}: {query}")
        logger.info(f"{'#'*60}")

        result = agent.run(query, verbose=True)

        logger.info(f"\n最终结果:")
        logger.info(f"  成功: {result['success']}")
        if result['success']:
            logger.info(f"  答案: {result['answer']}")
        else:
            logger.error(f"  错误: {result.get('error', 'N/A')}")
        logger.info(f"  迭代次数: {result['iterations']}")

    # 统计
    logger.info(f"\n{'='*60}")
    logger.info("统计信息")
    logger.info(f"{'='*60}")
    stats = agent.get_stats()
    logger.info(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    demo_agent_v2()
