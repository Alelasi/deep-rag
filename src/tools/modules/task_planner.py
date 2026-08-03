"""
任务规划模块（Planning）
将复杂查询拆解为多个子任务
"""

from typing import List, Dict, Any
import json
import httpx


try:
    from src.logging_config import get_logger
except Exception:
    import logging

    def get_logger(n):  # type: ignore
        return logging.getLogger(n)

logger = get_logger(__name__)

class TaskPlanner:
    """
    任务规划器

    核心能力：
    1. 将复杂查询拆解为多个子任务
    2. 支持多步SQL查询（先COUNT，再找MAX等）
    3. 识别任务依赖关系

    示例：
    输入："查询用户总数，然后找出年龄最大的用户"
    输出：[
        {"step": 1, "task": "统计用户总数", "sql": "SELECT COUNT(*) FROM users"},
        {"step": 2, "task": "找年龄最大的用户", "sql": "SELECT * FROM users ORDER BY age DESC LIMIT 1"}
    ]
    """

    def __init__(
        self,
        llm_base_url: str = "http://localhost:11434/v1/chat/completions",
        model: str = "gemma-3-4b-it"
    ):
        self.llm_base_url = llm_base_url
        self.model = model

    def plan(self, user_query: str, verbose: bool = False) -> Dict[str, Any]:
        """
        规划任务

        Args:
            user_query: 用户查询
            verbose: 是否打印详细日志

        Returns:
            {
                "is_complex": bool,  # 是否需要多步骤
                "subtasks": List[Dict],  # 子任务列表
                "original_query": str  # 原始查询
            }
        """
        # 构建规划Prompt
        system_prompt = self._build_planning_prompt()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]

        try:
            # 调用LLM进行规划
            response = self._call_llm(messages)
            plan_result = self._parse_plan_output(response)

            if verbose:
                logger.info(f"[TaskPlanner] 查询类型: {'复杂任务' if plan_result['is_complex'] else '简单任务'}")
                if plan_result['is_complex']:
                    logger.info(f"[TaskPlanner] 拆解为 {len(plan_result['subtasks'])} 个子任务:")
                    for subtask in plan_result['subtasks']:
                        logger.info(f"  步骤{subtask['step']}: {subtask['task']}")

            return plan_result

        except Exception as e:
            # 规划失败，降级为简单任务
            if verbose:
                logger.error(f"[TaskPlanner] 规划失败: {str(e)}")
                logger.info(f"[TaskPlanner] 降级为简单任务")

            return {
                "is_complex": False,
                "subtasks": [{
                    "step": 1,
                    "task": user_query,
                    "sql": None
                }],
                "original_query": user_query
            }

    def _build_planning_prompt(self) -> str:
        """构建规划Prompt"""
        return """你是一个任务规划专家，负责分析SQL查询需求。

# 任务
判断用户查询是否需要多步骤执行，如果需要则拆解为子任务。

# 输出格式（JSON）
{
  "is_complex": true/false,
  "subtasks": [
    {"step": 1, "task": "任务描述", "sql": "SELECT ..."},
    {"step": 2, "task": "任务描述", "sql": "SELECT ..."}
  ]
}

# 判断标准
**复杂任务**（is_complex=true）：
- 包含"先...再..."、"然后"、"接着"等连接词
- 需要基于第一个查询结果进行第二次查询
- 需要多表关联后再聚合

**简单任务**（is_complex=false）：
- 单个SELECT语句可以完成
- 一次查询返回结果

# 示例

## 示例1：复杂任务
用户: 查询用户总数，然后找出年龄最大的用户
输出:
{
  "is_complex": true,
  "subtasks": [
    {"step": 1, "task": "统计用户总数", "sql": "SELECT COUNT(*) FROM users"},
    {"step": 2, "task": "找年龄最大的用户", "sql": "SELECT * FROM users ORDER BY age DESC LIMIT 1"}
  ]
}

## 示例2：简单任务
用户: 查询年龄大于25的用户
输出:
{
  "is_complex": false,
  "subtasks": [
    {"step": 1, "task": "查询年龄大于25的用户", "sql": "SELECT * FROM users WHERE age > 25"}
  ]
}

## 示例3：复杂任务（基于结果的查询）
用户: 找出订单金额最高的用户的详细信息
输出:
{
  "is_complex": true,
  "subtasks": [
    {"step": 1, "task": "找订单金额最高的用户ID", "sql": "SELECT user_id FROM orders ORDER BY amount DESC LIMIT 1"},
    {"step": 2, "task": "查询该用户详细信息", "sql": "SELECT * FROM users WHERE id = ?"}
  ]
}

# 注意
- 只输出JSON，不要有其他文字
- sql字段必须是合法的SQL语句
- 对于简单任务，is_complex=false，subtasks只有1个元素"""

    def _call_llm(self, messages: List[Dict]) -> str:
        """调用LLM"""
        client = httpx.Client(timeout=30.0)
        response = client.post(
            self.llm_base_url,
            json={
                "model": self.model,
                "messages": messages,
                "max_tokens": 300,
                "temperature": 0.1
            }
        )

        if response.status_code != 200:
            raise Exception(f"LLM调用失败: HTTP {response.status_code}")

        result = response.json()
        content = result["choices"][0]["message"]["content"]

        if not content:
            raise Exception("LLM返回空内容")

        return content

    def _parse_plan_output(self, content: str) -> Dict[str, Any]:
        """
        解析规划输出

        Args:
            content: LLM返回的JSON字符串

        Returns:
            规划结果
        """
        # 提取JSON
        start_idx = content.find('{')
        end_idx = content.rfind('}')

        if start_idx == -1 or end_idx == -1:
            raise Exception(f"未找到JSON格式: {content[:100]}")

        json_str = content[start_idx:end_idx+1]

        try:
            data = json.loads(json_str)

            # 验证必需字段
            if "is_complex" not in data:
                raise Exception("缺少is_complex字段")
            if "subtasks" not in data or not isinstance(data["subtasks"], list):
                raise Exception("缺少subtasks字段或格式错误")

            # 验证每个subtask
            for i, subtask in enumerate(data["subtasks"], 1):
                if "step" not in subtask:
                    subtask["step"] = i
                if "task" not in subtask:
                    raise Exception(f"subtask {i} 缺少task字段")

            return data

        except json.JSONDecodeError as e:
            raise Exception(f"JSON解析失败: {str(e)}\n内容: {json_str[:200]}")


# ============================================================================
# 使用示例
# ============================================================================

def demo_task_planner():
    """演示任务规划器"""
    planner = TaskPlanner()

    test_cases = [
        "查询所有用户",
        "查询用户总数，然后找出年龄最大的用户",
        "找出订单金额最高的用户的详细信息",
        "统计每个城市的用户数量",
    ]

    for query in test_cases:
        logger.info(f"\n{'='*60}")
        logger.info(f"查询: {query}")
        logger.info(f"{'='*60}")

        result = planner.plan(query, verbose=True)

        logger.info(f"\n结果:")
        logger.info(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    demo_task_planner()
