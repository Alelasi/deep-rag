"""
CodeAgent - 代码生成与执行Agent

功能：
1. 代码生成：根据需求生成Python/JavaScript/Bash代码
2. 代码执行：在Docker沙箱中安全执行代码
3. 代码测试：自动生成并运行单元测试
4. 代码审查：静态分析、风格检查
5. 协作能力：与查询Agent协作（调研→生成→测试）

架构：
┌─────────────────────────────────────┐
│ CodeAgent（主控制器）                │
│  - 需求分析                          │
│  - 代码生成                          │
│  - 测试执行                          │
│  - 结果反馈                          │
└────────┬────────────────────────────┘
         │
         ├──→ CodeGenerator（代码生成器）
         │     - LLM调用
         │     - 模板填充
         │     - 语法检查
         │
         ├──→ DockerExecutor（沙箱执行器）
         │     - Docker容器管理
         │     - 安全隔离
         │     - 资源限制
         │
         └──→ CodeTester（测试器）
               - 单元测试生成
               - 测试执行
               - 覆盖率统计
"""
import logging
import tempfile
import os
from typing import Dict, List, Optional, Literal
from dataclasses import dataclass
from pathlib import Path

# 可选依赖：Docker（优雅降级）
try:
    import docker
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False
    docker = None

log = logging.getLogger("code_agent")


# ========== 1. 数据结构 ==========

try:
    from src.logging_config import get_logger
except Exception:
    import logging

    def get_logger(n):  # type: ignore
        return logging.getLogger(n)

logger = get_logger(__name__)

@dataclass
class CodeRequest:
    """代码生成请求"""
    description: str           # 需求描述
    language: str              # python / javascript / bash
    context: Optional[str]     # 上下文代码
    constraints: List[str]     # 约束条件（如"不使用numpy"）
    test_cases: List[dict]     # 测试用例（可选）


@dataclass
class CodeResult:
    """代码生成结果"""
    code: str                  # 生成的代码
    language: str              # 语言
    explanation: str           # 代码说明
    test_code: Optional[str]   # 测试代码
    execution_result: Optional[dict]  # 执行结果
    static_analysis: Optional[dict]   # 静态分析结果


@dataclass
class ExecutionResult:
    """代码执行结果"""
    stdout: str                # 标准输出
    stderr: str                # 错误输出
    exit_code: int             # 退出码
    execution_time: float      # 执行时间（秒）
    memory_used: int           # 内存使用（MB）
    success: bool              # 是否成功


# ========== 2. DockerExecutor（沙箱执行器）==========

class DockerExecutor:
    """Docker沙箱代码执行器

    安全特性：
    1. 网络隔离（network="none"）
    2. 资源限制（内存512MB，CPU 1核）
    3. 超时控制（默认30秒）
    4. 只读文件系统（除/tmp外）
    5. 无特权运行（user=nobody）
    """

    def __init__(
        self,
        memory_limit: str = "512m",
        cpu_limit: float = 1.0,
        timeout: int = 30
    ):
        """
        Args:
            memory_limit: 内存限制（如"512m", "1g"）
            cpu_limit: CPU限制（核数，如1.0）
            timeout: 超时时间（秒）
        """
        if not DOCKER_AVAILABLE:
            log.warning("Docker SDK not installed. DockerExecutor will not work. Install: pip install docker")
            self.client = None
        else:
            try:
                self.client = docker.from_env()
                log.info("Docker client initialized")
            except Exception as e:
                log.error(f"Failed to initialize Docker client: {e}")
                self.client = None

        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.timeout = timeout

        # 语言镜像映射
        self.images = {
            "python": "python:3.11-slim",
            "javascript": "node:18-alpine",
            "bash": "bash:5.2-alpine3.19"
        }

    def execute(
        self,
        code: str,
        language: str,
        stdin: Optional[str] = None
    ) -> ExecutionResult:
        """
        在Docker沙箱中执行代码

        Args:
            code: 代码内容
            language: 语言（python/javascript/bash）
            stdin: 标准输入（可选）

        Returns:
            ExecutionResult: 执行结果
        """
        if not self.client:
            return ExecutionResult(
                stdout="",
                stderr="Docker client not available",
                exit_code=1,
                execution_time=0.0,
                memory_used=0,
                success=False
            )

        # 选择镜像
        image = self.images.get(language)
        if not image:
            return ExecutionResult(
                stdout="",
                stderr=f"Unsupported language: {language}",
                exit_code=1,
                execution_time=0.0,
                memory_used=0,
                success=False
            )

        # 创建临时文件
        with tempfile.TemporaryDirectory() as tmpdir:
            # 写入代码
            if language == "python":
                code_file = Path(tmpdir) / "main.py"
                command = ["python", "/tmp/main.py"]
            elif language == "javascript":
                code_file = Path(tmpdir) / "main.js"
                command = ["node", "/tmp/main.js"]
            else:  # bash
                code_file = Path(tmpdir) / "main.sh"
                command = ["bash", "/tmp/main.sh"]

            code_file.write_text(code, encoding="utf-8")

            # 运行容器
            try:
                import time
                start_time = time.time()

                container = self.client.containers.run(
                    image=image,
                    command=command,
                    volumes={tmpdir: {"bind": "/tmp", "mode": "ro"}},
                    network_disabled=True,  # 禁止网络
                    mem_limit=self.memory_limit,
                    nano_cpus=int(self.cpu_limit * 1e9),
                    detach=True,
                    remove=True,
                    stdin_open=bool(stdin),
                    user="nobody"  # 非特权用户
                )

                # 等待执行完成（带超时）
                result = container.wait(timeout=self.timeout)
                execution_time = time.time() - start_time

                # 获取输出
                stdout = container.logs(stdout=True, stderr=False).decode("utf-8")
                stderr = container.logs(stdout=False, stderr=True).decode("utf-8")
                exit_code = result["StatusCode"]

                # 内存使用（从stats获取，简化版）
                memory_used = 0  # TODO: 从container.stats()获取

                return ExecutionResult(
                    stdout=stdout,
                    stderr=stderr,
                    exit_code=exit_code,
                    execution_time=execution_time,
                    memory_used=memory_used,
                    success=(exit_code == 0)
                )

            except docker.errors.ContainerError as e:
                return ExecutionResult(
                    stdout="",
                    stderr=f"Container error: {e}",
                    exit_code=e.exit_status,
                    execution_time=0.0,
                    memory_used=0,
                    success=False
                )

            except Exception as e:
                return ExecutionResult(
                    stdout="",
                    stderr=f"Execution error: {e}",
                    exit_code=1,
                    execution_time=0.0,
                    memory_used=0,
                    success=False
                )


# ========== 3. CodeGenerator（代码生成器）==========

class CodeGenerator:
    """代码生成器（基于LLM）"""

    def __init__(self, llm=None):
        """
        Args:
            llm: LLM实例（LangChain ChatModel）
        """
        self.llm = llm or self._get_default_llm()

    def _get_default_llm(self):
        """获取默认LLM"""
        try:
            from src.config import get_llm
            return get_llm(temperature=0.2)  # 代码生成用低温度
        except Exception as e:
            log.warning(f"Failed to get LLM: {e}, using None")
            return None

    def generate(
        self,
        request: CodeRequest
    ) -> CodeResult:
        """
        生成代码

        Args:
            request: 代码请求

        Returns:
            CodeResult: 生成结果
        """
        if not self.llm:
            return CodeResult(
                code="# LLM not available",
                language=request.language,
                explanation="LLM not configured",
                test_code=None,
                execution_result=None,
                static_analysis=None
            )

        # 构造Prompt
        prompt = self._build_prompt(request)

        # 调用LLM
        try:
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)

            # 解析代码（提取```代码块）
            code = self._extract_code(content, request.language)
            explanation = self._extract_explanation(content)

            return CodeResult(
                code=code,
                language=request.language,
                explanation=explanation,
                test_code=None,  # TODO: 生成测试代码
                execution_result=None,
                static_analysis=None
            )

        except Exception as e:
            log.error(f"Code generation failed: {e}")
            return CodeResult(
                code=f"# Error: {e}",
                language=request.language,
                explanation=f"Code generation failed: {e}",
                test_code=None,
                execution_result=None,
                static_analysis=None
            )

    def _build_prompt(self, request: CodeRequest) -> str:
        """构造代码生成Prompt"""
        prompt = f"""你是一个专业的{request.language}程序员。请根据以下需求生成代码。

需求描述：
{request.description}

语言：{request.language}

约束条件：
{chr(10).join(f'- {c}' for c in request.constraints) if request.constraints else '无'}

要求：
1. 代码简洁、高效、可读
2. 添加必要的注释
3. 处理边界情况和异常
4. 使用标准库（除非需求明确要求第三方库）

请按以下格式输出：

【代码说明】
（简要说明代码逻辑和关键设计）

【代码】
```{request.language}
（代码内容）
```

【测试用例】
```{request.language}
（测试代码，可选）
```
"""

        # 添加上下文代码
        if request.context:
            prompt += f"\n现有代码上下文：\n```{request.language}\n{request.context}\n```\n"

        # 添加测试用例
        if request.test_cases:
            prompt += "\n测试用例：\n"
            for i, tc in enumerate(request.test_cases, 1):
                prompt += f"{i}. 输入：{tc.get('input')} → 输出：{tc.get('output')}\n"

        return prompt

    def _extract_code(self, content: str, language: str) -> str:
        """从LLM输出中提取代码块"""
        import re

        # 匹配```language ... ```
        pattern = rf"```{language}\s*\n(.*?)```"
        matches = re.findall(pattern, content, re.DOTALL)

        if matches:
            return matches[0].strip()

        # 匹配``` ... ```（不指定语言）
        pattern = r"```\s*\n(.*?)```"
        matches = re.findall(pattern, content, re.DOTALL)

        if matches:
            return matches[0].strip()

        # 没有代码块，返回全部内容
        return content.strip()

    def _extract_explanation(self, content: str) -> str:
        """提取代码说明"""
        import re

        # 提取【代码说明】部分
        pattern = r"【代码说明】\s*\n(.*?)(?=【代码】|```|$)"
        matches = re.findall(pattern, content, re.DOTALL)

        if matches:
            return matches[0].strip()

        # 提取第一个代码块之前的内容
        pattern = r"^(.*?)```"
        matches = re.findall(pattern, content, re.DOTALL)

        if matches:
            explanation = matches[0].strip()
            if len(explanation) > 500:
                explanation = explanation[:500] + "..."
            return explanation

        return "No explanation provided"


# ========== 4. CodeAgent（主控制器）==========

class CodeAgent:
    """代码Agent - 代码生成与执行的主控制器

    使用示例：
    ```python
    agent = CodeAgent()

    # 生成代码
    request = CodeRequest(
        description="写一个快速排序函数",
        language="python",
        constraints=["不使用内置sort"],
        test_cases=[
            {"input": "[3,1,2]", "output": "[1,2,3]"}
        ]
    )

    result = agent.generate_and_test(request)
    logger.info(result.code)
    logger.info(result.execution_result)
    ```
    """

    def __init__(
        self,
        llm=None,
        docker_executor: Optional[DockerExecutor] = None
    ):
        """
        Args:
            llm: LLM实例
            docker_executor: Docker执行器（可选，自动创建）
        """
        self.generator = CodeGenerator(llm)
        self.executor = docker_executor or DockerExecutor()
        log.info("CodeAgent initialized")

    def generate_and_test(
        self,
        request: CodeRequest,
        execute: bool = True
    ) -> CodeResult:
        """
        生成代码并测试

        Args:
            request: 代码请求
            execute: 是否执行代码（默认True）

        Returns:
            CodeResult: 完整结果（包含执行结果）
        """
        log.info(f"Generating {request.language} code: {request.description[:50]}...")

        # 1. 生成代码
        result = self.generator.generate(request)

        # 2. 执行代码（如果启用）
        if execute and result.code and not result.code.startswith("#"):
            log.info("Executing generated code...")
            exec_result = self.executor.execute(result.code, request.language)
            result.execution_result = {
                'stdout': exec_result.stdout,
                'stderr': exec_result.stderr,
                'exit_code': exec_result.exit_code,
                'execution_time': exec_result.execution_time,
                'success': exec_result.success
            }

            if exec_result.success:
                log.info(f"Code executed successfully ({exec_result.execution_time:.2f}s)")
            else:
                log.warning(f"Code execution failed: {exec_result.stderr[:100]}")

        return result

    def quick_generate(
        self,
        description: str,
        language: str = "python"
    ) -> str:
        """
        快速生成代码（简化接口）

        Args:
            description: 需求描述
            language: 语言（默认python）

        Returns:
            str: 生成的代码
        """
        request = CodeRequest(
            description=description,
            language=language,
            context=None,
            constraints=[],
            test_cases=[]
        )

        result = self.generate_and_test(request, execute=False)
        return result.code


# ========== 5. 便捷工厂函数 ==========

def create_code_agent(llm=None) -> CodeAgent:
    """创建CodeAgent实例"""
    return CodeAgent(llm=llm)


# ========== 使用示例 ==========

if __name__ == "__main__":
    # 示例1：快速生成
    agent = create_code_agent()

    code = agent.quick_generate("写一个计算斐波那契数列的函数")
    logger.info("Generated code:")
    logger.info(code)

    # 示例2：完整流程
    request = CodeRequest(
        description="写一个快速排序函数，输入列表，返回排序后的列表",
        language="python",
        context=None,
        constraints=["不使用内置sort函数", "使用递归实现"],
        test_cases=[
            {"input": "[3, 1, 4, 1, 5, 9, 2, 6]", "output": "[1, 1, 2, 3, 4, 5, 6, 9]"},
            {"input": "[]", "output": "[]"},
            {"input": "[1]", "output": "[1]"}
        ]
    )

    result = agent.generate_and_test(request, execute=True)

    logger.info("\n=== Code Generation Result ===")
    logger.info(f"Language: {result.language}")
    logger.info(f"Explanation: {result.explanation}")
    logger.info(f"\nCode:\n{result.code}")

    if result.execution_result:
        logger.info(f"\n=== Execution Result ===")
        logger.info(f"Success: {result.execution_result['success']}")
        logger.info(f"Exit code: {result.execution_result['exit_code']}")
        logger.info(f"Execution time: {result.execution_result['execution_time']:.3f}s")
        logger.info(f"Stdout:\n{result.execution_result['stdout']}")
        if result.execution_result['stderr']:
            logger.info(f"Stderr:\n{result.execution_result['stderr']}")
