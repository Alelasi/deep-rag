"""
测试CodeAgent - 代码生成与执行Agent
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from src.agents.code_agent import (
    CodeAgent,
    CodeGenerator,
    DockerExecutor,
    CodeRequest,
    CodeResult,
    ExecutionResult,
    create_code_agent
)


class TestDockerExecutor:
    """测试Docker执行器"""

    @patch('src.agents.code_agent.docker')
    def test_execute_python_success(self, mock_docker):
        """测试Python代码执行成功"""
        # Mock Docker客户端
        mock_client = Mock()
        mock_container = Mock()

        mock_container.wait.return_value = {"StatusCode": 0}
        mock_container.logs.side_effect = [
            b"Hello, World!\n",  # stdout
            b""                  # stderr
        ]

        mock_client.containers.run.return_value = mock_container
        mock_docker.from_env.return_value = mock_client

        # 执行代码
        executor = DockerExecutor()
        result = executor.execute('print("Hello, World!")', "python")

        # 验证
        assert result.success is True
        assert result.exit_code == 0
        assert "Hello, World!" in result.stdout
        assert result.stderr == ""

    @patch('src.agents.code_agent.docker')
    def test_execute_python_error(self, mock_docker):
        """测试Python代码执行失败"""
        mock_client = Mock()
        mock_container = Mock()

        mock_container.wait.return_value = {"StatusCode": 1}
        mock_container.logs.side_effect = [
            b"",  # stdout
            b"SyntaxError: invalid syntax\n"  # stderr
        ]

        mock_client.containers.run.return_value = mock_container
        mock_docker.from_env.return_value = mock_client

        executor = DockerExecutor()
        result = executor.execute('print(', "python")

        assert result.success is False
        assert result.exit_code == 1
        assert "SyntaxError" in result.stderr

    def test_execute_unsupported_language(self):
        """测试不支持的语言"""
        with patch('src.agents.code_agent.docker.from_env'):
            executor = DockerExecutor()
            result = executor.execute('code', "ruby")

            assert result.success is False
            assert "Unsupported language" in result.stderr

    def test_docker_not_available(self):
        """测试Docker不可用"""
        with patch('src.agents.code_agent.docker.from_env', side_effect=Exception("Docker not found")):
            executor = DockerExecutor()
            result = executor.execute('print("test")', "python")

            assert result.success is False
            assert "Docker client not available" in result.stderr


class TestCodeGenerator:
    """测试代码生成器"""

    def test_generate_without_llm(self):
        """测试没有LLM时的行为"""
        generator = CodeGenerator(llm=None)
        request = CodeRequest(
            description="测试",
            language="python",
            context=None,
            constraints=[],
            test_cases=[]
        )

        result = generator.generate(request)

        assert "LLM not available" in result.code or "LLM not configured" in result.explanation

    def test_generate_with_mock_llm(self):
        """测试使用Mock LLM生成代码"""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = """
【代码说明】
这是一个简单的快速排序实现

【代码】
```python
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
```
"""
        mock_llm.invoke.return_value = mock_response

        generator = CodeGenerator(llm=mock_llm)
        request = CodeRequest(
            description="写一个快速排序",
            language="python",
            context=None,
            constraints=[],
            test_cases=[]
        )

        result = generator.generate(request)

        assert "def quicksort" in result.code
        assert result.language == "python"
        assert "快速排序" in result.explanation

    def test_extract_code_with_language_tag(self):
        """测试提取带语言标签的代码块"""
        generator = CodeGenerator(llm=None)
        content = "这是说明\n```python\nprint('hello')\n```\n更多说明"

        code = generator._extract_code(content, "python")

        assert code == "print('hello')"

    def test_extract_code_without_language_tag(self):
        """测试提取不带语言标签的代码块"""
        generator = CodeGenerator(llm=None)
        content = "说明\n```\nprint('hello')\n```"

        code = generator._extract_code(content, "python")

        assert code == "print('hello')"

    def test_build_prompt(self):
        """测试Prompt构造"""
        generator = CodeGenerator(llm=None)
        request = CodeRequest(
            description="写一个函数",
            language="python",
            context="# existing code",
            constraints=["不使用numpy"],
            test_cases=[{"input": "1", "output": "2"}]
        )

        prompt = generator._build_prompt(request)

        assert "写一个函数" in prompt
        assert "python" in prompt
        assert "不使用numpy" in prompt
        assert "existing code" in prompt
        assert "测试用例" in prompt


class TestCodeAgent:
    """测试CodeAgent主控制器"""

    def test_quick_generate(self):
        """测试快速生成接口"""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "```python\ndef fib(n):\n    return n if n <= 1 else fib(n-1) + fib(n-2)\n```"
        mock_llm.invoke.return_value = mock_response

        agent = CodeAgent(llm=mock_llm)
        code = agent.quick_generate("写一个斐波那契函数")

        assert "def fib" in code

    @patch('src.agents.code_agent.DockerExecutor')
    def test_generate_and_test_without_execution(self, mock_executor_class):
        """测试生成代码但不执行"""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "```python\nprint('test')\n```"
        mock_llm.invoke.return_value = mock_response

        agent = CodeAgent(llm=mock_llm)
        request = CodeRequest(
            description="打印test",
            language="python",
            context=None,
            constraints=[],
            test_cases=[]
        )

        result = agent.generate_and_test(request, execute=False)

        assert "print('test')" in result.code
        assert result.execution_result is None

    @patch('src.agents.code_agent.DockerExecutor')
    def test_generate_and_test_with_execution(self, mock_executor_class):
        """测试生成代码并执行"""
        # Mock LLM
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "```python\nprint('Hello')\n```"
        mock_llm.invoke.return_value = mock_response

        # Mock Executor
        mock_executor = Mock()
        mock_exec_result = ExecutionResult(
            stdout="Hello\n",
            stderr="",
            exit_code=0,
            execution_time=0.1,
            memory_used=10,
            success=True
        )
        mock_executor.execute.return_value = mock_exec_result
        mock_executor_class.return_value = mock_executor

        # 执行
        agent = CodeAgent(llm=mock_llm)
        request = CodeRequest(
            description="打印Hello",
            language="python",
            context=None,
            constraints=[],
            test_cases=[]
        )

        result = agent.generate_and_test(request, execute=True)

        # 验证
        assert "print('Hello')" in result.code
        assert result.execution_result is not None
        assert result.execution_result['success'] is True
        assert "Hello" in result.execution_result['stdout']

    def test_create_code_agent(self):
        """测试工厂函数"""
        agent = create_code_agent()

        assert isinstance(agent, CodeAgent)
        assert agent.generator is not None
        assert agent.executor is not None


# ========== 集成测试（需要真实Docker）==========

@pytest.mark.skip(reason="需要Docker环境")
def test_real_docker_execution():
    """真实Docker执行测试（需要Docker）"""
    executor = DockerExecutor(timeout=10)

    # 测试Python
    result = executor.execute('print("Hello from Docker")', "python")
    assert result.success is True
    assert "Hello from Docker" in result.stdout

    # 测试JavaScript
    result = executor.execute('console.log("Hello from Node")', "javascript")
    assert result.success is True
    assert "Hello from Node" in result.stdout

    # 测试Bash
    result = executor.execute('echo "Hello from Bash"', "bash")
    assert result.success is True
    assert "Hello from Bash" in result.stdout


@pytest.mark.skip(reason="需要LLM和Docker")
def test_end_to_end_code_generation():
    """端到端测试（需要真实LLM和Docker）"""
    agent = create_code_agent()

    request = CodeRequest(
        description="写一个函数计算列表的平均值",
        language="python",
        context=None,
        constraints=["不使用numpy"],
        test_cases=[
            {"input": "[1, 2, 3, 4, 5]", "output": "3.0"}
        ]
    )

    result = agent.generate_and_test(request, execute=True)

    assert result.code
    assert result.execution_result
    assert result.execution_result['success'] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
