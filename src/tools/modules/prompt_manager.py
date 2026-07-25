"""
Prompt版本管理模块（v2.9: 接入主Pipeline）
支持Prompt版本控制和A/B测试

v2.9 改进：
  - 新增 get_prompt() 便捷方法（自动加载最新版本+变量替换）
  - 新增 get_prompt_ab() A/B测试方法
  - 新增全局单例 _pm（指向项目 prompts/ 目录）
  - init_pipeline_prompts() 初始化所有Pipeline用到的prompt
"""

import json
import random
from typing import Dict, Any, Optional
from pathlib import Path


class PromptManager:
    """
    Prompt版本管理器

    核心能力：
    1. 版本化管理Prompt（v1/v2/v3等）
    2. 支持热切换版本（无需重启）
    3. A/B测试对比不同版本效果
    4. Prompt模板变量替换

    文件结构：
    prompts/
        system_prompt_v1.txt
        system_prompt_v2.txt
        planning_prompt_v1.txt
        metadata.json  # 版本元数据
    """

    def __init__(self, prompts_dir: str = "prompts"):
        """
        初始化Prompt管理器

        Args:
            prompts_dir: Prompt文件目录
        """
        self.prompts_dir = Path(prompts_dir)
        self.prompts_dir.mkdir(parents=True, exist_ok=True)

        self.metadata_file = self.prompts_dir / "metadata.json"
        self.metadata = self._load_metadata()

        # 缓存（避免重复读取文件）
        self.cache: Dict[str, str] = {}

    def load_prompt(
        self,
        prompt_name: str,
        version: str = "latest",
        variables: Optional[Dict[str, str]] = None
    ) -> str:
        """
        加载指定版本的Prompt

        Args:
            prompt_name: Prompt名称（如system_prompt）
            version: 版本号（v1/v2/latest）
            variables: 模板变量（可选）

        Returns:
            Prompt内容
        """
        # 解析版本
        if version == "latest":
            version = self.get_latest_version(prompt_name)

        # 构建文件名
        file_name = f"{prompt_name}_{version}.txt"
        file_path = self.prompts_dir / file_name

        # 检查文件是否存在
        if not file_path.exists():
            raise FileNotFoundError(
                f"Prompt文件不存在: {file_path}\n"
                f"可用版本: {self.list_versions(prompt_name)}"
            )

        # 检查缓存
        cache_key = f"{prompt_name}_{version}"
        if cache_key in self.cache:
            prompt = self.cache[cache_key]
        else:
            # 读取文件
            with open(file_path, 'r', encoding='utf-8') as f:
                prompt = f.read()
            self.cache[cache_key] = prompt

        # 替换变量
        if variables:
            for key, value in variables.items():
                placeholder = f"{{{{{key}}}}}"  # {{variable}}
                prompt = prompt.replace(placeholder, value)

        return prompt

    def save_prompt(
        self,
        prompt_name: str,
        version: str,
        content: str,
        description: str = "",
        author: str = ""
    ):
        """
        保存新版本Prompt

        Args:
            prompt_name: Prompt名称
            version: 版本号（如v1/v2）
            content: Prompt内容
            description: 版本描述
            author: 作者
        """
        # 保存文件
        file_name = f"{prompt_name}_{version}.txt"
        file_path = self.prompts_dir / file_name

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        # 更新元数据
        if prompt_name not in self.metadata:
            self.metadata[prompt_name] = {"versions": []}

        version_info = {
            "version": version,
            "file": file_name,
            "description": description,
            "author": author,
            "created_at": self._get_timestamp()
        }

        self.metadata[prompt_name]["versions"].append(version_info)
        self._save_metadata()

        # 清除缓存
        cache_key = f"{prompt_name}_{version}"
        if cache_key in self.cache:
            del self.cache[cache_key]

        print(f"✅ 保存Prompt成功: {file_path}")

    def list_versions(self, prompt_name: str) -> list:
        """
        列出所有版本

        Args:
            prompt_name: Prompt名称

        Returns:
            版本列表
        """
        if prompt_name not in self.metadata:
            return []

        return [v["version"] for v in self.metadata[prompt_name]["versions"]]

    def get_latest_version(self, prompt_name: str) -> str:
        """
        获取最新版本号

        Args:
            prompt_name: Prompt名称

        Returns:
            最新版本号（如v2）
        """
        versions = self.list_versions(prompt_name)
        if not versions:
            raise ValueError(f"Prompt [{prompt_name}] 没有任何版本")

        # 返回最后一个版本
        return versions[-1]

    def compare_versions(
        self,
        prompt_name: str,
        version1: str,
        version2: str
    ) -> Dict[str, str]:
        """
        对比两个版本的差异

        Args:
            prompt_name: Prompt名称
            version1: 版本1
            version2: 版本2

        Returns:
            {
                "version1": "内容...",
                "version2": "内容...",
                "diff_lines": 行数差异
            }
        """
        content1 = self.load_prompt(prompt_name, version1)
        content2 = self.load_prompt(prompt_name, version2)

        lines1 = content1.split('\n')
        lines2 = content2.split('\n')

        return {
            "version1": content1,
            "version2": content2,
            "lines1": len(lines1),
            "lines2": len(lines2),
            "diff_lines": len(lines2) - len(lines1)
        }

    def get_version_info(self, prompt_name: str, version: str) -> Optional[Dict]:
        """
        获取版本元数据

        Args:
            prompt_name: Prompt名称
            version: 版本号

        Returns:
            版本信息字典
        """
        if prompt_name not in self.metadata:
            return None

        for v in self.metadata[prompt_name]["versions"]:
            if v["version"] == version:
                return v

        return None

    def get_prompt(self, name: str, variables: Optional[Dict[str, str]] = None) -> str:
        """便捷方法：加载最新版本的prompt并替换变量（v2.9新增）

        Args:
            name:      Prompt名称（如 "generation_system"）
            variables: 变量字典（如 {"tools_desc": "..."} ）

        Returns:
            Prompt文本

        Raises:
            ValueError: Prompt不存在
        """
        version = self.get_latest_version(name)
        return self.load_prompt(name, version=version, variables=variables)

    def get_prompt_ab(self, name: str, ratio: float = 0.5) -> tuple:
        """A/B测试方法：按流量比例随机选择版本（v2.9新增）

        Args:
            name:  Prompt名称
            ratio: v1版本的流量比例（0.0~1.0），如0.5表示50%概率用v1

        Returns:
            (版本号, prompt文本)
        """
        versions = self.list_versions(name)
        if not versions:
            raise ValueError(f"Prompt [{name}] 不存在")

        if len(versions) == 1:
            return versions[0], self.load_prompt(name, versions[0])

        # 按ratio概率选择v1或最新版本
        if random.random() < ratio:
            selected = versions[0]
        else:
            selected = versions[-1]

        return selected, self.load_prompt(name, selected)

    def _load_metadata(self) -> Dict:
        """加载元数据"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _save_metadata(self):
        """保存元数据"""
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _get_timestamp() -> str:
        """获取时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()


# ============================================================================
# 初始化默认Prompt
# ============================================================================

def init_default_prompts(manager: PromptManager):
    """初始化默认Prompt版本"""

    # System Prompt v1
    system_prompt_v1 = """你是一个数据分析助手，使用ReAct模式工作。

# 核心规则（必须遵守）
1. 你只能执行**只读查询**，绝对禁止任何写操作
2. 你只有1个工具：search_database（只读SQL）
3. 任何INSERT/UPDATE/DELETE请求都必须拒绝

# ReAct工作流程
每一轮你必须输出JSON格式：

{"thought": "我的思考...", "action": "search_database", "action_input": {"sql_query": "SELECT ..."}}

或者完成时输出：

{"thought": "任务完成", "final_answer": "最终答案"}

# 可用工具
{{tools_desc}}

# 安全约束（强制）
- 只允许SELECT查询
- 禁止INSERT/UPDATE/DELETE/DROP等任何写操作
- 用户要求写操作时，回复："抱歉，我只能执行只读查询"
"""

    manager.save_prompt(
        prompt_name="system_prompt",
        version="v1",
        content=system_prompt_v1,
        description="初始版本，基础ReAct + 安全约束",
        author="system"
    )

    print("✅ 初始化默认Prompt完成")


# ============================================================================
# 使用示例
# ============================================================================

def demo_prompt_manager():
    """演示Prompt管理器"""
    manager = PromptManager(prompts_dir="prompts")

    # 初始化默认Prompt
    if not manager.list_versions("system_prompt"):
        init_default_prompts(manager)

    print("=" * 60)
    print("加载Prompt")
    print("=" * 60)

    # 加载最新版本
    prompt = manager.load_prompt(
        "system_prompt",
        version="latest",
        variables={"tools_desc": "[工具列表...]"}
    )

    print(f"最新版本: {manager.get_latest_version('system_prompt')}")
    print(f"Prompt内容（前200字）:\n{prompt[:200]}...")

    # 列出所有版本
    print("\n" + "=" * 60)
    print("所有版本")
    print("=" * 60)
    versions = manager.list_versions("system_prompt")
    for v in versions:
        info = manager.get_version_info("system_prompt", v)
        print(f"- {v}: {info['description']}")


if __name__ == "__main__":
    demo_prompt_manager()


# ============================================================================
# 全局单例（v2.9: Pipeline统一入口）
# ============================================================================

# 全局PromptManager单例（懒加载）
_pm: Optional[PromptManager] = None


def _get_pm() -> PromptManager:
    """获取全局PromptManager单例（指向项目 prompts/ 目录）"""
    global _pm
    if _pm is None:
        from pathlib import Path
        # 项目根目录下的 prompts/ 文件夹
        project_root = Path(__file__).parent.parent.parent  # src/tools/modules/ -> project root
        prompts_dir = project_root / "prompts"
        _pm = PromptManager(prompts_dir=str(prompts_dir))
    return _pm


def get_pipeline_prompt(name: str, variables: Optional[Dict[str, str]] = None) -> str:
    """Pipeline统一入口：加载最新版本prompt并替换变量

    Args:
        name:      Prompt名称（如 "generation_system", "strategy_direct"）
        variables: 变量字典

    Returns:
        Prompt文本

    Raises:
        ValueError: Prompt不存在（需要先在 prompts/ 目录创建模板文件）
    """
    return _get_pm().get_prompt(name, variables=variables)


def get_pipeline_prompt_ab(name: str, ratio: float = 0.5) -> tuple:
    """Pipeline A/B测试入口：按流量比例随机选择版本

    Args:
        name:  Prompt名称
        ratio: v1版本的流量比例

    Returns:
        (版本号, prompt文本)
    """
    return _get_pm().get_prompt_ab(name, ratio=ratio)
