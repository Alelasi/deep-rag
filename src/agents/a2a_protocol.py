"""A2A协议基础框架 — v2.9.2新增

实现Agent-to-Agent通信协议，支持多Agent协作。

核心组件：
1. Agent Card — 能力声明JSON（类似API文档）
2. Task状态机 — submitted→working→completed/failed
3. Agent间任务委托机制

参考：Google A2A协议规范 (2025)
面试要点：A2A是Agent间的横向通信，MCP是Agent到工具的纵向通信

用法：
    from src.agents.a2a_protocol import AgentCard, Task, A2AProtocol

    # 1. 声明Agent能力
    card = AgentCard(
        name="research_agent",
        description="多轮知识检索Agent",
        skills=[{"name": "deep_search", "description": "深度知识检索"}],
    )

    # 2. 创建协议实例
    protocol = A2AProtocol()
    protocol.register_agent(card)

    # 3. 委托任务
    task = protocol.delegate_task(
        from_agent="coordinator",
        to_agent="research_agent",
        task_type="deep_search",
        payload={"query": "什么是RAG？"},
    )

    # 4. 查询任务状态
    status = protocol.get_task_status(task.task_id)
"""
import time
import uuid
import logging
import threading
from enum import Enum
from typing import Optional, Dict, List, Any, Callable
from dataclasses import dataclass, field

log = logging.getLogger("deeprag")


# ============================================================
# 1. Task状态机
# ============================================================

class TaskStatus(str, Enum):
    """Task生命周期状态

    submitted → working → completed / failed
    """
    SUBMITTED = "submitted"    # 已提交，等待处理
    WORKING = "working"        # 执行中
    COMPLETED = "completed"    # 已完成
    FAILED = "failed"          # 已失败
    CANCELLED = "cancelled"    # 已取消


@dataclass
class TaskArtifact:
    """任务产出物"""
    name: str                  # 产出物名称
    content: Any               # 产出物内容
    mime_type: str = "text/plain"  # MIME类型
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Task:
    """A2A任务 — 一等公民

    支持完整的生命周期管理：submitted → working → completed/failed
    """
    task_id: str                          # 唯一ID
    from_agent: str                       # 委托方Agent
    to_agent: str                         # 执行方Agent
    task_type: str                        # 任务类型（对应Skill名）
    payload: Dict[str, Any]               # 任务参数
    status: TaskStatus = TaskStatus.SUBMITTED
    artifacts: List[TaskArtifact] = field(default_factory=list)
    error: Optional[str] = None           # 失败原因
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def elapsed_seconds(self) -> Optional[float]:
        """任务耗时（秒）"""
        if self.started_at is None:
            return None
        end = self.completed_at or time.time()
        return round(end - self.started_at, 2)

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "task_id": self.task_id,
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "task_type": self.task_type,
            "status": self.status.value,
            "elapsed_seconds": self.elapsed_seconds,
            "error": self.error,
            "artifacts_count": len(self.artifacts),
            "created_at": self.created_at,
        }


# ============================================================
# 2. Agent Card
# ============================================================

@dataclass
class AgentSkill:
    """Agent技能声明"""
    name: str                  # 技能名称
    description: str           # 技能描述
    input_schema: Optional[Dict] = None   # 输入参数Schema
    output_schema: Optional[Dict] = None  # 输出Schema
    examples: List[Dict] = field(default_factory=list)  # 示例输入


@dataclass
class AgentCard:
    """Agent Card — 能力声明

    类似微服务的API文档，声明Agent能做什么。
    对应A2A规范中的 /.well-known/agent-card.json

    属性：
        name: Agent唯一标识
        description: 一句话描述
        skills: 技能列表
        endpoint: 通信端点（可选，用于远程Agent）
        supports_streaming: 是否支持流式返回
        supports_push: 是否支持异步回调
    """
    name: str
    description: str
    skills: List[AgentSkill] = field(default_factory=list)
    endpoint: Optional[str] = None
    supports_streaming: bool = False
    supports_push: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        """导出为标准Agent Card JSON格式"""
        return {
            "name": self.name,
            "description": self.description,
            "skills": [
                {
                    "name": s.name,
                    "description": s.description,
                    "input_schema": s.input_schema,
                    "output_schema": s.output_schema,
                }
                for s in self.skills
            ],
            "endpoint": self.endpoint,
            "capabilities": {
                "streaming": self.supports_streaming,
                "push_notifications": self.supports_push,
            },
            "metadata": self.metadata,
        }

    def has_skill(self, skill_name: str) -> bool:
        """检查是否拥有指定技能"""
        return any(s.name == skill_name for s in self.skills)

    def match_skill(self, task_description: str) -> Optional[AgentSkill]:
        """根据任务描述匹配最佳技能

        简单关键词匹配，实际项目可用向量相似度。
        """
        task_lower = task_description.lower()
        best_skill = None
        best_score = 0

        for skill in self.skills:
            score = 0
            # 名称匹配
            if skill.name.lower() in task_lower:
                score += 3
            # 描述关键词匹配
            desc_words = skill.description.lower().split()
            for word in desc_words:
                if word in task_lower:
                    score += 1

            if score > best_score:
                best_score = score
                best_skill = skill

        return best_skill if best_score > 0 else None


# ============================================================
# 3. A2A Protocol
# ============================================================

class A2AProtocol:
    """A2A协议管理器

    管理Agent注册、任务委托、状态查询。

    用法：
        protocol = A2AProtocol()

        # 注册Agent
        protocol.register_agent(card)

        # 注册任务处理器
        protocol.register_handler("research_agent", "deep_search", handler_fn)

        # 委托任务
        task = protocol.delegate_task("coordinator", "research_agent", "deep_search", payload)

        # 查询状态
        status = protocol.get_task_status(task.task_id)
    """

    def __init__(self):
        self._agents: Dict[str, AgentCard] = {}
        self._tasks: Dict[str, Task] = {}
        self._handlers: Dict[str, Dict[str, Callable]] = {}  # {agent: {skill: handler}}
        self._lock = threading.Lock()
        log.info("[A2A] 协议初始化完成")

    # === Agent管理 ===

    def register_agent(self, card: AgentCard):
        """注册Agent

        Args:
            card: Agent的能力声明
        """
        with self._lock:
            self._agents[card.name] = card
            if card.name not in self._handlers:
                self._handlers[card.name] = {}
        log.info(f"[A2A] 注册Agent: {card.name}, 技能: {[s.name for s in card.skills]}")

    def unregister_agent(self, agent_name: str):
        """注销Agent"""
        with self._lock:
            self._agents.pop(agent_name, None)
            self._handlers.pop(agent_name, None)
        log.info(f"[A2A] 注销Agent: {agent_name}")

    def get_agent(self, agent_name: str) -> Optional[AgentCard]:
        """获取Agent Card"""
        return self._agents.get(agent_name)

    def list_agents(self) -> List[AgentCard]:
        """列出所有注册的Agent"""
        return list(self._agents.values())

    def find_agent_for_skill(self, skill_name: str) -> Optional[AgentCard]:
        """查找拥有指定技能的Agent"""
        for card in self._agents.values():
            if card.has_skill(skill_name):
                return card
        return None

    def discover_agents(self, task_description: str) -> List[AgentCard]:
        """根据任务描述发现合适的Agent

        返回按匹配度排序的Agent列表。
        """
        matches = []
        for card in self._agents.values():
            skill = card.match_skill(task_description)
            if skill:
                matches.append((card, skill))

        # 按技能名称匹配度排序
        matches.sort(key=lambda x: 0 if x[1].name in task_description else 1)
        return [m[0] for m in matches]

    # === Handler注册 ===

    def register_handler(self, agent_name: str, skill_name: str, handler: Callable):
        """注册任务处理器

        Args:
            agent_name: Agent名称
            skill_name: 技能名称
            handler: 处理函数，签名 fn(payload: Dict) -> Any
        """
        with self._lock:
            if agent_name not in self._handlers:
                self._handlers[agent_name] = {}
            self._handlers[agent_name][skill_name] = handler
        log.debug(f"[A2A] 注册处理器: {agent_name}/{skill_name}")

    # === 任务管理 ===

    def delegate_task(
        self,
        from_agent: str,
        to_agent: str,
        task_type: str,
        payload: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Task:
        """委托任务

        Args:
            from_agent: 委托方Agent名称
            to_agent: 执行方Agent名称
            task_type: 任务类型（对应Skill名）
            payload: 任务参数
            metadata: 附加元数据

        Returns:
            创建的Task对象
        """
        # 验证目标Agent存在
        if to_agent not in self._agents:
            raise ValueError(f"目标Agent未注册: {to_agent}")

        # 验证目标Agent有对应技能
        card = self._agents[to_agent]
        if not card.has_skill(task_type):
            log.warning(f"[A2A] Agent {to_agent} 没有技能 {task_type}，尝试继续...")

        # 创建Task
        task = Task(
            task_id=str(uuid.uuid4())[:8],
            from_agent=from_agent,
            to_agent=to_agent,
            task_type=task_type,
            payload=payload,
            metadata=metadata or {},
        )

        with self._lock:
            self._tasks[task.task_id] = task

        log.info(f"[A2A] 任务委托: {from_agent} → {to_agent}/{task_type} (ID: {task.task_id})")
        return task

    def execute_task(self, task: Task) -> Any:
        """执行任务

        查找注册的处理器并执行。更新任务状态。

        Args:
            task: 待执行的Task

        Returns:
            任务执行结果
        """
        # 更新状态为working
        task.status = TaskStatus.WORKING
        task.started_at = time.time()

        handler = self._handlers.get(task.to_agent, {}).get(task.task_type)

        if handler is None:
            # 没有处理器，尝试使用默认处理器
            log.warning(f"[A2A] 未找到处理器: {task.to_agent}/{task.task_type}")
            result = self._default_handler(task)
        else:
            try:
                result = handler(task.payload)
            except Exception as e:
                log.error(f"[A2A] 任务执行失败: {task.task_id}, 错误: {e}")
                task.status = TaskStatus.FAILED
                task.error = str(e)
                task.completed_at = time.time()
                return None

        # 更新状态为completed
        task.status = TaskStatus.COMPLETED
        task.completed_at = time.time()

        # 存储产出物
        if result is not None:
            artifact = TaskArtifact(
                name=f"{task.task_type}_result",
                content=result,
            )
            task.artifacts.append(artifact)

        log.info(
            f"[A2A] 任务完成: {task.task_id}, "
            f"耗时: {task.elapsed_seconds}s"
        )
        return result

    def delegate_and_execute(
        self,
        from_agent: str,
        to_agent: str,
        task_type: str,
        payload: Dict[str, Any],
    ) -> Any:
        """委托并同步执行任务

        便捷方法，一步完成委托和执行。
        """
        task = self.delegate_task(from_agent, to_agent, task_type, payload)
        return self.execute_task(task)

    def _default_handler(self, task: Task) -> str:
        """默认处理器 — 当未找到注册的处理器时使用"""
        return f"[A2A] 任务 {task.task_id} 已提交但未找到处理器: {task.to_agent}/{task.task_type}"

    # === 任务状态查询 ===

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """查询任务状态"""
        task = self._tasks.get(task_id)
        return task.to_dict() if task else None

    def get_task(self, task_id: str) -> Optional[Task]:
        """获取任务对象"""
        return self._tasks.get(task_id)

    def list_tasks(
        self,
        agent_name: Optional[str] = None,
        status: Optional[TaskStatus] = None,
    ) -> List[Task]:
        """列出任务

        Args:
            agent_name: 按Agent过滤（委托方或执行方）
            status: 按状态过滤
        """
        tasks = list(self._tasks.values())

        if agent_name:
            tasks = [
                t for t in tasks
                if t.from_agent == agent_name or t.to_agent == agent_name
            ]

        if status:
            tasks = [t for t in tasks if t.status == status]

        return tasks

    def cancel_task(self, task_id: str) -> bool:
        """取消任务（仅submitted状态可取消）"""
        task = self._tasks.get(task_id)
        if task and task.status == TaskStatus.SUBMITTED:
            task.status = TaskStatus.CANCELLED
            task.completed_at = time.time()
            log.info(f"[A2A] 任务已取消: {task_id}")
            return True
        return False

    # === 统计 ===

    def get_stats(self) -> Dict[str, Any]:
        """获取协议统计"""
        with self._lock:
            total_tasks = len(self._tasks)
            by_status = {}
            for task in self._tasks.values():
                s = task.status.value
                by_status[s] = by_status.get(s, 0) + 1

            by_agent = {}
            for task in self._tasks.values():
                agent = task.to_agent
                if agent not in by_agent:
                    by_agent[agent] = {"received": 0, "completed": 0, "failed": 0}
                by_agent[agent]["received"] += 1
                if task.status == TaskStatus.COMPLETED:
                    by_agent[agent]["completed"] += 1
                elif task.status == TaskStatus.FAILED:
                    by_agent[agent]["failed"] += 1

            return {
                "registered_agents": len(self._agents),
                "total_tasks": total_tasks,
                "by_status": by_status,
                "by_agent": by_agent,
            }

    def export_agent_cards(self) -> List[Dict[str, Any]]:
        """导出所有Agent Card为JSON格式"""
        return [card.to_json() for card in self._agents.values()]


# ============================================================
# 4. 预定义Agent Cards
# ============================================================

def create_research_agent_card() -> AgentCard:
    """创建研究Agent的能力声明"""
    return AgentCard(
        name="research_agent",
        description="多轮知识检索Agent，支持向量检索、图谱查询、Web搜索",
        skills=[
            AgentSkill(
                name="deep_search",
                description="深度知识检索，多轮检索+结果融合",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "检索问题"},
                        "collection_name": {"type": "string", "description": "知识库集合名"},
                        "max_rounds": {"type": "integer", "description": "最大检索轮数"},
                    },
                    "required": ["query"],
                },
            ),
            AgentSkill(
                name="vector_search",
                description="向量语义检索",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "top_k": {"type": "integer"},
                    },
                    "required": ["query"],
                },
            ),
        ],
    )


def create_verify_agent_card() -> AgentCard:
    """创建验证Agent的能力声明"""
    return AgentCard(
        name="verify_agent",
        description="多源交叉验证和事实核查Agent",
        skills=[
            AgentSkill(
                name="fact_check",
                description="事实核查，验证答案准确性",
                input_schema={
                    "type": "object",
                    "properties": {
                        "claim": {"type": "string", "description": "待验证声明"},
                        "sources": {"type": "array", "description": "参考来源"},
                    },
                    "required": ["claim"],
                },
            ),
            AgentSkill(
                name="cross_validate",
                description="多源交叉验证",
                input_schema={
                    "type": "object",
                    "properties": {
                        "answers": {"type": "array", "description": "待验证答案列表"},
                    },
                    "required": ["answers"],
                },
            ),
        ],
    )


def create_precision_agent_card() -> AgentCard:
    """创建精准模式Agent的能力声明"""
    return AgentCard(
        name="precision_agent",
        description="双Agent精准模式，支持矛盾检测和仲裁",
        skills=[
            AgentSkill(
                name="precision_generate",
                description="双Agent并行生成+矛盾检测",
                input_schema={
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "fast_mode": {"type": "boolean"},
                    },
                    "required": ["question"],
                },
            ),
        ],
    )


# ============================================================
# 5. 全局实例
# ============================================================

_protocol_instance: Optional[A2AProtocol] = None


def get_a2a_protocol() -> A2AProtocol:
    """获取全局A2A协议实例（懒加载）"""
    global _protocol_instance
    if _protocol_instance is None:
        _protocol_instance = A2AProtocol()

        # 注册预定义Agent
        _protocol_instance.register_agent(create_research_agent_card())
        _protocol_instance.register_agent(create_verify_agent_card())
        _protocol_instance.register_agent(create_precision_agent_card())

        log.info("[A2A] 全局协议实例已初始化，注册3个预定义Agent")

    return _protocol_instance


def export_agent_card_json(output_path: str = None) -> Dict[str, Any]:
    """导出Agent Card为JSON格式

    符合A2A规范的 /.well-known/agent-card.json 格式。

    Args:
        output_path: 可选的输出文件路径

    Returns:
        Agent Card字典
    """
    import json
    from datetime import datetime

    protocol = get_a2a_protocol()
    cards = protocol.export_agent_cards()

    agent_card = {
        "schemaVersion": "1.0",
        "name": "deeprag-multi-agent-system",
        "description": "DeepRAG企业级Multi-Agent RAG系统",
        "version": "2.9.2",
        "capabilities": {
            "streaming": False,
            "push_notifications": False,
        },
        "agents": cards,
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "protocol": "a2a",
            "skills_count": sum(len(a.get("skills", [])) for a in cards),
        },
    }

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(agent_card, f, ensure_ascii=False, indent=2)
        log.info(f"[A2A] Agent Card已导出: {output_path}")

    return agent_card
