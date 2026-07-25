"""
创建测试数据库并对比Agent查询结果
"""

import sqlite3
import sys
sys.path.insert(0, '.')

from src.tools import register_builtin_tools, AgentExecutor, get_registry

# ============================================================================
# 步骤1: 创建真实数据库
# ============================================================================

print('='*70)
print('步骤1: 创建测试数据库')
print('='*70)

# 创建数据库
conn = sqlite3.connect('test_database.db')
cursor = conn.cursor()

# 创建users表
cursor.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    age INTEGER NOT NULL,
    email TEXT,
    created_at TEXT
)
''')

# 插入测试数据
test_users = [
    (1, '张三', 23, 'zhangsan@example.com', '2024-01-15'),
    (2, '李四', 28, 'lisi@example.com', '2024-02-20'),
    (3, '王五', 26, 'wangwu@example.com', '2024-03-10'),
    (4, '赵六', 32, 'zhaoliu@example.com', '2024-04-05'),
    (5, '钱七', 19, 'qianqi@example.com', '2024-05-12'),
]

cursor.execute('DELETE FROM users')  # 清空旧数据
cursor.executemany('INSERT INTO users VALUES (?, ?, ?, ?, ?)', test_users)
conn.commit()

print(f'✅ 数据库创建成功: test_database.db')
print(f'   - 表: users')
print(f'   - 记录数: {len(test_users)}')

# 显示所有数据
cursor.execute('SELECT * FROM users ORDER BY id')
all_users = cursor.fetchall()
print(f'\n所有用户数据:')
for user in all_users:
    print(f'   {user}')

# ============================================================================
# 步骤2: 修改search_database函数连接真实数据库
# ============================================================================

print('\n' + '='*70)
print('步骤2: 配置Agent使用真实数据库')
print('='*70)

# 临时修改search_database函数
from src.tools import builtin_tools

def search_database_real(sql_query: str):
    """连接真实数据库的search_database"""
    import re

    # 安全检查（复用原有逻辑）
    query_lower = sql_query.lower().strip()

    if not query_lower.startswith("select"):
        raise ValueError("❌ 只允许SELECT查询")

    dangerous = [
        "insert", "update", "delete", "drop", "truncate", "alter",
        "create", "replace", "merge", "exec", "execute", "call",
        "grant", "revoke", "into", "set"
    ]
    for kw in dangerous:
        if re.search(rf'\b{kw}\b', query_lower):
            raise ValueError(f"❌ 禁止使用 {kw.upper()} 操作")

    # 强制LIMIT 100
    if "limit" not in query_lower:
        sql_query = sql_query.rstrip(";") + " LIMIT 100"
    else:
        limit_match = re.search(r'limit\s+(\d+)', query_lower)
        if limit_match and int(limit_match.group(1)) > 100:
            sql_query = re.sub(r'limit\s+\d+', 'LIMIT 100', sql_query, flags=re.IGNORECASE)

    # 连接真实数据库
    conn = sqlite3.connect('test_database.db', timeout=5.0)
    conn.row_factory = sqlite3.Row  # 返回字典格式
    cursor = conn.execute(sql_query)

    # 转为字典列表
    results = [dict(row) for row in cursor.fetchall()]
    conn.close()

    return results

# 替换函数
builtin_tools.search_database = search_database_real

print('✅ 已配置使用真实数据库')

# ============================================================================
# 步骤3: 测试查询并对比
# ============================================================================

print('\n' + '='*70)
print('步骤3: Agent查询 vs 人工查询对比')
print('='*70)

# 重新注册工具（使用新函数）
get_registry()._tools.clear()
register_builtin_tools()

# 创建Agent
agent = AgentExecutor(
    model="nvidia/nemotron-3-nano-4b",
    max_iterations=5
)

# 测试用例
test_cases = [
    {
        "question": "查询所有年龄大于25的用户",
        "expected_sql": "SELECT * FROM users WHERE age > 25",
    },
    {
        "question": "统计用户总数",
        "expected_sql": "SELECT COUNT(*) FROM users",
    },
    {
        "question": "查询年龄最大的用户",
        "expected_sql": "SELECT * FROM users ORDER BY age DESC LIMIT 1",
    },
]

for i, case in enumerate(test_cases, 1):
    print(f'\n{"="*70}')
    print(f'测试 {i}: {case["question"]}')
    print(f'{"="*70}')

    # Agent查询
    print(f'\n【Agent查询】')
    agent_result = agent.run(case["question"], verbose=False)

    if agent_result["success"]:
        print(f'✅ Agent回答: {agent_result["answer"][:200]}...')
    else:
        print(f'❌ Agent失败: {agent_result.get("error")}')

    # 人工查询（预期SQL）
    print(f'\n【人工查询】')
    print(f'SQL: {case["expected_sql"]}')

    cursor = conn.execute(case["expected_sql"])
    cursor.row_factory = sqlite3.Row
    human_results = [dict(row) for row in cursor.fetchall()]

    print(f'结果: {human_results}')

    # 对比分析
    print(f'\n【对比分析】')

    # 提取Agent使用的SQL（从历史中）
    agent_sql = None
    for step in agent_result.get("history", []):
        if step.get("action") == "search_database":
            agent_sql = "已执行search_database"
            break

    if agent_sql:
        print(f'✅ Agent调用了工具')
    else:
        print(f'⚠️  Agent未调用工具（可能直接回答）')

    # 检查结果是否合理
    answer = agent_result.get("answer", "")
    if human_results:
        # 检查答案中是否包含预期数据
        first_result = human_results[0]
        found = any(str(v) in answer for v in first_result.values() if v is not None)
        if found:
            print(f'✅ Agent答案包含预期数据')
        else:
            print(f'⚠️  Agent答案可能不包含预期数据')
            print(f'   预期: {human_results}')
            print(f'   实际: {answer[:100]}...')

conn.close()

print('\n' + '='*70)
print('✅ 对比测试完成')
print('='*70)
