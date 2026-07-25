"""
Agent查询质量评估 - 准确率、召回率、精确问题 vs 含糊问题

评估维度：
1. 准确率（Precision）：Agent返回的结果中，有多少是正确的
2. 召回率（Recall）：所有正确结果中，Agent返回了多少
3. F1分数：准确率和召回率的调和平均
4. 问题类型：精确问题 vs 含糊问题的效果对比
"""

import sqlite3
import sys
sys.path.insert(0, '.')

from src.tools import register_builtin_tools, AgentExecutor, get_registry
from src.tools import builtin_tools

# ============================================================================
# 步骤1: 准备数据库和配置
# ============================================================================

print('='*70)
print('Agent查询质量评估 - Precision & Recall & 问题类型对比')
print('='*70)

# 创建数据库
conn = sqlite3.connect('test_database.db')
cursor = conn.cursor()

# 确保有数据
cursor.execute('SELECT COUNT(*) FROM users')
count = cursor.fetchone()[0]
print(f'\n✅ 数据库连接成功，users表有 {count} 条记录')

# 显示所有数据
cursor.execute('SELECT * FROM users ORDER BY id')
all_users = cursor.fetchall()
print(f'\n数据概览:')
for user in all_users:
    print(f'  ID={user[0]}, 姓名={user[1]}, 年龄={user[2]}')

# 配置Agent使用真实数据库
def search_database_real(sql_query: str):
    import re
    query_lower = sql_query.lower().strip()

    if not query_lower.startswith("select"):
        raise ValueError("❌ 只允许SELECT查询")

    dangerous = ["insert", "update", "delete", "drop", "truncate", "alter",
                 "create", "replace", "merge", "exec", "execute", "call"]
    for kw in dangerous:
        if re.search(rf'\b{kw}\b', query_lower):
            raise ValueError(f"❌ 禁止使用 {kw.upper()} 操作")

    if "limit" not in query_lower:
        sql_query = sql_query.rstrip(";") + " LIMIT 100"

    conn_temp = sqlite3.connect('test_database.db', timeout=5.0)
    cursor_temp = conn_temp.execute(sql_query)
    results = [{"id": row[0], "name": row[1], "age": row[2], "email": row[3]}
               for row in cursor_temp.fetchall()]
    conn_temp.close()

    return results

builtin_tools.search_database = search_database_real
get_registry()._tools.clear()
register_builtin_tools()

agent = AgentExecutor(model="nvidia/nemotron-3-nano-4b", max_iterations=5)

# ============================================================================
# 步骤2: 定义测试用例（精确问题 vs 含糊问题）
# ============================================================================

test_cases = [
    # 精确问题（具体数字、明确条件）
    {
        "id": 1,
        "type": "精确",
        "question": "查询所有年龄大于25的用户",
        "ground_truth_sql": "SELECT * FROM users WHERE age > 25",
        "ground_truth_ids": [2, 3, 4],  # 李四(28), 王五(26), 赵六(32)
    },
    {
        "id": 2,
        "type": "精确",
        "question": "查询年龄小于等于23的用户",
        "ground_truth_sql": "SELECT * FROM users WHERE age <= 23",
        "ground_truth_ids": [1, 5],  # 张三(23), 钱七(19)
    },
    {
        "id": 3,
        "type": "精确",
        "question": "查询年龄在20到30之间的用户",
        "ground_truth_sql": "SELECT * FROM users WHERE age BETWEEN 20 AND 30",
        "ground_truth_ids": [1, 2, 3],  # 张三(23), 李四(28), 王五(26)
    },

    # 含糊问题（模糊词汇、相对概念）
    {
        "id": 4,
        "type": "含糊",
        "question": "查询年龄比较大的用户",  # "比较大"是模糊词
        "ground_truth_sql": "SELECT * FROM users WHERE age > 25",  # 假设>25算"比较大"
        "ground_truth_ids": [2, 3, 4],
    },
    {
        "id": 5,
        "type": "含糊",
        "question": "查询年轻的用户",  # "年轻"是相对概念
        "ground_truth_sql": "SELECT * FROM users WHERE age < 25",  # 假设<25算"年轻"
        "ground_truth_ids": [1, 5],  # 张三(23), 钱七(19)
    },
    {
        "id": 6,
        "type": "含糊",
        "question": "找出年龄中等的用户",  # "中等"更模糊
        "ground_truth_sql": "SELECT * FROM users WHERE age BETWEEN 23 AND 28",
        "ground_truth_ids": [1, 2, 3],  # 张三(23), 李四(28), 王五(26)
    },
]

# ============================================================================
# 步骤3: 执行测试并计算指标
# ============================================================================

print('\n' + '='*70)
print('开始测试')
print('='*70)

results = []

for case in test_cases:
    print(f'\n{"="*70}')
    print(f'测试 {case["id"]} ({case["type"]}): {case["question"]}')
    print(f'{"="*70}')

    # 获取标准答案（Ground Truth）
    cursor.execute(case["ground_truth_sql"])
    ground_truth_ids = set(case["ground_truth_ids"])

    print(f'\n【标准答案】')
    print(f'SQL: {case["ground_truth_sql"]}')
    print(f'结果ID: {sorted(ground_truth_ids)}')
    print(f'数量: {len(ground_truth_ids)}')

    # Agent查询
    print(f'\n【Agent查询】')
    agent_result = agent.run(case["question"], verbose=False)

    # 提取Agent调用工具时的实际结果
    agent_ids = set()
    for step in agent_result.get("history", []):
        observation = step.get("observation")
        if observation and isinstance(observation, str) and observation.startswith('['):
            try:
                import json
                data = json.loads(observation)
                agent_ids = set([item['id'] for item in data if 'id' in item])
                break
            except:
                pass

    print(f'Agent返回ID: {sorted(agent_ids) if agent_ids else "未提取到"}')
    print(f'数量: {len(agent_ids)}')

    # 计算准确率和召回率
    if len(agent_ids) == 0:
        print(f'\n⚠️  Agent未返回结果（可能直接拒绝或理解错误）')
        results.append({
            "case_id": case["id"],
            "case_type": case["type"],
            "question": case["question"],
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "tp": 0,
            "fp": 0,
            "fn": len(ground_truth_ids),
        })
        continue

    # TP/FP/FN
    true_positive = agent_ids & ground_truth_ids
    false_positive = agent_ids - ground_truth_ids
    false_negative = ground_truth_ids - agent_ids

    # 计算指标
    precision = len(true_positive) / len(agent_ids) if len(agent_ids) > 0 else 0.0
    recall = len(true_positive) / len(ground_truth_ids) if len(ground_truth_ids) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f'\n【质量指标】')
    print(f'TP (正确返回): {sorted(true_positive)} ({len(true_positive)}个)')
    print(f'FP (错误返回): {sorted(false_positive)} ({len(false_positive)}个)')
    print(f'FN (遗漏): {sorted(false_negative)} ({len(false_negative)}个)')
    print(f'')
    print(f'准确率 (Precision): {precision:.2%}')
    print(f'召回率 (Recall):    {recall:.2%}')
    print(f'F1分数:             {f1:.2%}')

    # 判断质量
    if precision == 1.0 and recall == 1.0:
        print(f'评价: ✅ 完美')
    elif precision >= 0.9 and recall >= 0.9:
        print(f'评价: ✅ 优秀')
    elif f1 >= 0.8:
        print(f'评价: ⚠️  良好')
    else:
        print(f'评价: ❌ 需改进')

    results.append({
        "case_id": case["id"],
        "case_type": case["type"],
        "question": case["question"],
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": len(true_positive),
        "fp": len(false_positive),
        "fn": len(false_negative),
    })

# ============================================================================
# 步骤4: 分类汇总统计
# ============================================================================

print('\n' + '='*70)
print('分类汇总统计')
print('='*70)

# 按类型分组
precise_results = [r for r in results if r["case_type"] == "精确"]
vague_results = [r for r in results if r["case_type"] == "含糊"]

def calculate_avg(results_list):
    if len(results_list) == 0:
        return 0.0, 0.0, 0.0
    avg_p = sum(r["precision"] for r in results_list) / len(results_list)
    avg_r = sum(r["recall"] for r in results_list) / len(results_list)
    avg_f1 = sum(r["f1"] for r in results_list) / len(results_list)
    return avg_p, avg_r, avg_f1

# 精确问题统计
print(f'\n【精确问题】({len(precise_results)}个)')
if len(precise_results) > 0:
    p, r, f = calculate_avg(precise_results)
    print(f'平均准确率: {p:.2%}')
    print(f'平均召回率: {r:.2%}')
    print(f'平均F1分数: {f:.2%}')
else:
    print(f'无有效结果')

# 含糊问题统计
print(f'\n【含糊问题】({len(vague_results)}个)')
if len(vague_results) > 0:
    p, r, f = calculate_avg(vague_results)
    print(f'平均准确率: {p:.2%}')
    print(f'平均召回率: {r:.2%}')
    print(f'平均F1分数: {f:.2%}')
else:
    print(f'无有效结果')

# 对比分析
print(f'\n【对比分析】')
if len(precise_results) > 0 and len(vague_results) > 0:
    p_prec, r_prec, f_prec = calculate_avg(precise_results)
    p_vague, r_vague, f_vague = calculate_avg(vague_results)

    diff_p = p_prec - p_vague
    diff_r = r_prec - r_vague
    diff_f = f_prec - f_vague

    print(f'准确率差异: {diff_p:+.2%} ({"精确更好" if diff_p > 0 else "含糊更好" if diff_p < 0 else "相同"})')
    print(f'召回率差异: {diff_r:+.2%} ({"精确更好" if diff_r > 0 else "含糊更好" if diff_r < 0 else "相同"})')
    print(f'F1分数差异: {diff_f:+.2%} ({"精确更好" if diff_f > 0 else "含糊更好" if diff_f < 0 else "相同"})')

    # 结论
    print(f'\n【结论】')
    if abs(diff_f) < 0.05:
        print(f'✅ 两种问题类型的效果相近（差异<5%）')
    elif diff_f > 0.1:
        print(f'⚠️  精确问题效果明显更好（F1高出{diff_f:.1%}）')
    elif diff_f < -0.1:
        print(f'⚠️  含糊问题效果意外更好（F1高出{-diff_f:.1%}），需检查标准答案设定')

# 详细表格
print(f'\n【详细结果表】')
print(f'{"ID":<4} {"类型":<6} {"Precision":<12} {"Recall":<12} {"F1":<12} {"TP":<4} {"FP":<4} {"FN":<4}')
print('-'*70)
for r in results:
    print(f'{r["case_id"]:<4} {r["case_type"]:<6} {r["precision"]:<12.2%} {r["recall"]:<12.2%} '
          f'{r["f1"]:<12.2%} {r["tp"]:<4} {r["fp"]:<4} {r["fn"]:<4}')

# 总体评级
print(f'\n【总体评级】')
all_f1 = [r["f1"] for r in results]
avg_f1_all = sum(all_f1) / len(all_f1) if len(all_f1) > 0 else 0.0

if avg_f1_all >= 0.95:
    print(f'⭐⭐⭐⭐⭐ 优秀 (F1≥95%)')
elif avg_f1_all >= 0.90:
    print(f'⭐⭐⭐⭐ 良好 (F1≥90%)')
elif avg_f1_all >= 0.80:
    print(f'⭐⭐⭐ 合格 (F1≥80%)')
else:
    print(f'⭐⭐ 需改进 (F1<80%)')

print(f'总体F1分数: {avg_f1_all:.2%}')

conn.close()

print('\n' + '='*70)
print('✅ 评估完成')
print('='*70)
