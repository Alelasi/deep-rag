#!/usr/bin/env python3
"""扩展 multidim_cases：每个学科细化分支补 1 题（已存在 id 跳过）。"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CASES = ROOT / "multidim_cases.json"


def case(
    id: str,
    question: str,
    keywords: list[str],
    collection: str,
    category: str,
    discipline: str,
    subdiscipline: str,
    difficulty: str = "medium",
    allow_refuse: bool = True,
) -> dict:
    return {
        "id": id,
        "question": question,
        "expected_keywords": keywords,
        "collection": collection,
        "category": category,
        "discipline": discipline,
        "subdiscipline": subdiscipline,
        "difficulty": difficulty,
        "allow_refuse": allow_refuse,
    }


EXTRA = [
    # 心理学
    case("psych_personality", "大五人格模型包含哪五个维度？", ["开放", "尽责", "外向", "宜人", "神经"], "proj_psychology", "psychology", "心理学", "人格/大五人格"),
    case("psych_memory", "工作记忆与长时记忆有何区别？", ["工作记忆", "长时", "容量"], "proj_psychology", "psychology", "心理学", "认知/记忆"),
    case("psych_emotion", "詹姆斯-朗格情绪理论的基本观点是什么？", ["生理", "情绪", "反馈"], "proj_psychology", "psychology", "心理学", "情绪心理学", "hard"),
    case("psych_motivation", "马斯洛需求层次从低到高大致有哪些层？", ["生理", "安全", "归属", "尊重", "自我实现"], "proj_psychology", "psychology", "心理学", "动机理论", "easy"),
    case("psych_psychometrics", "信度与效度分别衡量测验的什么？", ["信度", "效度", "可靠", "有效"], "proj_psychology", "psychology", "心理学", "心理测量"),
    # 计算机/AI
    case("cs_prompt", "什么是提示工程prompt engineering？", ["提示", "prompt", "指令"], "proj_work", "cs_ai", "计算机/AI", "提示工程", "easy"),
    case("cs_embedding", "文本embedding向量通常用来做什么？", ["语义", "向量", "相似"], "proj_work", "cs_ai", "计算机/AI", "Embedding", "easy"),
    case("cs_vectordb", "向量数据库相对传统数据库适合什么场景？", ["相似", "检索", "向量"], "proj_work", "cs_ai", "计算机/AI", "向量数据库"),
    case("cs_crag", "Corrective RAG的主要思路是什么？", ["纠错", "检索", "相关"], "proj_work", "cs_ai", "计算机/AI", "Corrective RAG"),
    case("cs_tooluse", "大模型工具调用tool calling解决什么问题？", ["工具", "函数", "外部"], "proj_work", "cs_ai", "计算机/AI", "工具调用"),
    case("cs_context", "长上下文对RAG系统意味着什么取舍？", ["上下文", "窗口", "成本"], "proj_work", "cs_ai", "计算机/AI", "长上下文", "hard"),
    # 机器学习
    case("ml_biasvar", "偏差-方差权衡描述了什么？", ["偏差", "方差", "过拟合"], "proj_thesis", "ml", "机器学习", "偏差方差"),
    case("ml_reg", "正则化L1与L2的主要差别是什么？", ["稀疏", "L1", "L2"], "proj_thesis", "ml", "机器学习", "正则化"),
    case("ml_svm", "支持向量机SVM的核心思想是什么？", ["间隔", "超平面", "支持向量"], "proj_thesis", "ml", "机器学习", "SVM"),
    case("ml_cluster", "K-means聚类算法的基本步骤是什么？", ["聚类", "中心", "迭代"], "proj_thesis", "ml", "机器学习", "聚类", "easy"),
    case("ml_dim", "主成分分析PCA用于什么目的？", ["降维", "主成分", "方差"], "proj_thesis", "ml", "机器学习", "降维/PCA"),
    case("ml_dl_backprop", "反向传播算法解决什么问题？", ["梯度", "反向", "参数"], "proj_thesis", "ml", "机器学习", "反向传播"),
    # 经济学
    case("econ_elastic", "需求价格弹性如何定义？", ["弹性", "价格", "需求"], "proj_social", "economics", "经济学", "弹性理论", "easy"),
    case("econ_market", "完全竞争市场的基本特征有哪些？", ["众多", "同质", "信息"], "proj_social", "economics", "经济学", "市场结构"),
    case("econ_externality", "外部性是什么，举例说明？", ["外部", "成本", "社会"], "proj_social", "economics", "经济学", "外部性"),
    case("econ_trade", "比较优势理论说明国际贸易的什么道理？", ["比较", "优势", "贸易"], "proj_social", "economics", "经济学", "国际贸易"),
    case("econ_fiscal", "财政政策主要通过什么工具影响经济？", ["税收", "支出", "政府"], "proj_social", "economics", "经济学", "财政政策", "easy"),
    # 社会学
    case("soc_role", "社会角色与社会地位的关系是什么？", ["角色", "地位", "期望"], "proj_social", "sociology", "社会学", "角色理论"),
    case("soc_culture", "文化相对主义主张什么？", ["文化", "相对", "评价"], "proj_social", "sociology", "社会学", "文化研究"),
    case("soc_gender", "社会性别gender与生理性别sex的区别？", ["社会", "性别", "建构"], "proj_social", "sociology", "社会学", "性别研究"),
    case("soc_urban", "城市化可能带来哪些社会问题？", ["城市", "流动", "空间"], "proj_social", "sociology", "社会学", "城市社会学"),
    # 政治学
    case("pol_state", "韦伯如何定义国家？", ["暴力", "垄断", "合法性"], "proj_social", "politics", "政治学", "国家理论", "hard"),
    case("pol_party", "政党在代议民主中的主要功能是什么？", ["政党", "选举", "利益"], "proj_social", "politics", "政治学", "政党政治"),
    case("pol_ideology", "意识形态大致指什么？", ["观念", "政治", "体系"], "proj_social", "politics", "政治学", "意识形态", "easy"),
    # 法学
    case("law_rights", "基本权利与人权有何联系？", ["权利", "宪法", "保障"], "proj_social", "law", "法学", "基本权利"),
    case("law_admin", "行政法中的合法行政原则要求什么？", ["合法", "行政", "职权"], "proj_social", "law", "法学", "行政法"),
    case("law_procedure", "程序正义为什么重要？", ["程序", "公正", "过程"], "proj_social", "law", "法学", "程序法"),
    # 历史
    case("hist_ancient", "春秋战国时期社会变动的主要特征是什么？", ["诸侯", "变法", "礼崩"], "proj_social", "history", "历史学", "中国古代史"),
    case("hist_reform", "改革开放的起点通常指哪一历史事件？", ["十一届三中全会", "改革", "开放"], "proj_social", "history", "历史学", "改革开放史", "easy"),
    case("hist_global", "全球化历史进程中的关键动力有哪些？", ["贸易", "技术", "流动"], "proj_social", "history", "历史学", "全球史", "hard"),
    # 哲学
    case("phil_exist", "存在主义强调人的什么特征？", ["自由", "存在", "选择"], "proj_social", "philosophy", "哲学", "存在主义"),
    case("phil_epistemology", "经验论与唯理论的主要分歧是什么？", ["经验", "理性", "知识"], "proj_social", "philosophy", "哲学", "知识论流派", "hard"),
    case("phil_mind", "心身问题讨论的是什么？", ["心灵", "身体", "意识"], "proj_social", "philosophy", "哲学", "心灵哲学", "hard"),
    # 教育
    case("edu_curriculum", "课程目标、内容与评价应如何对齐？", ["目标", "内容", "评价"], "proj_social", "education", "教育学", "课程论"),
    case("edu_motivation", "内在动机与外在动机有何区别？", ["内在", "外在", "动机"], "proj_social", "education", "教育学", "学习动机", "easy"),
    case("edu_inclusive", "全纳教育主张什么？", ["全纳", "特殊", "融合"], "proj_social", "education", "教育学", "全纳教育"),
    # 语言学
    case("ling_semantic", "语义学主要研究语言的什么层面？", ["意义", "语义", "指称"], "proj_social", "linguistics", "语言学", "语义学", "easy"),
    case("ling_socioling", "社会语言学关注语言与什么因素的关系？", ["社会", "阶层", "变体"], "proj_social", "linguistics", "语言学", "社会语言学"),
    case("ling_acquisition", "第二语言习得中的关键期假说指什么？", ["关键期", "习得", "年龄"], "proj_social", "linguistics", "语言学", "二语习得", "hard"),
    # 金融
    case("fin_bond", "债券价格与利率大致呈什么关系？", ["利率", "价格", "反向"], "proj_work", "finance", "金融", "债券/利率"),
    case("fin_option", "看涨期权的买方最大损失是什么？", ["权利金", "期权", "损失"], "proj_work", "finance", "金融", "衍生品/期权"),
    case("fin_factor", "多因子选股模型试图解释什么？", ["因子", "收益", "风险"], "proj_work", "finance", "金融", "多因子", "hard"),
    case("fin_macro", "货币政策宽松通常如何影响资产价格？", ["利率", "流动性", "资产"], "proj_work", "finance", "金融", "宏观金融"),
    # 数学
    case("math_prob", "条件概率与全概率公式解决什么问题？", ["条件", "概率", "划分"], "proj_thesis", "math", "数学", "条件概率"),
    case("math_matrix", "矩阵特征值在应用中有什么意义？", ["特征", "向量", "变换"], "proj_thesis", "math", "数学", "特征值", "hard"),
    case("math_info", "信息熵衡量的是什么？", ["熵", "不确定", "信息"], "proj_thesis", "math", "数学", "信息论"),
    case("math_opt", "凸优化问题为何相对好解？", ["凸", "局部", "全局"], "proj_thesis", "math", "数学", "凸优化", "hard"),
    # 物理
    case("phys_rel", "狭义相对论的两个基本假设是什么？", ["光速", "相对性", "惯性"], "proj_social", "physics", "物理学", "相对论", "hard"),
    case("phys_wave", "波粒二象性指什么？", ["波", "粒子", "量子"], "proj_social", "physics", "物理学", "波动/量子", "medium"),
    case("phys_thermo2", "热力学第二定律的常见表述是什么？", ["熵", "自发", "不可逆"], "proj_social", "physics", "物理学", "热力学第二定律"),
    # 生物
    case("bio_gene", "基因表达的中心法则是什么？", ["DNA", "RNA", "蛋白质"], "proj_social", "biology", "生物学", "中心法则", "easy"),
    case("bio_immune", "先天免疫与适应性免疫有何不同？", ["先天", "适应", "抗体"], "proj_social", "biology", "生物学", "免疫基础"),
    case("bio_neuron", "神经元如何传递信号？", ["电位", "突触", "神经"], "proj_social", "biology", "生物学", "神经生物学"),
    # 化学
    case("chem_organic", "有机化学中官能团决定什么？", ["官能团", "性质", "反应"], "proj_social", "chemistry", "化学", "有机化学", "easy"),
    case("chem_thermo", "化学热力学关注反应的什么问题？", ["能量", "自发", "焓"], "proj_social", "chemistry", "化学", "化学热力学"),
    case("chem_kinetics", "反应速率受哪些因素影响？", ["温度", "浓度", "催化剂"], "proj_social", "chemistry", "化学", "化学动力学", "easy"),
    # 文学
    case("lit_realism", "现实主义文学的主要特征是什么？", ["真实", "典型", "社会"], "proj_social", "literature", "文学", "现实主义"),
    case("lit_modern", "现代主义文学常见的形式实验有哪些？", ["意识流", "碎片", "现代"], "proj_social", "literature", "文学", "现代主义", "hard"),
    case("lit_rhetoric", "比喻与象征有何区别？", ["比喻", "象征", "修辞"], "proj_social", "literature", "文学", "修辞学"),
    # 管理
    case("mgmt_lean", "精益管理的核心目标是什么？", ["浪费", "价值", "流程"], "proj_work", "management", "管理学", "精益管理"),
    case("mgmt_leader", "变革型领导与交易型领导有何不同？", ["变革", "交易", "激励"], "proj_work", "management", "管理学", "领导力"),
    case("mgmt_project", "项目管理中的铁三角通常指什么？", ["范围", "时间", "成本"], "proj_work", "management", "管理学", "项目管理", "easy"),
    # 地理
    case("geo_gis", "GIS地理信息系统的基本功能是什么？", ["空间", "数据", "分析"], "proj_social", "geography", "地理学", "GIS"),
    case("geo_resource", "可再生与不可再生资源如何划分？", ["可再生", "资源", "枯竭"], "proj_social", "geography", "地理学", "资源地理", "easy"),
    case("geo_region", "区域地理综合分析通常考虑哪些要素？", ["自然", "人文", "区域"], "proj_social", "geography", "地理学", "区域地理"),
    # 医学
    case("med_pharma", "药代动力学ADME指哪些过程？", ["吸收", "分布", "代谢", "排泄"], "proj_social", "medicine", "医学", "药代动力学", "hard"),
    case("med_epid", "流行病学中的发病率与患病率有何区别？", ["发病", "患病", "新病例"], "proj_social", "medicine", "医学", "流行病学"),
    case("med_public", "公共卫生三级预防分别对应什么？", ["一级", "二级", "三级", "预防"], "proj_social", "medicine", "医学", "公共卫生"),
    # 艺术/音乐
    case("art_persp", "透视法在绘画中的作用是什么？", ["透视", "空间", "深度"], "proj_social", "art", "艺术", "透视", "easy"),
    case("art_style", "艺术风格史为什么要分期？", ["风格", "时代", "流派"], "proj_social", "art", "艺术", "艺术史分期"),
    case("music_form", "奏鸣曲式的基本结构是什么？", ["呈示", "展开", "再现"], "proj_social", "music", "音乐", "曲式", "hard"),
    case("music_rhythm", "节奏与节拍有何区别？", ["节奏", "节拍", "时值"], "proj_social", "music", "音乐", "节奏节拍", "easy"),
    # 交叉
    case("cross_observ", "AI系统可观测性通常包括哪些信号？", ["日志", "指标", "追踪"], "proj_work", "cross", "交叉/工程", "可观测性"),
    case("cross_safety", "大模型应用中的安全护栏可以做什么？", ["安全", "过滤", "策略"], "proj_work", "cross", "交叉/工程", "AI安全"),
    case("cross_data", "数据飞轮对AI产品意味着什么？", ["数据", "反馈", "迭代"], "proj_work", "cross", "交叉/工程", "数据飞轮", "hard"),
    # 新增学科
    case("stat_hypothesis", "假设检验中的p值如何解释？", ["p值", "原假设", "显著"], "proj_thesis", "stats", "统计学", "假设检验"),
    case("stat_ci", "置信区间表达的是什么不确定性？", ["置信", "区间", "参数"], "proj_thesis", "stats", "统计学", "置信区间"),
    case("stat_sampling", "随机抽样为何重要？", ["随机", "样本", "偏差"], "proj_thesis", "stats", "统计学", "抽样", "easy"),
    case("comm_media", "议程设置理论认为媒体如何影响公众？", ["议程", "媒体", "关注"], "proj_social", "comms", "传播学", "议程设置"),
    case("comm_semiotics", "传播学中的编码与解码指什么？", ["编码", "解码", "意义"], "proj_social", "comms", "传播学", "编码解码"),
    case("anthro_culture", "文化人类学如何理解文化？", ["文化", "习得", "共享"], "proj_social", "anthro", "人类学", "文化概念"),
    case("anthro_field", "田野调查ethnography的核心方法是什么？", ["参与观察", "田野", "民族志"], "proj_social", "anthro", "人类学", "田野方法", "hard"),
    case("env_sustain", "可持续发展的三重底线通常指什么？", ["经济", "社会", "环境"], "proj_social", "env", "环境科学", "可持续发展", "easy"),
    case("env_carbon", "碳中和与碳达峰分别指什么？", ["碳", "达峰", "中和"], "proj_social", "env", "环境科学", "碳中和"),
    case("cs_os", "操作系统的进程与线程有何区别？", ["进程", "线程", "资源"], "proj_work", "cs_sys", "计算机系统", "进程线程"),
    case("cs_net", "TCP与UDP的主要差别是什么？", ["可靠", "连接", "UDP"], "proj_work", "cs_sys", "计算机系统", "计算机网络", "easy"),
    case("cs_db", "数据库事务ACID分别代表什么？", ["原子", "一致", "隔离", "持久"], "proj_work", "cs_sys", "计算机系统", "数据库事务"),
]


def main() -> None:
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    by_id = {c["id"]: c for c in cases}
    added = 0
    for e in EXTRA:
        if e["id"] in by_id:
            continue
        cases.append(e)
        by_id[e["id"]] = e
        added += 1

    for c in cases:
        if c.get("expect_refuse"):
            c["web_fallback"] = False
        c.setdefault("allow_refuse", True)
        c.setdefault("discipline", c.get("category") or "未标注")
        c.setdefault("subdiscipline", c.get("category") or "通用")

    CASES.write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    by = defaultdict(list)
    for c in cases:
        by[c.get("discipline")].append(c.get("subdiscipline"))
    print(f"total={len(cases)} added={added} disciplines={len(by)}")
    for d, v in sorted(by.items(), key=lambda x: (-len(x[1]), x[0])):
        print(f"  {d}: {len(v)}")
        for s in v:
            print(f"    - {s}")


if __name__ == "__main__":
    main()
