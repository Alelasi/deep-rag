# DeepRAG Showcase

技术作品集展示网站，部署到 GitHub Pages。

## 目录结构

```
showcase/
├── index.html          # 单页入口（6 区块）
├── assets/
│   ├── css/style.css   # 深色科技风主题
│   ├── js/
│   │   ├── main.js     # 滚动动画 + 导航 + 打字机
│   │   ├── pipeline.js # 7层Pipeline Mermaid流程图
│   │   ├── routing.js  # D3.js Agentic工具路由力导向图
│   │   └── charts.js   # Chart.js 性能数据图表
│   └── data/
│       ├── metrics.json     # 真实指标数据（含来源标注）
│       └── vector-db.json   # 向量数据库对比数据
└── README.md
```

## 页面区块

1. **Hero** - 标题 + 打字机动画 + CTA 按钮
2. **项目背景** - 痛点叙事 + 传统RAG vs Agentic RAG 对比
3. **7层Pipeline** - Mermaid.js 流程图（含 Corrective RAG + Self-RAG 循环）
4. **Agentic路由** - D3.js 力导向图（4种查询场景动态高亮）
5. **技术栈** - 12个核心组件图标网格
6. **性能数据** - 4张 Chart.js 图表 + 评测资产 + 版本演进时间线

## 数据来源

所有数字可追溯至项目源文件：
- `README.md` - 检索准确率、版本演进
- `docs/向量数据库性能对比报告.md` - 向量DB性能数据
- `docs/审计真相表_P0.md` - 评测资产规模
- `evaluation_reports/ragas_report_20260521_140012.md` - RAGAS得分

## 本地预览

```bash
cd showcase
python -m http.server 8080
# 访问 http://localhost:8080
```

## 部署

GitHub Actions 自动部署（`.github/workflows/pages.yml`），push 到 main 时触发。

访问: https://alelasi.github.io/deep-rag/
