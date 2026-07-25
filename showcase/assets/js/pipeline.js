/* pipeline.js - 7层 Pipeline Mermaid 流程图 */

document.addEventListener('DOMContentLoaded', () => {
  const mermaidConfig = `
    %%{init: {
      'theme': 'base',
      'themeVariables': {
        'primaryColor': '#1a1a24',
        'primaryTextColor': '#e4e4ef',
        'primaryBorderColor': '#6366f1',
        'lineColor': '#4a4a5e',
        'secondaryColor': '#22222e',
        'tertiaryColor': '#0a0a0f',
        'fontFamily': 'Inter, sans-serif'
      }
    }}%%
    graph TD
      Start([用户查询]) --> A["1. Query Analysis<br/>查询分析"]
      A --> B["2. Retrieval<br/>混合检索 BM25+Vector+RRF"]
      B --> C["3. Document Grading<br/>文档评分"]
      C --> D{相关文档?}
      D -->|有相关文档| E["4. Answer Generation<br/>答案生成"]
      D -->|无相关文档| F["Query Rewrite<br/>查询改写"]
      F --> B
      D -->|重试耗尽| G["Web Search Fallback<br/>网络搜索兜底"]
      G --> E
      E --> H["5. Fact Checking<br/>事实校验"]
      H --> I{幻觉检测}
      I -->|检测通过| J["6. Conflict Detection<br/>冲突检测"]
      I -->|检测失败| K["Regenerate<br/>重新生成"]
      K --> E
      J --> L["7. Final Answer<br/>最终答案"]
      L --> End([输出结果])

      style Start fill:#06b6d4,stroke:#06b6d4,color:#0a0a0f
      style End fill:#10b981,stroke:#10b981,color:#0a0a0f
      style A fill:#1e293b,stroke:#6366f1,color:#e4e4ef
      style B fill:#1e293b,stroke:#6366f1,color:#e4e4ef
      style C fill:#1e293b,stroke:#8b5cf6,color:#e4e4ef
      style E fill:#1e293b,stroke:#f59e0b,color:#e4e4ef
      style H fill:#1e293b,stroke:#ec4899,color:#e4e4ef
      style J fill:#1e293b,stroke:#10b981,color:#e4e4ef
      style F fill:#22222e,stroke:#666680,color:#a0a0b8
      style G fill:#22222e,stroke:#666680,color:#a0a0b8
      style K fill:#22222e,stroke:#666680,color:#a0a0b8
      style D fill:#22222e,stroke:#8b5cf6,color:#e4e4ef
      style I fill:#22222e,stroke:#ec4899,color:#e4e4ef
  `;

  const container = document.getElementById('mermaid-pipeline');
  if (container) {
    container.innerHTML = `<div class="mermaid">${mermaidConfig}</div>`;
    if (window.mermaid) {
      window.mermaid.initialize({ startOnLoad: true, securityLevel: 'loose' });
      window.mermaid.run();
    }
  }
});
