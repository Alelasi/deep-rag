/* routing.js - D3.js Agentic 工具路由力导向图 */

document.addEventListener('DOMContentLoaded', () => {
  const container = document.getElementById('routing-graph');
  if (!container || !window.d3) return;

  const width = container.clientWidth;
  const height = 450;

  const svg = d3.select('#routing-graph')
    .append('svg')
    .attr('width', width)
    .attr('height', height);

  // 定义箭头和渐变
  const defs = svg.append('defs');
  const colors = { blue: '#6366f1', purple: '#8b5cf6', pink: '#ec4899', cyan: '#06b6d4', green: '#10b981', orange: '#f59e0b', muted: '#4a4a5e' };

  Object.entries(colors).forEach(([name, color]) => {
    const grad = defs.append('linearGradient')
      .attr('id', `grad-${name}`)
      .attr('x1', '0%').attr('y1', '0%')
      .attr('x2', '0%').attr('y2', '100%');
    grad.append('stop').attr('offset', '0%').attr('stop-color', color).attr('stop-opacity', 0.8);
    grad.append('stop').attr('offset', '100%').attr('stop-color', color).attr('stop-opacity', 0.3);
  });

  // 节点数据
  const nodes = [
    { id: 'router', label: 'Agent Router\n智能路由', type: 'router', x: width / 2, y: 80 },
    { id: 'exact', label: 'Exact Match\n精确匹配', type: 'tool', desc: '版本号/订单号', x: width * 0.15, y: 250 },
    { id: 'vector', label: 'Vector Search\n向量检索', type: 'tool', desc: '语义检索', x: width * 0.38, y: 250 },
    { id: 'graph', label: 'Graph Search\n图检索', type: 'tool', desc: '依赖/对比关系', x: width * 0.62, y: 250 },
    { id: 'web', label: 'Web Search\n网络搜索', type: 'tool', desc: '实时信息', x: width * 0.85, y: 250 },
    { id: 'aggregate', label: 'Result Aggregation\n结果汇总', type: 'aggregate', x: width / 2, y: 400 }
  ];

  const links = [
    { source: 'router', target: 'exact', active: false },
    { source: 'router', target: 'vector', active: false },
    { source: 'router', target: 'graph', active: false },
    { source: 'router', target: 'web', active: false },
    { source: 'exact', target: 'aggregate', active: true },
    { source: 'vector', target: 'aggregate', active: true },
    { source: 'graph', target: 'aggregate', active: true },
    { source: 'web', target: 'aggregate', active: true }
  ];

  // 路由场景
  const scenarios = {
    'concept': { tools: ['vector'], label: '概念查询' },
    'version': { tools: ['exact', 'web'], label: '版本+实时' },
    'compare': { tools: ['vector', 'graph'], label: '对比查询' },
    'complex': { tools: ['vector', 'web', 'graph'], label: '复杂多跳' }
  };

  // 绘制连线
  const linkGroup = svg.append('g').attr('class', 'links');
  const linkElements = linkGroup.selectAll('line')
    .data(links)
    .enter().append('line')
    .attr('stroke', d => d.active ? colors.muted : colors.muted)
    .attr('stroke-width', 2)
    .attr('stroke-dasharray', d => d.active ? 'none' : '4 4')
    .attr('opacity', 0.6);

  // 绘制节点
  const nodeGroup = svg.append('g').attr('class', 'nodes');
  const nodeElements = nodeGroup.selectAll('g.node')
    .data(nodes)
    .enter().append('g')
    .attr('class', 'node')
    .attr('transform', d => `translate(${d.x}, ${d.y})`);

  // 节点圆形
  nodeElements.append('circle')
    .attr('r', d => d.type === 'router' ? 45 : d.type === 'aggregate' ? 40 : 38)
    .attr('fill', d => {
      if (d.type === 'router') return 'url(#grad-blue)';
      if (d.type === 'aggregate') return 'url(#grad-green)';
      return 'url(#grad-purple)';
    })
    .attr('stroke', d => {
      if (d.type === 'router') return colors.blue;
      if (d.type === 'aggregate') return colors.green;
      return colors.purple;
    })
    .attr('stroke-width', 2)
    .attr('opacity', 0.9);

  // 节点文字
  nodeElements.append('text')
    .attr('text-anchor', 'middle')
    .attr('dy', d => d.desc ? '-0.3em' : '0.3em')
    .attr('fill', '#e4e4ef')
    .attr('font-size', d => d.type === 'router' ? '11px' : '10px')
    .attr('font-weight', '600')
    .each(function(d) {
      const lines = d.label.split('\n');
      const text = d3.select(this);
      if (lines.length > 1) {
        text.text('');
        lines.forEach((line, i) => {
          text.append('tspan')
            .attr('x', 0)
            .attr('dy', i === 0 ? `-${(lines.length - 1) * 0.6}em` : '1.2em')
            .text(line);
        });
      } else {
        text.text(lines[0]);
      }
    });

  // 节点描述
  nodeElements.filter(d => d.desc)
    .append('text')
    .attr('text-anchor', 'middle')
    .attr('dy', '3em')
    .attr('fill', '#666680')
    .attr('font-size', '9px')
    .text(d => d.desc);

  // 高亮路由路径
  function highlightRoute(scenarioKey) {
    const scenario = scenarios[scenarioKey];
    if (!scenario) return;

    // 重置所有
    linkElements
      .attr('stroke', colors.muted)
      .attr('stroke-width', 2)
      .attr('opacity', 0.3)
      .attr('stroke-dasharray', '4 4');

    nodeElements.select('circle')
      .attr('opacity', 0.3);

    // 高亮路由器到工具的路径
    scenario.tools.forEach(toolId => {
      linkElements
        .filter(d => (d.source.id || d.source) === 'router' && (d.target.id || d.target) === toolId)
        .attr('stroke', colors.cyan)
        .attr('stroke-width', 3)
        .attr('opacity', 1)
        .attr('stroke-dasharray', 'none');

      // 高亮工具到汇总
      linkElements
        .filter(d => (d.source.id || d.source) === toolId && (d.target.id || d.target) === 'aggregate')
        .attr('stroke', colors.green)
        .attr('stroke-width', 3)
        .attr('opacity', 1)
        .attr('stroke-dasharray', 'none');

      // 高亮工具节点
      nodeElements
        .filter(d => d.id === toolId)
        .select('circle')
        .attr('opacity', 1)
        .attr('stroke', colors.cyan)
        .attr('stroke-width', 3);
    });

    // 始终高亮路由器和汇总
    nodeElements
      .filter(d => d.id === 'router' || d.id === 'aggregate')
      .select('circle')
      .attr('opacity', 1);
  }

  // 按钮事件
  document.querySelectorAll('.routing-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.routing-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      highlightRoute(btn.dataset.scenario);
    });
  });

  // 默认高亮概念查询
  highlightRoute('concept');
});
