/* charts.js - Chart.js 性能数据图表 */

document.addEventListener('DOMContentLoaded', () => {
  if (!window.Chart) return;

  // 全局 Chart.js 暗色主题
  Chart.defaults.color = '#a0a0b8';
  Chart.defaults.borderColor = '#2a2a3a';
  Chart.defaults.font.family = 'Inter, sans-serif';

  // 加载数据并渲染图表
  Promise.all([
    fetch('assets/data/metrics.json').then(r => r.json()),
    fetch('assets/data/vector-db.json').then(r => r.json())
  ]).then(([metrics, vectorDb]) => {
    renderRetrievalChart(metrics);
    renderVectorDbChart(vectorDb);
    renderVersionChart(metrics);
    renderRagasChart(metrics);
  }).catch(err => console.error('数据加载失败:', err));

  // === 图1: 检索准确率提升 ===
  function renderRetrievalChart(metrics) {
    const ctx = document.getElementById('chart-retrieval');
    if (!ctx) return;
    const data = metrics.retrieval_accuracy.data;

    new Chart(ctx, {
      type: 'bar',
      data: {
        labels: data.map(d => d.stage),
        datasets: [{
          label: 'Top-5 准确率 (%)',
          data: data.map(d => d.accuracy),
          backgroundColor: data.map(d => d.color + '80'),
          borderColor: data.map(d => d.color),
          borderWidth: 2,
          borderRadius: 8,
          barThickness: 50
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: '#1a1a24',
            borderColor: '#3a3a4e',
            borderWidth: 1,
            titleColor: '#e4e4ef',
            bodyColor: '#a0a0b8',
            callbacks: {
              label: ctx => `准确率: ${ctx.parsed.y}%`
            }
          }
        },
        scales: {
          y: {
            beginAtZero: true,
            max: 100,
            grid: { color: '#2a2a3a' },
            ticks: { callback: v => v + '%' }
          },
          x: {
            grid: { display: false }
          }
        }
      }
    });
  }

  // === 图2: 向量数据库写入速度对比 ===
  function renderVectorDbChart(vectorDb) {
    const ctx = document.getElementById('chart-vectordb');
    if (!ctx) return;
    const dbs = vectorDb.databases.filter(d => d.index_speed > 0);

    new Chart(ctx, {
      type: 'bar',
      data: {
        labels: dbs.map(d => d.name),
        datasets: [{
          label: '索引速度 (docs/s)',
          data: dbs.map(d => d.index_speed),
          backgroundColor: ['#06b6d480', '#10b98180', '#f59e0b80', '#8b5cf680', '#ec489980'],
          borderColor: ['#06b6d4', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899'],
          borderWidth: 2,
          borderRadius: 8
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: '#1a1a24',
            borderColor: '#3a3a4e',
            borderWidth: 1,
            callbacks: {
              label: ctx => {
                const db = dbs[ctx.dataIndex];
                return [
                  `索引: ${db.index_speed.toLocaleString()} docs/s`,
                  db.search_speed ? `检索: ${db.search_speed} ms` : '检索: 未测试',
                  db.memory ? `内存: ${db.memory} MB` : '内存: 未测试'
                ];
              }
            }
          }
        },
        scales: {
          x: {
            type: 'logarithmic',
            grid: { color: '#2a2a3a' },
            ticks: { callback: v => v >= 1000 ? (v / 1000) + 'K' : v }
          },
          y: {
            grid: { display: false }
          }
        }
      }
    });
  }

  // === 图3: 版本演进准确率 ===
  function renderVersionChart(metrics) {
    const ctx = document.getElementById('chart-version');
    if (!ctx) return;
    const data = metrics.version_evolution.data;

    new Chart(ctx, {
      type: 'line',
      data: {
        labels: data.map(d => d.version),
        datasets: [{
          label: '准确率 (%)',
          data: data.map(d => d.accuracy),
          borderColor: '#6366f1',
          backgroundColor: 'rgba(99, 102, 241, 0.1)',
          borderWidth: 3,
          fill: true,
          tension: 0.3,
          stepped: true,
          pointBackgroundColor: '#8b5cf6',
          pointBorderColor: '#e4e4ef',
          pointBorderWidth: 2,
          pointRadius: 6,
          pointHoverRadius: 8
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: '#1a1a24',
            borderColor: '#3a3a4e',
            borderWidth: 1,
            callbacks: {
              title: ctx => data[ctx[0].dataIndex].version + ' (' + data[ctx[0].dataIndex].date + ')',
              label: ctx => `准确率: ${ctx.parsed.y}%`,
              afterLabel: ctx => data[ctx.dataIndex].description
            }
          }
        },
        scales: {
          y: {
            beginAtZero: true,
            max: 100,
            grid: { color: '#2a2a3a' },
            ticks: { callback: v => v + '%' }
          },
          x: {
            grid: { display: false }
          }
        }
      }
    });
  }

  // === 图4: RAGAS 评测雷达图 ===
  function renderRagasChart(metrics) {
    const ctx = document.getElementById('chart-ragas');
    if (!ctx) return;
    const data = metrics.ragas_scores.data;

    new Chart(ctx, {
      type: 'radar',
      data: {
        labels: data.map(d => d.metric),
        datasets: [{
          label: 'RAGAS 得分',
          data: data.map(d => d.score * 100),
          backgroundColor: 'rgba(139, 92, 246, 0.15)',
          borderColor: '#8b5cf6',
          borderWidth: 2,
          pointBackgroundColor: '#ec4899',
          pointBorderColor: '#e4e4ef',
          pointBorderWidth: 2,
          pointRadius: 5
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: '#1a1a24',
            borderColor: '#3a3a4e',
            borderWidth: 1,
            callbacks: {
              label: ctx => `${ctx.label}: ${(ctx.parsed.r / 100).toFixed(3)}`
            }
          }
        },
        scales: {
          r: {
            beginAtZero: true,
            max: 100,
            grid: { color: '#2a2a3a' },
            angleLines: { color: '#2a2a3a' },
            pointLabels: { color: '#a0a0b8', font: { size: 11 } },
            ticks: {
              color: '#666680',
              backdropColor: 'transparent',
              callback: v => (v / 100).toFixed(1)
            }
          }
        }
      }
    });
  }
});
