/* main.js - 滚动动画 + 导航 + 打字机效果 */

document.addEventListener('DOMContentLoaded', () => {
  // === 打字机效果 ===
  const typewriterEl = document.querySelector('.typewriter');
  if (typewriterEl) {
    const phrases = [
      '混合检索 + Corrective RAG + Self-RAG 闭环',
      'LangGraph 7层 Pipeline · 自纠错 · 多工具路由',
      '60 条 Golden · 150 条意图 · 20 条 E2E 回归',
      'Qdrant 向量库 · BM25 + RRF 融合 · Cross-Encoder 精排'
    ];
    let phraseIdx = 0;
    let charIdx = 0;
    let isDeleting = false;

    function type() {
      const current = phrases[phraseIdx];
      if (isDeleting) {
        charIdx--;
      } else {
        charIdx++;
      }
      typewriterEl.textContent = current.substring(0, charIdx);

      let delay = isDeleting ? 30 : 60;
      if (!isDeleting && charIdx === current.length) {
        delay = 2000;
        isDeleting = true;
      } else if (isDeleting && charIdx === 0) {
        isDeleting = false;
        phraseIdx = (phraseIdx + 1) % phrases.length;
        delay = 500;
      }
      setTimeout(type, delay);
    }
    type();
  }

  // === 滚动淡入动画 ===
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
      }
    });
  }, { threshold: 0.1, rootMargin: '0px 0px -50px 0px' });

  document.querySelectorAll('.fade-in').forEach(el => observer.observe(el));

  // === 导航栏滚动效果 ===
  const navbar = document.querySelector('.navbar');
  let lastScroll = 0;
  window.addEventListener('scroll', () => {
    const currentScroll = window.scrollY;
    if (currentScroll > 50) {
      navbar.style.borderBottomColor = 'var(--border-bright)';
    } else {
      navbar.style.borderBottomColor = 'var(--border)';
    }
    lastScroll = currentScroll;
  });

  // === 平滑滚动到锚点 ===
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', (e) => {
      const target = document.querySelector(anchor.getAttribute('href'));
      if (target) {
        e.preventDefault();
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    });
  });
});
