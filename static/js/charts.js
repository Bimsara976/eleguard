/* ============================================================
   EIDS — charts.js  |  Result page animations & visual helpers
   ============================================================ */

/* ──────────────────────────────────────────────────────────
   Animated number counter
   Usage: countUp(element, endValue, duration, suffix)
   ────────────────────────────────────────────────────────── */
function countUp(el, end, duration = 900, suffix = '') {
  const start = 0;
  const startTime = performance.now();

  function update(now) {
    const elapsed = now - startTime;
    const progress = Math.min(elapsed / duration, 1);
    // ease-out cubic
    const ease = 1 - Math.pow(1 - progress, 3);
    const current = (start + (end - start) * ease).toFixed(1);
    el.textContent = current + suffix;
    if (progress < 1) requestAnimationFrame(update);
  }
  requestAnimationFrame(update);
}

/* ──────────────────────────────────────────────────────────
   Animate all .metric-val elements that contain a number
   ────────────────────────────────────────────────────────── */
function animateMetrics() {
  document.querySelectorAll('.metric-val').forEach(el => {
    const raw = el.textContent.trim();
    const num = parseFloat(raw);
    if (!isNaN(num) && raw.includes('.')) {
      const suffix = raw.replace(/[\d.]/g, '');
      countUp(el, num, 800, suffix);
    }
  });
}

/* ──────────────────────────────────────────────────────────
   Staggered reveal for result grid children
   ────────────────────────────────────────────────────────── */
function staggerReveal(selector, delayStep = 80) {
  document.querySelectorAll(selector).forEach((el, i) => {
    el.style.opacity = '0';
    el.style.transform = 'translateY(10px)';
    el.style.transition = 'opacity .35s ease, transform .35s ease';
    setTimeout(() => {
      el.style.opacity = '1';
      el.style.transform = 'translateY(0)';
    }, i * delayStep + 150);
  });
}

/* ──────────────────────────────────────────────────────────
   Risk level colouring helper
   Sets the correct CSS class on .metric-val based on threshold
   ────────────────────────────────────────────────────────── */
function applyRiskColour(el, value, warnThreshold = 40, dangerThreshold = 70) {
  el.classList.remove('warn', 'danger');
  if (value >= dangerThreshold) el.classList.add('danger');
  else if (value >= warnThreshold) el.classList.add('warn');
}

/* ──────────────────────────────────────────────────────────
   Initialise on DOM ready
   ────────────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {

  // Animate metric values if a result card is present
  if (document.querySelector('.result-card')) {
    animateMetrics();
    staggerReveal('.metric');
    staggerReveal('.prob-bar-wrap', 60);
  }

  // Animate all prob-bar-fill elements (data-width driven)
  // Bars should have style="width:0" in HTML and data-width="<real %>"
  const fills = document.querySelectorAll('.prob-bar-fill[data-width]');
  if (fills.length) {
    requestAnimationFrame(() => {
      setTimeout(() => {
        fills.forEach(bar => {
          bar.style.width = bar.dataset.width + '%';
        });
      }, 100);
    });
  }

  // Apply colour classes to movement risk metric
  document.querySelectorAll('[data-risk-value]').forEach(el => {
    const val = parseFloat(el.dataset.riskValue);
    if (!isNaN(val)) applyRiskColour(el, val);
  });

});
