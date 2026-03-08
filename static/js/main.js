/* ============================================================
   EIDS — main.js  |  Global UI interactions
   ============================================================ */

document.addEventListener('DOMContentLoaded', () => {

  /* ──────────────────────────────────────────────────────────
     1. Active nav link highlight
     ────────────────────────────────────────────────────────── */
  const currentPath = window.location.pathname;
  document.querySelectorAll('.nav-links a').forEach(link => {
    const href = link.getAttribute('href');
    if (href && currentPath.startsWith(href) && href !== '/') {
      link.classList.add('active');
    }
  });

  /* ──────────────────────────────────────────────────────────
     2. Animate probability bars on page load
        Bars start at width:0 in HTML; JS sets real width
        after a short delay so the CSS transition fires.
     ────────────────────────────────────────────────────────── */
  const bars = document.querySelectorAll('.prob-bar-fill[data-width]');
  requestAnimationFrame(() => {
    setTimeout(() => {
      bars.forEach(bar => {
        bar.style.width = bar.dataset.width + '%';
      });
    }, 80);
  });

  /* ──────────────────────────────────────────────────────────
     3. Flash message auto-dismiss after 5 seconds
     ────────────────────────────────────────────────────────── */
  document.querySelectorAll('.flash').forEach(el => {
    setTimeout(() => {
      el.style.transition = 'opacity .4s';
      el.style.opacity = '0';
      setTimeout(() => el.remove(), 400);
    }, 5000);
  });

  /* ──────────────────────────────────────────────────────────
     4. Prediction form — loading state on submit
     ────────────────────────────────────────────────────────── */
  document.querySelectorAll('form').forEach(form => {
    form.addEventListener('submit', () => {
      const btn = form.querySelector('.btn-primary, .btn-submit');
      if (btn) {
        btn.disabled = true;
        btn.textContent = 'Running…';
        btn.style.opacity = '.7';
      }
    });
  });

  /* ──────────────────────────────────────────────────────────
     5. Number input — scroll wheel adjusts value
     ────────────────────────────────────────────────────────── */
  document.querySelectorAll('input[type="number"]').forEach(input => {
    input.addEventListener('wheel', e => {
      e.preventDefault();
      const step = parseFloat(input.step) || 1;
      const delta = e.deltaY < 0 ? step : -step;
      const current = parseFloat(input.value) || 0;
      const min = input.min !== '' ? parseFloat(input.min) : -Infinity;
      const max = input.max !== '' ? parseFloat(input.max) : Infinity;
      const next = Math.min(max, Math.max(min, parseFloat((current + delta).toFixed(6))));
      input.value = next;
    });
  });

  /* ──────────────────────────────────────────────────────────
     6. Result card — copy-to-clipboard on headline double-click
     ────────────────────────────────────────────────────────── */
  document.querySelectorAll('.result-headline').forEach(el => {
    el.title = 'Double-click to copy';
    el.style.cursor = 'pointer';
    el.addEventListener('dblclick', () => {
      navigator.clipboard.writeText(el.textContent.trim()).then(() => {
        const orig = el.textContent;
        el.textContent = '✓ Copied to clipboard';
        setTimeout(() => { el.textContent = orig; }, 1500);
      });
    });
  });

});
