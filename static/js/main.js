// static/js/main.js

// Animate progress bars on page load (when results are present)
document.addEventListener('DOMContentLoaded', () => {
  // Trigger width transitions by briefly setting width to 0 then restoring
  document.querySelectorAll('[data-bar]').forEach(bar => {
    const target = bar.style.width;
    bar.style.width = '0';
    requestAnimationFrame(() => {
      requestAnimationFrame(() => { bar.style.width = target; });
    });
  });
});