// Runs before the page draws, so a saved theme shows from the first frame. It is a file rather
// than inline, so the content security policy allows it as part of the site.
document.documentElement.classList.add('js');
try {
  const saved = localStorage.getItem('tv-theme');
  if (saved === 'light' || saved === 'dark') document.documentElement.dataset.theme = saved;
} catch {
  // Storage can be unavailable; fall back to the system theme.
}
