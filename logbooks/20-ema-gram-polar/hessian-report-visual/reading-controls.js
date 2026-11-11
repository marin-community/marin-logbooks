(() => {
  const $ = id => document.getElementById(id);
  const chapters = [...document.querySelectorAll('.chapter')];
  let size = 20, extra = false, focus = false, current = 0;
  try { const saved = JSON.parse(localStorage.getItem('hessian-reading') || '{}'); size = Math.max(18, Math.min(28, saved.size || 20)); extra = saved.extra === true; } catch (_) {}
  function appearance() {
    document.documentElement.style.setProperty('--read-size', size + 'px');
    document.documentElement.style.setProperty('--leading', extra ? '2.1' : '1.8');
    document.documentElement.style.setProperty('--paragraph', extra ? '1.6em' : '1.2em');
    $('spacing').setAttribute('aria-pressed', String(extra));
    $('settings').textContent = size + ' px' + (extra ? ' · spacious' : '');
    $('smaller').disabled = size <= 18; $('larger').disabled = size >= 28;
    try { localStorage.setItem('hessian-reading', JSON.stringify({size, extra})); } catch (_) {}
  }
  $('smaller').onclick = () => { size = Math.max(18, size - 2); appearance(); };
  $('larger').onclick = () => { size = Math.min(28, size + 2); appearance(); };
  $('spacing').onclick = () => { extra = !extra; appearance(); };
  function show(scroll = false) {
    chapters.forEach((chapter, i) => { chapter.hidden = focus && i !== current; });
    $('step-nav').hidden = !focus;
    $('focus').setAttribute('aria-pressed', String(focus));
    $('focus').textContent = focus ? 'Show all sections' : 'One section at a time';
    $('previous').disabled = current === 0; $('next').disabled = current === chapters.length - 1;
    $('progress').textContent = (current + 1) + ' of ' + chapters.length;
    if (scroll) chapters[current].querySelector('h2').focus({preventScroll:true});
    if (scroll) chapters[current].scrollIntoView({block:'start'});
  }
  $('focus').onclick = () => { focus = !focus; show(focus); };
  $('previous').onclick = () => { current = Math.max(0, current - 1); show(true); };
  $('next').onclick = () => { current = Math.min(chapters.length - 1, current + 1); show(true); };
  document.querySelectorAll('a[href^="#chapter-"]').forEach(a => a.addEventListener('click', () => {
    current = Number(a.hash.split('-').pop()); show(); document.querySelector('.contents').open = false;
  }));
  let lastFigure;
  document.querySelectorAll('.zoom').forEach(button => button.addEventListener('click', () => {
    lastFigure = button;
    const image = button.querySelector('img'); $('large-image').src = image.src; $('large-image').alt = image.alt;
    $('figure-dialog').showModal(); $('close-figure').focus();
  }));
  $('close-figure').onclick = () => $('figure-dialog').close();
  $('figure-dialog').addEventListener('close', () => lastFigure?.focus());
  appearance(); show();
})();
