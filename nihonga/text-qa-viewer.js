// Progressive enhancement for every sheet viewer on the page: without JavaScript the
// first sheet and the direct links remain; with it, previous/next/select controls appear.
document.querySelectorAll('.sheet-viewer').forEach((viewer) => {
  const select = viewer.querySelector('.sheet-select');
  const controls = viewer.querySelector('.text-qa-controls');
  const image = viewer.querySelector('.sheet-image');
  const full = viewer.querySelector('.sheet-full');
  const previous = viewer.querySelector('.sheet-prev');
  const next = viewer.querySelector('.sheet-next');
  const position = viewer.querySelector('.sheet-position');
  if (!select || !controls || !image || !full || !previous || !next || !position) return;
  const per = Number(viewer.dataset.perSheet || 3);
  const total = Number(viewer.dataset.total || select.length * per);
  const altTemplate = viewer.dataset.alt || 'Prompts {start} through {end}';

  const show = (index) => {
    select.selectedIndex = index;
    const path = select.value;
    const start = per * index + 1;
    const end = Math.min(per * index + per, total);
    image.src = path;
    image.alt = altTemplate.replace('{start}', start).replace('{end}', end);
    full.href = path;
    position.textContent = `${index + 1} of ${select.length}`;
    previous.disabled = index === 0;
    next.disabled = index === select.length - 1;
  };

  select.addEventListener('change', () => show(select.selectedIndex));
  previous.addEventListener('click', () => show(select.selectedIndex - 1));
  next.addEventListener('click', () => show(select.selectedIndex + 1));
  controls.hidden = false;
  show(0);
});
