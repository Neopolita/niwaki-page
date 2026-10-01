(() => {
  const select = document.getElementById('text-qa-select');
  const controls = document.querySelector('.text-qa-controls');
  const image = document.getElementById('text-qa-image');
  const full = document.getElementById('text-qa-full');
  const previous = document.getElementById('text-qa-prev');
  const next = document.getElementById('text-qa-next');
  const position = document.getElementById('text-qa-position');
  if (!select || !controls || !image || !full || !previous || !next || !position) return;

  const show = (index) => {
    select.selectedIndex = index;
    const path = select.value;
    image.src = path;
    image.alt = `Text QA prompts ${3 * index + 1} through ${3 * index + 3}: teacher, pretrained bridge, update 120, update 180`;
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
})();
