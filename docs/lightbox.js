(function () {
  const modal = document.createElement('div');
  modal.className = 'lb-modal';
  modal.innerHTML =
    '<button class="lb-close" aria-label="Close">Close (Esc)</button>' +
    '<img alt=""><div class="lb-cap"></div>';
  document.body.appendChild(modal);

  const imgEl = modal.querySelector('img');
  const capEl = modal.querySelector('.lb-cap');
  const close = () => modal.classList.remove('open');

  modal.querySelector('.lb-close').addEventListener('click', close);
  modal.addEventListener('click', (e) => { if (e.target === modal) close(); });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') close(); });

  document.querySelectorAll('.scard img').forEach((im) => {
    im.addEventListener('click', () => {
      imgEl.src = im.src;
      imgEl.alt = im.alt;
      const meta = im.closest('.scard')?.querySelector('.smeta');
      const title = meta?.querySelector('b')?.textContent || im.alt;
      const sub = meta?.querySelector('.subline')?.textContent || '';
      capEl.textContent = sub ? `${title} — ${sub}` : title;
      modal.classList.add('open');
    });
  });
})();
