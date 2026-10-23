/* Ikon garis 24px (stroke 1.75), tanpa pustaka luar. ic("check") -> string <svg>. */
(function () {
  var P = {
    arrow: '<path d="M5 12h14M13 6l6 6-6 6"/>',
    check: '<path d="M5 12.5l4.5 4.5L19 7.5"/>',
    x: '<path d="M6.5 6.5l11 11M17.5 6.5l-11 11"/>',
    bottle: '<path d="M10 2.5h4v2.8l1.6 2.6V19.5a2 2 0 0 1-2 2h-3.2a2 2 0 0 1-2-2V7.9L10 5.3z"/><path d="M8.4 11.5h7.2M8.4 15.5h7.2"/>',
    cap: '<path d="M9 7h6v3H9zM10 10v2.5M14 10v2.5"/><path d="M8 16.5h8l1.5 4h-11z"/><path d="M12 2.5v2.2M10.6 3.6L12 2.3l1.4 1.3"/>',
    drop: '<path d="M12 3.2s6 6.4 6 10.8a6 6 0 0 1-12 0c0-4.4 6-10.8 6-10.8z"/><path d="M9.2 14.6a2.9 2.9 0 0 0 2.6 2.6"/>',
    lay: '<path d="M2.5 10h11l2.6-1.6h3.4v7.2h-3.4L13.5 14h-11z"/><path d="M2 19.5h20"/>',
    leaf: '<path d="M5.5 19.5C5 11 10.5 5 20 4.5c.4 9.6-5.5 15.2-14.5 15"/><path d="M5.5 19.5L13 12"/>',
    qr: '<path d="M4 4h6v6H4zM14 4h6v6h-6zM4 14h6v6H4zM14 14h2.5v2.5H14zM17.5 17.5H20V20h-2.5zM14 18.5h1.5V20H14zM18.5 14H20v1.5h-1.5z"/>',
    heart: '<path d="M12 20s-7.5-4.6-7.5-10.2A4.2 4.2 0 0 1 12 7.2a4.2 4.2 0 0 1 7.5 2.6C19.5 15.4 12 20 12 20z"/>',
    gift: '<path d="M4.5 11h15v9h-15zM3.5 7.5h17V11h-17zM12 7.5V20"/><path d="M12 7.5C10.5 4.4 7 4.2 7 6s3 1.5 5 1.5zM12 7.5c1.5-3.1 5-3.3 5-1.5s-3 1.5-5 1.5z"/>',
    user: '<circle cx="12" cy="8" r="4"/><path d="M4.5 20.5c.4-3.8 3.6-5.8 7.5-5.8s7.1 2 7.5 5.8"/>',
    clock: '<circle cx="12" cy="12" r="8.5"/><path d="M12 7.5V12l3 2"/>',
    wallet: '<path d="M4 7.5h14.5a1.5 1.5 0 0 1 1.5 1.5v9.5a1.5 1.5 0 0 1-1.5 1.5H4z"/><path d="M4 7.5V5.8A1.8 1.8 0 0 1 5.8 4H16v3.5M15.5 13.8h1"/>',
    history: '<path d="M4.5 12a7.5 7.5 0 1 0 2.2-5.3"/><path d="M4.5 4.5v3.8h3.8M12 8.2V12l2.8 1.8"/>',
    logout: '<path d="M14.5 4.5h4v15h-4M10 8l-4 4 4 4M6.2 12H16"/>',
    phone: '<path d="M8 2.8h8a1.2 1.2 0 0 1 1.2 1.2v16a1.2 1.2 0 0 1-1.2 1.2H8A1.2 1.2 0 0 1 6.8 20V4A1.2 1.2 0 0 1 8 2.8z"/><path d="M11 18h2"/>',
    scan: '<path d="M4 8.5V5.5A1.5 1.5 0 0 1 5.5 4h3M15.5 4h3A1.5 1.5 0 0 1 20 5.5v3M20 15.5v3a1.5 1.5 0 0 1-1.5 1.5h-3M8.5 20h-3A1.5 1.5 0 0 1 4 18.5v-3M4 12h16"/>',
    scale: '<path d="M12 4v16M5 20h14M6 8h12M6 8l-3 6a3 3 0 0 0 6 0zM18 8l-3 6a3 3 0 0 0 6 0z"/>',
    spark: '<path d="M12 3v4M12 17v4M3 12h4M17 12h4M6 6l2.5 2.5M15.5 15.5L18 18M18 6l-2.5 2.5M8.5 15.5L6 18"/>',
    back: '<path d="M19 12H5M11 18l-6-6 6-6"/>',
    lock: '<rect x="5" y="10.5" width="14" height="10" rx="2"/><path d="M8.5 10.5V7.8a3.5 3.5 0 0 1 7 0v2.7"/>',
    info: '<circle cx="12" cy="12" r="8.5"/><path d="M12 11v5.5M12 7.8v.4"/>',
    wifi: '<path d="M3.5 9.5a12 12 0 0 1 17 0M6.5 12.8a7.5 7.5 0 0 1 11 0M9.5 16a3 3 0 0 1 5 0"/><path d="M12 19.5h.01"/>'
  };
  window.ic = function (name, cls) {
    return '<svg class="ico ' + (cls || '') + '" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' + (P[name] || '') + '</svg>';
  };
  window.hydrateIcons = function (root) {
    (root || document).querySelectorAll('[data-ic]').forEach(function (el) {
      el.insertAdjacentHTML('afterbegin', window.ic(el.getAttribute('data-ic'), el.getAttribute('data-ic-cls')));
      el.removeAttribute('data-ic');
    });
  };
  window.fmt = function (n) { return Number(n || 0).toLocaleString('id-ID'); };
  window.gramText = function (g) { g = Number(g || 0); return g >= 1000 ? (g / 1000).toLocaleString('id-ID', { maximumFractionDigits: 1 }) + ' kg' : fmt(g) + ' g'; };
  window.ago = function (ts) {
    var s = Math.max(0, Date.now() / 1000 - ts);
    if (s < 60) return 'baru saja';
    if (s < 3600) return Math.floor(s / 60) + ' mnt lalu';
    if (s < 86400) return Math.floor(s / 3600) + ' jam lalu';
    return Math.floor(s / 86400) + ' hari lalu';
  };
  window.esc = function (s) { return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]; }); };
  window.countUp = function (el, to, ms) {
    var from = Number(el.getAttribute('data-v') || 0); to = Number(to || 0); el.setAttribute('data-v', to);
    if (from === to || window.matchMedia('(prefers-reduced-motion: reduce)').matches) { el.textContent = fmt(to); return; }
    var t0 = performance.now(); ms = ms || 700;
    (function step(t) {
      var k = Math.min(1, (t - t0) / ms), e = 1 - Math.pow(1 - k, 4);
      el.textContent = fmt(Math.round(from + (to - from) * e));
      if (k < 1) requestAnimationFrame(step);
    })(t0);
  };
  window.api = function (url, body, method) {
    return fetch(url, {
      method: method || (body === undefined ? 'GET' : 'POST'), credentials: 'same-origin',
      headers: body === undefined ? {} : { 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body)
    }).then(function (r) {
      return r.json().catch(function () { return {}; }).then(function (j) {
        if (!r.ok) { var e = new Error(j.pesan || j.detail && (j.detail[0] && j.detail[0].msg || j.detail) || ('Galat ' + r.status)); e.status = r.status; throw e; }
        return j;
      });
    }, function () { throw new Error('Tidak bisa menghubungi mesin. Pastikan HP tersambung ke Wi-Fi yang sama.'); });
  };
})();
