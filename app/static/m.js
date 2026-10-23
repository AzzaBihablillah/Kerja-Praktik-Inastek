/* Halaman HP: /k/<kode> klaim poin, /k ketik kode, /akun dompet + hadiah + donasi + riwayat. */
(function () {
  'use strict';
  var view = document.getElementById('view');
  var ME = null;           // profil bila sudah masuk
  var BOTTLE_BG = '<svg class="bottle-bg" viewBox="0 0 200 420" aria-hidden="true"><path d="M80 10h40v34l18 28v300a28 28 0 0 1-28 28H90a28 28 0 0 1-28-28V72l18-28z" fill="none" stroke="currentColor" stroke-width="10"/><path d="M62 150h76M62 250h76" stroke="currentColor" stroke-width="10"/></svg>';

  function render(html) { view.innerHTML = html; hydrateIcons(view); window.scrollTo(0, 0); }
  function when(ts) { return new Date(ts * 1000).toLocaleString('id-ID', { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' }); }
  function q(sel, root) { return (root || view).querySelector(sel); }

  function loadMe() {
    return api('/api/akun').then(function (d) {
      ME = d.masuk ? d.profil : null;
      var me = document.getElementById('me');
      if (ME) { me.hidden = false; me.innerHTML = '<span class="av">' + esc(ME.nama.charAt(0).toUpperCase()) + '</span>' + fmt(ME.saldo) + ' poin'; }
      else me.hidden = true;
      return ME;
    }).catch(function () { return null; });
  }

  /* ------------------------------------------------ masuk / daftar */
  function authCard(container, onDone, intro) {
    var mode = 'masuk';
    function draw() {
      container.innerHTML =
        '<form class="card" novalidate>' + (intro ? '<p class="lead">' + intro + '</p>' : '') +
        '<div class="seg" role="tablist"><button type="button" role="tab" data-m="masuk" aria-selected="' + (mode === 'masuk') + '">Masuk</button>' +
        '<button type="button" role="tab" data-m="daftar" aria-selected="' + (mode === 'daftar') + '">Buat akun</button></div>' +
        '<div class="field"><label for="fHp">Nomor HP</label><input id="fHp" name="hp" type="tel" inputmode="tel" autocomplete="tel" placeholder="0812 3456 7890" required></div>' +
        (mode === 'daftar' ? '<div class="field"><label for="fNama">Nama panggilan</label><input id="fNama" name="nama" autocomplete="nickname" maxlength="24" placeholder="mis. Rara" required><span class="hint">Tampil di akunmu. Tidak perlu nama lengkap.</span></div>' : '') +
        '<div class="field"><label for="fPin">PIN 6 angka</label><input id="fPin" class="pin" name="pin" type="password" inputmode="numeric" pattern="[0-9]*" maxlength="6" autocomplete="' + (mode === 'daftar' ? 'new-password' : 'current-password') + '" required>' +
        (mode === 'daftar' ? '<span class="hint">Jangan pakai PIN yang sama dengan rekening bank.</span>' : '') + '</div>' +
        '<p class="err" role="alert"></p>' +
        '<button class="btn btn-lime btn-block" type="submit">' + (mode === 'masuk' ? 'Masuk' : 'Buat akun & lanjut') + '</button></form>';
      hydrateIcons(container);
      container.querySelectorAll('[data-m]').forEach(function (b) { b.addEventListener('click', function () { mode = b.getAttribute('data-m'); draw(); }); });
      var f = container.querySelector('form');
      f.pin.addEventListener('input', function () { this.value = this.value.replace(/\D/g, '').slice(0, 6); });
      f.addEventListener('submit', function (e) {
        e.preventDefault();
        var btn = f.querySelector('[type=submit]'), err = f.querySelector('.err');
        var body = { hp: f.hp.value, pin: f.pin.value }; if (mode === 'daftar') body.nama = f.nama.value;
        btn.disabled = true; err.textContent = '';
        api('/api/akun/' + mode, body).then(loadMe).then(function () { onDone(); })
          .catch(function (x) { err.textContent = x.message; btn.disabled = false; });
      });
      setTimeout(function () { f.hp.focus(); }, 50);
    }
    draw();
  }

  /* ------------------------------------------------ klaim */
  function ticketHtml(k, label) {
    return '<div class="ticket">' + BOTTLE_BG + '<p class="eyebrow">' + label + '</p>' +
      '<p class="big"><span class="num">' + fmt(k.points) + '</span><span>poin</span></p>' +
      '<p class="sub">≈ ' + gramText(k.grams) + ' plastik tidak berakhir di TPA</p>' +
      '<div class="foot"><span>Kode <span class="mono">' + esc(k.code_fmt) + '</span></span><span>' +
      (k.status === 'open' ? 'Berlaku s.d. ' + when(k.expires) : '') + '</span></div></div>';
  }

  function claimView(code) {
    Promise.all([api('/api/klaim/' + encodeURIComponent(code)), loadMe()]).then(function (r) {
      var k = r[0].klaim, camps = r[0].kampanye;
      if (k.status !== 'open') {
        var msg = { claimed: ['Poin ini sudah disimpan.', 'Kode ini sudah dipakai untuk menyimpan poin ke akun.'],
          donated: ['Poin ini sudah didonasikan.', 'Disalurkan ke ' + (k.campaign || 'kampanye') + '. Terima kasih.'],
          expired: ['Kode sudah kedaluwarsa.', 'Kode klaim berlaku 7 hari. Poin yang tidak diklaim sampai batas itu didonasikan ke program lingkungan.'] }[k.status];
        render(ticketHtml(k, 'Kode klaim') + '<div class="stack"><h1 class="mh">' + msg[0] + '</h1><p class="lead">' + esc(msg[1]) + '</p></div>' +
          '<a class="btn btn-block" href="/akun">' + (ME ? 'Buka akunku' : 'Masuk ke akun') + '</a>');
        return;
      }
      render(ticketHtml(k, 'Hasil setor botolmu') +
        '<div class="stack"><h1 class="mh">Mau dipakai untuk apa?</h1></div>' +
        '<div class="stack" id="choices">' +
        '<button class="choice primary" id="cSave" type="button"><span class="c-ic" data-ic="wallet"></span><span><b>Simpan ke akun</b><span class="d">' +
        (ME ? 'Masuk sebagai ' + esc(ME.nama) + ' · saldo ' + fmt(ME.saldo) + ' poin' : 'Kumpulkan, lalu tukar hadiah') + '</span></span><span data-ic="arrow"></span></button>' +
        '<button class="choice" id="cDonate" type="button"><span class="c-ic" data-ic="heart"></span><span><b>Donasikan</b><span class="d">Salurkan ke program lingkungan</span></span><span data-ic="arrow"></span></button>' +
        '</div><div id="step"></div>');
      var step = q('#step');
      q('#cSave').addEventListener('click', function () {
        if (ME) return save();
        authCard(step, save, 'Masuk dengan nomor HP. Belum punya akun? Pilih <b>Buat akun</b>, hanya butuh 20 detik.');
        step.scrollIntoView({ behavior: 'smooth', block: 'start' });
      });
      q('#cDonate').addEventListener('click', function () {
        pickCampaign(step, camps, 'Donasikan ' + fmt(k.points) + ' poin', function (cid, btn, err) {
          api('/api/klaim/' + k.code + '/donasi', { campaign_id: cid }).then(function (d) { donated(d.points, d.kampanye); })
            .catch(function (x) { err.textContent = x.message; btn.disabled = false; });
        });
        step.scrollIntoView({ behavior: 'smooth', block: 'start' });
      });
      function save() {
        step.innerHTML = '<div class="sk" style="height:56px;border-radius:999px"></div>';
        api('/api/klaim/' + k.code + '/simpan', {}).then(function (d) {
          loadMe();
          render('<div class="done"><span class="badge">' + ic('check') + '</span><h1 class="mh">' + fmt(d.points) + ' poin masuk.</h1>' +
            '<p class="lead">Saldo ' + esc(ME ? ME.nama : 'kamu') + ' sekarang <b>' + fmt(d.saldo) + ' poin</b>. Kumpulkan terus, lalu tukar dengan hadiah atau donasikan.</p>' +
            '<a class="btn btn-lime btn-block" href="/akun#hadiah">' + ic('gift') + 'Lihat hadiah</a><a class="btn btn-ghost btn-block" href="/akun#riwayat">Riwayat poin</a></div>');
        }).catch(function (x) { step.innerHTML = '<p class="err" role="alert">' + esc(x.message) + '</p>'; });
      }
    }).catch(function (e) {
      if (e.status === 404) return codeView('Kode <b class="mono">' + esc(code) + '</b> tidak ditemukan. Ketik ulang kode di layar mesin.');
      render('<div class="empty">' + ic('wifi') + '<b>Gagal memuat.</b><span>' + esc(e.message) + '</span><button class="btn btn-quiet" onclick="location.reload()">Coba lagi</button></div>');
    });
  }

  function pickCampaign(container, camps, label, onGo) {
    var pick = null;
    container.innerHTML = '<div class="card"><p class="lead"><b>Pilih program</b> <span class="chip">simulasi</span></p><ul class="rows" role="radiogroup">' +
      camps.map(function (c, i) {
        var p = Math.min(1, c.terkumpul / c.target);
        return '<li class="row-item radio-item" role="radio" tabindex="0" aria-checked="false" data-id="' + c.id + '" style="animation-delay:' + (i * .06) + 's">' +
          '<span class="r-ic">' + ic(i === 0 ? 'leaf' : i === 1 ? 'bottle' : 'drop') + '</span><span><b>' + esc(c.nama) + '</b><span class="d">' + esc(c.ringkas) + '</span>' +
          '<div class="bar"><i style="transform:scaleX(' + p.toFixed(3) + ')"></i></div><span class="d">' + fmt(c.terkumpul) + ' dari ' + fmt(c.target) + ' poin</span></span><span></span></li>';
      }).join('') + '</ul><p class="err" role="alert"></p><button class="btn btn-lime btn-block" type="button" disabled>' + esc(label) + '</button></div>';
    var btn = container.querySelector('.btn'), err = container.querySelector('.err');
    container.querySelectorAll('.radio-item').forEach(function (li) {
      function choose() {
        pick = Number(li.getAttribute('data-id'));
        container.querySelectorAll('.radio-item').forEach(function (x) { x.setAttribute('aria-checked', x === li ? 'true' : 'false'); });
        btn.disabled = false;
      }
      li.addEventListener('click', choose);
      li.addEventListener('keydown', function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); choose(); } });
    });
    btn.addEventListener('click', function () { btn.disabled = true; err.textContent = ''; onGo(pick, btn, err); });
  }

  function donated(points, nama) {
    render('<div class="done"><span class="badge dark">' + ic('heart') + '</span><h1 class="mh">Terima kasih.</h1>' +
      '<p class="lead"><b>' + fmt(points) + ' poin</b> disalurkan ke ' + esc(nama) + '. Botolmu hari ini ikut membiayai program ini.</p>' +
      '<a class="btn btn-block" href="/akun">' + (ME ? 'Kembali ke akun' : 'Buat akun untuk setoran berikutnya') + '</a></div>');
  }

  function codeView(msg) {
    render('<div class="stack"><p class="eyebrow">Klaim poin</p><h1 class="mh">Ketik kode dari layar mesin.</h1>' +
      (msg ? '<p class="lead">' + msg + '</p>' : '<p class="lead">Kode 6 huruf/angka di bawah QR, mis. <span class="mono">K8J-FM2</span>.</p>') + '</div>' +
      '<form class="card" novalidate><div class="field"><label for="fCode">Kode klaim</label><input id="fCode" class="mono" autocomplete="off" autocapitalize="characters" maxlength="7" placeholder="XXX-XXX" style="font-size:24px;letter-spacing:.2em"></div>' +
      '<button class="btn btn-lime btn-block" type="submit">Buka</button></form>');
    var f = q('form'), inp = q('#fCode');
    inp.addEventListener('input', function () {
      var v = this.value.toUpperCase().replace(/[^0-9A-Z]/g, '').slice(0, 6);
      this.value = v.length > 3 ? v.slice(0, 3) + '-' + v.slice(3) : v;
    });
    f.addEventListener('submit', function (e) { e.preventDefault(); var c = inp.value.replace('-', ''); if (c.length === 6) location.href = '/k/' + c; });
    inp.focus();
  }

  /* ------------------------------------------------ akun */
  var KIND = { setor: 'Setor botol', tukar: 'Tukar hadiah', donasi: 'Donasi' };
  function accountView() {
    loadMe().then(function (me) {
      if (!me) {
        render('<div class="stack"><p class="eyebrow">Dompet poin</p><h1 class="mh">Masuk untuk melihat poinmu.</h1></div><div id="auth"></div>' +
          '<p class="small muted">Punya kode dari mesin? <a href="/k">Klaim poin di sini</a>.</p>');
        authCard(q('#auth'), accountView);
        return;
      }
      var tab = (location.hash || '#hadiah').slice(1); if (['hadiah', 'donasi', 'riwayat'].indexOf(tab) < 0) tab = 'hadiah';
      render('<div class="hi"><div><p class="eyebrow">' + esc(me.hp_mask) + '</p><h1>Hai, ' + esc(me.nama) + '.</h1></div>' +
        '<button class="icon-btn" id="logout" type="button" aria-label="Keluar">' + ic('logout') + '</button></div>' +
        '<div class="ticket wallet">' + BOTTLE_BG + '<p class="eyebrow">Saldo poin</p><p class="big"><span class="num" id="saldo" data-v="0">0</span><span>poin</span></p>' +
        '<div class="stats"><div><b>' + gramText(me.total_gram) + '</b><span>plastik disetor</span></div><div><b>' + fmt(me.total_sesi) + '</b><span>kali setor</span></div>' +
        '<div><b>' + fmt(me.total_donasi) + '</b><span>poin didonasikan</span></div></div></div>' +
        '<div class="seg three" role="tablist">' + ['hadiah', 'donasi', 'riwayat'].map(function (t) {
          return '<button type="button" role="tab" data-t="' + t + '" aria-selected="' + (t === tab) + '">' + t.charAt(0).toUpperCase() + t.slice(1) + '</button>';
        }).join('') + '</div><div id="tab"></div>');
      countUp(q('#saldo'), me.saldo, 900);
      q('#logout').addEventListener('click', function () { api('/api/akun/keluar', {}).then(function () { location.href = '/akun'; }); });
      view.querySelectorAll('[data-t]').forEach(function (b) {
        b.addEventListener('click', function () { history.replaceState(null, '', '#' + b.getAttribute('data-t')); accountView(); });
      });
      ({ hadiah: tabHadiah, donasi: tabDonasi, riwayat: tabRiwayat })[tab](q('#tab'), me);
    });
  }

  function tabHadiah(el, me) {
    el.innerHTML = '<div class="sk" style="height:76px;border-radius:20px"></div>';
    api('/api/hadiah').then(function (d) {
      el.innerHTML = '<div class="stack"><p class="note">' + ic('info') + '<span>Hadiah simulasi untuk pengembangan. Kode yang keluar berawalan DEMO dan tidak berlaku di toko.</span></p><ul class="rows">' +
        d.hadiah.map(function (r, i) {
          var can = me.saldo >= r.cost && r.stok > 0, p = Math.min(1, me.saldo / r.cost);
          return '<li class="row-item" style="animation-delay:' + (i * .05) + 's"><span class="r-ic">' + ic('gift') + '</span><span><b>' + esc(r.nama) + '</b><span class="d">' + esc(r.ringkas) + '</span>' +
            (can ? '' : '<div class="bar"><i style="transform:scaleX(' + p.toFixed(3) + ')"></i></div><span class="d">' + (r.stok > 0 ? 'Kurang ' + fmt(r.cost - me.saldo) + ' poin' : 'Stok habis') + '</span>') +
            '</span><span class="cost">' + fmt(r.cost) + '<small>poin</small>' + (can ? '<button class="btn btn-lime" type="button" data-r="' + r.id + '">Tukar</button>' : '') + '</span></li>';
        }).join('') + '</ul></div>';
      el.querySelectorAll('[data-r]').forEach(function (b) {
        b.addEventListener('click', function () {
          var r = d.hadiah.filter(function (x) { return x.id === Number(b.getAttribute('data-r')); })[0];
          sheet('<h2 id="sheetTitle" style="font-size:26px">Tukar ' + esc(r.nama) + '?</h2><p class="lead">' + fmt(r.cost) + ' poin dipotong dari saldo. Sisa saldo <b>' + fmt(me.saldo - r.cost) + ' poin</b>.</p>' +
            '<p class="err" role="alert"></p><div class="btn-row"><button class="btn btn-quiet" type="button" data-close>Batal</button><button class="btn btn-lime" type="button" id="go">Tukar sekarang</button></div>',
          function (s) {
            s.querySelector('#go').addEventListener('click', function () {
              var go = this; go.disabled = true;
              api('/api/hadiah/' + r.id + '/tukar', {}).then(function (x) {
                s.innerHTML = '<div class="done" style="padding-top:0"><span class="badge">' + ic('gift') + '</span><h2 id="sheetTitle" style="font-size:26px">' + esc(x.hadiah.nama) + '</h2>' +
                  '<div class="voucher"><small>Kode voucher (simulasi)</small><span class="mono">' + esc(x.hadiah.kode) + '</span><small>Tunjukkan ke petugas. Kode tersimpan di Riwayat.</small></div>' +
                  '<button class="btn btn-block" type="button" data-close>Selesai</button></div>';
                hydrateIcons(s); bindClose(); loadMe();
              }).catch(function (e) { s.querySelector('.err').textContent = e.message; go.disabled = false; });
            });
          });
        });
      });
    });
  }

  function tabDonasi(el, me) {
    el.innerHTML = '<div class="sk" style="height:76px;border-radius:20px"></div>';
    api('/api/hadiah').then(function (d) {
      var wrap = document.createElement('div'); el.innerHTML = ''; el.appendChild(wrap);
      pickCampaign(wrap, d.kampanye, 'Pilih jumlah poin', function (cid, btn) {
        btn.disabled = false;
        var c = d.kampanye.filter(function (x) { return x.id === cid; })[0], amt = Math.min(100, me.saldo);
        var opts = [50, 100, 250].filter(function (v) { return v < me.saldo; }).concat([me.saldo]);
        sheet('<h2 id="sheetTitle" style="font-size:26px">Donasi ke ' + esc(c.nama) + '</h2><p class="lead">Saldo ' + fmt(me.saldo) + ' poin.</p>' +
          (me.saldo < 10 ? '<p class="err">Saldo belum cukup (minimal 10 poin).</p><button class="btn btn-quiet" type="button" data-close>Tutup</button>' :
            '<div class="amounts">' + opts.map(function (v) { return '<button type="button" data-a="' + v + '" aria-pressed="' + (v === amt) + '">' + (v === me.saldo ? 'Semua (' + fmt(v) + ')' : fmt(v)) + '</button>'; }).join('') + '</div>' +
            '<p class="err" role="alert"></p><div class="btn-row"><button class="btn btn-quiet" type="button" data-close>Batal</button><button class="btn btn-lime" type="button" id="go">Donasikan ' + fmt(amt) + '</button></div>'),
        function (s) {
          s.querySelectorAll('[data-a]').forEach(function (b) {
            b.addEventListener('click', function () {
              amt = Number(b.getAttribute('data-a'));
              s.querySelectorAll('[data-a]').forEach(function (x) { x.setAttribute('aria-pressed', x === b ? 'true' : 'false'); });
              s.querySelector('#go').textContent = 'Donasikan ' + fmt(amt);
            });
          });
          var go = s.querySelector('#go'); if (!go) return;
          go.addEventListener('click', function () {
            go.disabled = true;
            api('/api/akun/donasi', { campaign_id: cid, points: amt }).then(function () { closeSheet(); loadMe().then(function () { donated(amt, c.nama); }); })
              .catch(function (e) { s.querySelector('.err').textContent = e.message; go.disabled = false; });
          });
        });
      });
    });
  }

  function tabRiwayat(el, me) {
    var h = me.riwayat;
    el.innerHTML = h.length ? '<ul class="hist">' + h.map(function (x, i) {
      var plus = x.delta > 0;
      var t = x.kind === 'setor' ? 'Setor botol' : x.kind === 'tukar' ? esc(x.note) : 'Donasi · ' + esc(x.note);
      var s = when(x.ts) + (x.kind === 'tukar' ? ' · <span class="mono">' + esc(x.ref) + '</span>' : x.kind === 'setor' ? ' · kode ' + esc(x.ref) : '');
      return '<li style="animation-delay:' + Math.min(i, 8) * .04 + 's"><span class="h-t">' + t + '</span><span class="h-s">' + s + '</span><span class="h-v' + (plus ? ' plus' : '') + '">' + (plus ? '+' : '−') + fmt(Math.abs(x.delta)) + '</span></li>';
    }).join('') + '</ul>' : '<div class="empty">' + ic('history') + '<b>Belum ada riwayat.</b><span>Setor botol di mesin, lalu pindai QR di layar untuk menyimpan poin ke sini.</span></div>';
  }

  /* ------------------------------------------------ lembar bawah */
  var sh = document.getElementById('sheet'), bg = document.getElementById('sheetBg');
  function sheet(html, after) {
    sh.innerHTML = html; sh.hidden = false; bg.hidden = false; hydrateIcons(sh); bindClose(); if (after) after(sh);
    var f = sh.querySelector('button:not([data-close])') || sh.querySelector('button'); if (f) f.focus();
  }
  function bindClose() { sh.querySelectorAll('[data-close]').forEach(function (b) { b.addEventListener('click', function () { closeSheet(); if (location.pathname === '/akun') accountView(); }); }); }
  function closeSheet() { sh.hidden = true; bg.hidden = true; sh.innerHTML = ''; }
  bg.addEventListener('click', closeSheet);
  document.addEventListener('keydown', function (e) { if (e.key === 'Escape' && !sh.hidden) closeSheet(); });

  /* ------------------------------------------------ router */
  var p = location.pathname.replace(/\/+$/, '');
  var m = p.match(/^\/k\/([0-9A-Za-z-]+)$/);
  if (m) claimView(m[1].toUpperCase().replace('-', ''));
  else if (p === '/k') { loadMe(); codeView(); }
  else accountView();
})();
