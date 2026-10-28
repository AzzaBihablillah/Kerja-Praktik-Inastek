/* Kiosk RVM: tiga layar (awal, sesi, ringkasan) digerakkan oleh kejadian WebSocket dari server. */
(function () {
  'use strict';
  hydrateIcons();
  var $ = function (id) { return document.getElementById(id); };
  var S = { screen: 'attract', machine: null, idleMax: 30, summarySec: 75, claim: null, pick: null, camps: [], defaultSlug: '', step: '', backSec: 75, backT0: 0, backTimer: 0, doneShown: false, donateTo: 0, claimLinkSec: 180, verdictTimer: 0, sim: false };

  /* ---------------- suara (WebAudio, tanpa berkas) ---------------- */
  var AC = null;
  function tone(seq) {
    try {
      AC = AC || new (window.AudioContext || window.webkitAudioContext)();
      var t = AC.currentTime;
      seq.forEach(function (n) {
        var o = AC.createOscillator(), g = AC.createGain();
        o.type = 'sine'; o.frequency.value = n[0];
        g.gain.setValueAtTime(0.0001, t + n[1]); g.gain.exponentialRampToValueAtTime(0.22, t + n[1] + 0.02);
        g.gain.exponentialRampToValueAtTime(0.0001, t + n[1] + n[2]);
        o.connect(g); g.connect(AC.destination); o.start(t + n[1]); o.stop(t + n[1] + n[2] + 0.05);
      });
    } catch (e) { /* tanpa audio tidak apa-apa */ }
  }
  var SND_OK = [[784, 0, .14], [1175, .1, .22]], SND_BAD = [[330, 0, .18], [262, .16, .26]], SND_START = [[523, 0, .12], [659, .09, .12], [784, .18, .2]];

  /* ---------------- jam ---------------- */
  function clock() { var d = new Date(); $('clock').textContent = String(d.getHours()).padStart(2, '0') + '.' + String(d.getMinutes()).padStart(2, '0'); }
  clock(); setInterval(clock, 10000);

  /* ---------------- layar ---------------- */
  function show(name) {
    if (S.screen === name) return;
    S.screen = name; document.body.setAttribute('data-screen', name);
    $('camImg').src = name === 'session' ? '/video_feed?t=' + Date.now() : 'data:,';   // putus aliran MJPEG di luar sesi
    if (name === 'attract') loadAttract();
    if (name !== 'summary') stopBack();
    $('devbar').hidden = !(S.sim && name === 'session');
  }

  /* ---------------- status mesin ---------------- */
  var STATUS_TEXT = { siap: 'Mesin siap', memuat: 'Menyiapkan mesin', galat_kamera: 'Kamera bermasalah', galat_model: 'Model gagal dimuat' };
  function setMachine(m) {
    S.machine = m; S.sim = !!m.sim;
    $('capChip').hidden = !m.capturer; document.body.classList.toggle('capturer', !!m.capturer);
    var ok = m.status === 'siap';
    $('machineText').textContent = (STATUS_TEXT[m.status] || m.status) + (m.sim ? ' · simulasi' : '');
    $('machineChip').querySelector('.dot').className = 'dot' + (ok ? '' : ' off');
    var b = $('btnStart'); b.disabled = !ok;
    $('btnStartText').textContent = ok ? 'Mulai setor botol' : (m.status === 'memuat' ? 'Menyiapkan mesin…' : 'Mesin belum bisa dipakai');
    $('startMsg').textContent = ok ? '' : (m.status === 'memuat' ? '' : 'Panggil petugas. ' + (m.pesan || ''));
    $('simAuto').classList.toggle('on', !!m.auto);
  }

  /* ---------------- layar awal ---------------- */
  var attractTimer = 0;
  function loadAttract() {
    clearTimeout(attractTimer);
    api('/api/kiosk').then(function (d) {
      var st = d.stats;
      var n = $('todayN'); n.classList.remove('sk'); countUp(n, st.botol_bulan_ini, 900);
      $('todaySub').textContent = st.botol_bulan_ini
        ? '≈ ' + gramText(st.gram_bulan_ini) + ' plastik tidak berakhir di TPA. Total ' + fmt(st.botol_total) + ' botol sejak mesin dipasang.'
        : (st.botol_total ? 'Belum ada botol bulan ini. Total ' + fmt(st.botol_total) + ' botol sejak mesin dipasang.' : 'Jadilah penyetor pertama bulan ini.');
      Object.keys(d.poin).forEach(function (k) { var el = document.querySelector('[data-p="' + k + '"]'); if (el) el.textContent = d.poin[k]; });
      S.camps = d.kampanye;
      $('camps').innerHTML = d.kampanye.map(function (c) {
        var p = Math.min(1, c.terkumpul / c.target);
        return '<li><div class="row"><b>' + esc(c.nama) + '</b><span>' + fmt(c.terkumpul) + ' / ' + fmt(c.target) + '</span></div>' +
          '<div class="bar"><i style="transform:scaleX(' + p.toFixed(3) + ')"></i></div></li>';
      }).join('');
      var f = st.terbaru;
      $('feed').innerHTML = f.length ? f.slice(0, 5).map(function (b, i) {
        var ok = b.decision === 'terima';
        var t = ok ? 'Botol ' + b.size + ' diterima · +' + b.points + (b.bonus_merek ? ' (termasuk bonus Coca-Cola)' : '') : 'Ditolak: ' + ({ tutup: 'masih bertutup', cairan: 'masih berisi', tutup_cairan: 'bertutup & berisi', leher: 'leher tak terlihat', lebih_dari_satu: 'lebih dari satu benda' }[b.reason] || 'tidak sah');
        return '<li style="animation-delay:' + (i * 0.06) + 's"><span class="fi' + (ok ? '' : ' bad') + '">' + ic(ok ? 'check' : 'x') + '</span><span class="ft">' + esc(t) + '</span><time>' + ago(b.ts) + '</time></li>';
      }).join('') : '<li class="empty">Belum ada botol yang masuk. Aktivitas setor akan muncul di sini.</li>';
      setMachine(d.mesin);
    }).catch(function () { /* coba lagi pada siklus berikut */ });
    attractTimer = setTimeout(function () { if (S.screen === 'attract') loadAttract(); }, 20000);
  }

  $('btnStart').addEventListener('click', function () {
    var b = this; b.disabled = true; tone(SND_START);
    api('/api/sesi/mulai', {}).catch(function (e) { $('startMsg').textContent = e.message; b.disabled = false; });
  });

  /* ---------------- sesi ---------------- */
  function resetSession(s) {
    $('tickets').innerHTML = ''; document.querySelector('.side').classList.remove('has-tickets');
    $('sPts').setAttribute('data-v', 0);
    updateCounts(s || { points: 0, valid: 0, invalid: 0, grams: 0 });
    setRing(S.idleMax);
    (s && s.bottles || []).forEach(function (b) { addTicket(b, true); });
  }
  function updateCounts(s) {
    countUp($('sPts'), s.points, 650);
    $('sValid').textContent = s.valid; $('sInvalid').textContent = s.invalid; $('sGram').textContent = gramText(s.grams);
  }
  function setRing(left) {
    var k = Math.max(0, Math.min(1, left / S.idleMax));
    $('ringFg').style.strokeDashoffset = (175.9 * (1 - k)).toFixed(1);
    $('ringN').textContent = left >= 100 ? Math.ceil(left / 60) + 'm' : left;   // kiosk capturer (batas 1 jam): menit, agar muat di lingkaran
    var warn = left <= 10;
    $('ring').classList.toggle('warn', warn);
    var note = $('idleNote');
    note.classList.toggle('warn', warn);
    note.textContent = warn ? 'Sesi selesai dalam ' + left + ' detik. Masukkan botol lagi untuk lanjut.'
      : (S.machine && S.machine.capturer ? 'Kiosk capturer: sesi tetap berjalan sampai ditekan Selesai (berhenti sendiri setelah ' + Math.round(S.idleMax / 60) + ' menit tanpa botol).'
        : 'Sesi berhenti sendiri bila ' + S.idleMax + ' detik tidak ada botol.');
  }
  var SIZE_ML = { kecil: '≤ 350 ml', tanggung: '351–800 ml', besar: '> 800 ml', '?': 'ukuran tak terbaca' };
  function addTicket(b, quiet) {
    var ok = b.decision === 'terima';
    var li = document.createElement('li'); if (!ok) li.className = 'bad';
    li.innerHTML = '<span class="t-ic">' + ic(ok ? 'check' : 'x') + '</span><div class="t-main"><b>' +
      esc(ok ? 'Botol ' + b.size_text.toLowerCase() : b.reason_text) + '</b><span>' + esc(ok ? SIZE_ML[b.size] : b.tip) + '</span></div>' +
      '<span class="t-pts">' + (ok ? '+' + b.points : 'ditolak') + '</span>';
    if (quiet) li.style.animation = 'none';
    var list = $('tickets'); list.insertBefore(li, list.firstChild);
    while (list.children.length > 8) list.removeChild(list.lastChild);
    document.querySelector('.side').classList.add('has-tickets');
  }
  function verdict(b) {
    var ok = b.decision === 'terima', v = $('verdict');
    v.className = 'verdict ' + (ok ? 'ok' : 'bad');
    v.innerHTML = '<span class="v-ic">' + ic(ok ? 'check' : (b.reason === 'tutup' || b.reason === 'tutup_cairan' ? 'cap' : b.reason === 'cairan' ? 'drop' : 'x')) + '</span>' +
      '<div class="v-t"><b>' + esc(ok ? 'Diterima. Botol ' + b.size_text.toLowerCase() + '.' : b.reason_text) + '</b><span>' +
      esc(ok ? (b.bonus_merek ? 'Bonus produk Coca-Cola +' + b.bonus_merek + '. ' : '') + 'Terima kasih. Lanjutkan botol berikutnya.' : b.tip) + '</span></div>' + (ok ? '<span class="v-pts">+' + b.points + '</span>' : '');
    void v.offsetWidth; v.classList.add('show');
    clearTimeout(S.verdictTimer);
    S.verdictTimer = setTimeout(function () { v.classList.remove('show'); }, ok ? 2600 : 4200);
    tone(ok ? SND_OK : SND_BAD);
  }
  $('btnFinish').addEventListener('click', function () {
    var b = this; b.disabled = true;
    api('/api/sesi/selesai', {}).catch(function () { }).then(function () { b.disabled = false; });
  });

  /* ---------------- ringkasan: pilih -> (ambil | donasi) -> selesai ---------------- */
  var STEP_SEC = { selesai: 12, kosong: 20, titip: 25 };   // pilih/ambil/donasi memakai S.summarySec, hp memakai S.claimLinkSec
  var CAMP_IC = ['leaf', 'bottle', 'drop'];
  // Judul ringkasan per langkah (copy dari kanvas Design user, 26 Okt); langkah lain memakai judul awal sesi.
  var STEP_TITLE = { ambil: 'Terima kasih, Pahlawan Lingkungan!', donasi: 'Terima kasih, Pahlawan Lingkungan!',
    hp: 'Terima kasih, Pahlawan Lingkungan!', titip: 'Terima kasih, Pahlawan Lingkungan!', selesai: 'Langkah Kecilmu, Napas Baru Bagi Bumi!' };
  function setStep(step) {
    S.step = step; $('summary').setAttribute('data-step', step);
    if (S.baseTitle) $('sumTitle').textContent = STEP_TITLE[step] || S.baseTitle;
    var b = $('btnDone');
    b.hidden = step === 'pilih' || step === 'donasi';  // dijawab lewat kartu besar; waktu habis = poin dititipkan, bukan didonasikan
    b.textContent = step === 'ambil' ? 'Sudah dipindai' : 'Selesai';
    startBack(step === 'hp' ? S.claimLinkSec : (STEP_SEC[step] || S.summarySec));
  }
  function loadCamps() {
    return api('/api/kiosk').then(function (d) { S.camps = d.kampanye; S.defaultSlug = d.kampanye_default; return d.kampanye; });
  }
  function campById(id) { return S.camps.filter(function (c) { return c.id === id; })[0]; }

  var ZERO_SUB = null;
  function showSummary(sum) {
    var s = sum.session, c = sum.claim;
    S.claim = c; S.pick = null; S.doneShown = false; S.baseTitle = '';
    if (ZERO_SUB === null) ZERO_SUB = $('zeroSub').textContent;
    if (sum.capturer) {                          // kiosk capturer: poin hanya ditampilkan, tanpa QR/klaim
      $('summary').classList.toggle('zero', !s.points);
      $('sumTitle').textContent = 'Sesi capturer selesai.';
      $('sumStats').innerHTML = ['<b>' + s.valid + '</b>diterima', '<b>' + s.invalid + '</b>ditolak', '<b>' + (s.valid + s.invalid) + '</b>botol tersimpan sebagai data']
        .map(function (x) { return '<li>' + x + '</li>'; }).join('');
      $('zeroSub').textContent = 'Mode kiosk capturer: poin hanya untuk uji alur dan tidak bisa diklaim. Frame & keputusan setiap botol tersimpan bersama label kelompok.';
      $('sumPts').setAttribute('data-v', 0);
      show('summary'); setStep('kosong');
      if (s.points) setTimeout(function () { countUp($('sumPts'), s.points, 1100); }, 250);
      return;
    }
    $('zeroSub').textContent = ZERO_SUB;
    var zero = !s.points || !c;
    $('summary').classList.toggle('zero', zero);
    $('sumTitle').textContent = zero ? (s.invalid ? 'Belum ada botol yang diterima.' : 'Belum ada botol yang masuk.') : (s.valid >= 5 ? 'Luar biasa, terima kasih.' : 'Kerja bagus, terima kasih.');
    if (!zero) S.baseTitle = $('sumTitle').textContent;
    var reasons = {};
    (s.bottles || []).forEach(function (b) { if (b.reason) reasons[b.reason] = (reasons[b.reason] || 0) + 1; });
    var t = (reasons.tutup || 0) + (reasons.tutup_cairan || 0), cr = (reasons.cairan || 0) + (reasons.tutup_cairan || 0);
    var items = ['<b>' + s.valid + '</b>botol diterima', '<b>' + gramText(s.grams) + '</b>plastik terselamatkan'];
    if (s.invalid) items.push('<b>' + s.invalid + '</b>ditolak' + (t ? ' · ' + t + ' bertutup' : '') + (cr ? ' · ' + cr + ' berisi' : '') + (reasons.bukan_botol ? ' · ' + reasons.bukan_botol + ' bukan botol' : ''));
    $('sumStats').innerHTML = items.map(function (x) { return '<li>' + x + '</li>'; }).join('');
    $('sumPts').setAttribute('data-v', 0);
    show('summary');
    if (zero) { setStep('kosong'); return; }
    setTimeout(function () { countUp($('sumPts'), s.points, 1100); }, 250);
    $('qr').innerHTML = c.qr || '<p class="qr-or" style="padding-top:40%">QR tidak tersedia. Ketik kode di bawah.</p>';
    var host = '/k';
    try { host = new URL(c.url).host + '/k'; } catch (e) { /* biarkan */ }
    $('code').textContent = c.code_fmt; $('qrHost').textContent = host;
    document.querySelectorAll('#summary [data-code]').forEach(function (el) { el.textContent = c.code_fmt; });
    document.querySelectorAll('#summary [data-host]').forEach(function (el) { el.textContent = host; });
    $('titipUntil').textContent = 'Berlaku sampai ' + new Date(c.expires * 1000).toLocaleDateString('id-ID', { weekday: 'long', day: 'numeric', month: 'long' }) +
      '. Setelah itu poin yang belum diklaim didonasikan ke program lingkungan.';
    $('donateMsg').textContent = ''; $('donateMsg').className = 'donate-msg';
    setStep('pilih');
    loadCamps().then(function () {
      renderPicks();
      if (c.status && c.status !== 'open') claimDone(c);   // sudah diklaim (mis. layar dimuat ulang)
    });
  }

  function renderPicks() {
    $('picks').innerHTML = S.camps.map(function (c, i) {
      var p = Math.min(1, c.terkumpul / c.target);
      return '<button type="button" class="pick" role="radio" aria-checked="false" data-id="' + c.id + '">' +
        '<span class="p-ic">' + ic(CAMP_IC[i % CAMP_IC.length]) + '</span><b>' + esc(c.nama) + '</b><span class="p-d">' + esc(c.ringkas) + '</span>' +
        '<div class="bar"><i style="transform:scaleX(' + p.toFixed(3) + ')"></i></div><span class="p-num">' + fmt(c.terkumpul) + ' dari ' + fmt(c.target) + ' poin</span></button>';
    }).join('');
    var btn = $('btnDonate'); btn.disabled = true; btn.className = 'btn btn-xl'; btn.textContent = 'Pilih yayasan dulu';
  }

  $('chSelf').addEventListener('click', function () { setStep('ambil'); });
  $('chGive').addEventListener('click', function () { setStep('donasi'); });
  document.querySelectorAll('[data-step-to]').forEach(function (b) {
    b.addEventListener('click', function () { S.pick = null; renderPicks(); setStep(b.getAttribute('data-step-to')); });
  });
  $('picks').addEventListener('click', function (e) {
    var p = e.target.closest('.pick'); if (!p) return;
    S.pick = Number(p.getAttribute('data-id'));
    [].forEach.call(this.children, function (x) { x.setAttribute('aria-checked', x === p ? 'true' : 'false'); });
    var btn = $('btnDonate'), c = campById(S.pick);
    btn.disabled = false; btn.className = 'btn btn-xl btn-lime';
    btn.innerHTML = 'Donasikan ' + fmt(S.claim.points) + ' poin' + ic('arrow');
    $('donateMsg').textContent = c ? 'Ke ' + c.nama : '';
  });
  $('btnDonate').addEventListener('click', function () {
    if (!S.claim || !S.pick) return;
    this.disabled = true; donate(S.pick);
  });

  function donate(cid) {                       // hanya atas pilihan pengguna; waktu habis tidak pernah mendonasikan
    S.donateTo = cid;
    return api('/api/klaim/' + S.claim.code + '/donasi', { campaign_id: cid }).then(function (d) {
      showDone('give', d.points, cid);
    }).catch(function (e) {
      $('donateMsg').textContent = e.message; $('donateMsg').className = 'donate-msg err'; $('btnDonate').disabled = false;
    });
  }
  function claimDone(c) {                      // dari HP (claim_update) atau layar dimuat ulang
    if (c.status === 'donated') {
      var camp = S.camps.filter(function (x) { return x.nama === c.campaign; })[0];
      showDone('give', c.points, camp ? camp.id : S.donateTo);
    } else if (c.status === 'claimed') showDone('self', c.points);
    else if (c.status === 'expired') show('attract');
  }
  function showDone(kind, points, cid) {
    if (S.doneShown) return;
    S.doneShown = true;
    var give = kind === 'give', camp = give ? campById(cid) : null;
    var badge = $('doneBadge'); badge.className = 'done-badge' + (give ? ' give' : ''); badge.innerHTML = ic(give ? 'heart' : 'check');
    $('doneTitle').textContent = give ? 'Terima kasih ya!' : 'Poin tersimpan.';
    $('doneSub').textContent = give
      ? fmt(points) + ' poin disalurkan ke ' + (camp ? camp.nama : 'yayasan') + '. Botolmu hari ini ikut membiayai program ini.'
      : fmt(points) + ' poin sudah masuk ke akunmu. Cek saldo dan tukar hadiah dari HP.';
    var box = $('doneCamp');
    box.hidden = !camp;
    if (camp) {
      var before = camp.terkumpul, after = before + points, bar = $('doneCampBar');
      $('doneCampName').textContent = camp.nama;
      $('doneCampNum').textContent = fmt(after) + ' / ' + fmt(camp.target) + ' poin';
      bar.style.transition = 'none'; bar.style.transform = 'scaleX(' + Math.min(1, before / camp.target).toFixed(3) + ')';
      void bar.offsetWidth; bar.style.transition = '';
      requestAnimationFrame(function () { bar.style.transform = 'scaleX(' + Math.min(1, after / camp.target).toFixed(3) + ')'; });
    }
    setStep('selesai');
    tone(SND_OK);
  }

  function startBack(sec) {
    stopBack(); S.backSec = sec; S.backT0 = Date.now();
    S.backTimer = setInterval(tickBack, 250); tickBack();
  }
  function tickBack() {
    var el = (Date.now() - S.backT0) / 1000, left = Math.max(0, Math.ceil(S.backSec - el));
    $('backBar').style.transform = 'scaleX(' + Math.max(0, 1 - el / S.backSec).toFixed(3) + ')';
    var note;
    if (S.step === 'pilih') note = 'Belum memilih dalam <b>' + left + ' detik</b>? Tenang, poinmu tidak hilang: kami simpan dengan kode klaim.';
    else if (S.step === 'donasi') note = 'Belum dikonfirmasi dalam <b>' + left + ' detik</b>? Poin tidak didonasikan; kami simpan dengan kode klaim.';
    else if (S.step === 'ambil') note = 'Layar QR tertutup dalam <b>' + left + ' detik</b>. Sudah memindai? Lanjutkan saja di HP, poin tetap aman.';
    else if (S.step === 'hp') note = 'Kembali ke layar awal dalam ' + left + ' detik. Proses di HP tetap bisa dilanjutkan.';
    else note = 'Kembali ke layar awal dalam ' + left + ' detik';
    $('backNote').innerHTML = note;
    if (el >= S.backSec) onTimeout();
  }
  function onTimeout() {
    stopBack();
    if ((S.step === 'pilih' || S.step === 'donasi' || S.step === 'ambil') && S.claim && !S.doneShown) {
      setStep('titip'); return;                // poin tetap di kode klaim; tunjukkan kodenya sebelum kembali ke awal
    }
    show('attract');
  }
  function stopBack() { clearInterval(S.backTimer); S.backTimer = 0; }
  $('summary').addEventListener('pointerdown', function () {
    if (S.backTimer && /^(pilih|ambil|donasi|hp)$/.test(S.step)) S.backT0 = Date.now();   // masih dipakai: tunda hitung mundur
  });
  $('btnDone').addEventListener('click', function () { show('attract'); });

  /* ---------------- simulasi ---------------- */
  function sim(body) { api('/api/sim/botol', body).catch(function () { }); }
  document.querySelectorAll('[data-sim]').forEach(function (b) { b.addEventListener('click', function () { sim(JSON.parse(b.getAttribute('data-sim'))); }); });
  $('simAuto').addEventListener('click', function () {
    var on = !this.classList.contains('on'); var b = this;
    api('/api/sim/auto', { on: on }).then(function () { b.classList.toggle('on', on); });
  });
  document.addEventListener('keydown', function (e) {
    if (!S.sim || S.screen !== 'session') return;
    var m = { '1': { size: 'kecil' }, '2': { size: 'tanggung' }, '3': { size: 'besar' }, '4': { size: 'tanggung', cap: true }, '5': { size: 'tanggung', liquid: true }, '6': { benda: true } }[e.key];
    if (m) sim(m);
    if (e.key === 'a' || e.key === 'A') $('simAuto').click();
  });

  /* ---------------- WebSocket ---------------- */
  var ws, retry = 0;
  function connect() {
    ws = new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws');
    ws.onopen = function () { retry = 0; $('offline').hidden = true; };
    ws.onclose = function () { $('offline').hidden = false; setTimeout(connect, Math.min(8000, 800 * Math.pow(1.6, retry++))); };
    ws.onmessage = function (m) {
      var ev = JSON.parse(m.data);
      switch (ev.type) {
        case 'init':
          S.idleMax = ev.idle_max; S.summarySec = ev.summary_sec; S.claimLinkSec = ev.claim_link_sec || S.claimLinkSec; setMachine(ev.machine);
          if (ev.session) { resetSession(ev.session); show('session'); }
          else if (ev.summary) showSummary(ev.summary);
          else { S.screen = ''; show('attract'); }
          break;
        case 'machine': setMachine(ev); break;
        case 'session_start': S.idleMax = ev.idle_max || S.idleMax; resetSession(ev.session); show('session'); break;
        case 'bottle':
          if (S.screen !== 'session') { resetSession(ev.session); show('session'); } else { updateCounts(ev.session); addTicket(ev.bottle); }
          verdict(ev.bottle); break;
        case 'tick': setRing(ev.idle_left); $('camInfo').textContent = 'Kamera periksa · ' + ev.fps + ' fps'; break;
        case 'session_end': showSummary(ev); break;
        case 'claim_update': if (S.claim && ev.claim && ev.claim.code === S.claim.code && S.screen === 'summary') claimDone(ev.claim); break;
        case 'claim_opened':                       // HP membuka tautan QR: tunggu lebih lama, proses lanjut di HP
          if (S.claim && ev.code === S.claim.code && S.screen === 'summary' && !S.doneShown && S.step !== 'hp') { setStep('hp'); tone(SND_START); }
          break;
      }
    };
  }
  connect();
  loadAttract();
})();
