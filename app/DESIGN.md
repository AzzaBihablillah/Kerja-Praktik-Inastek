# Design — Tukar Botol (app/)

> Sumber kebenaran tipografi, warna, gerak, dan tata letak untuk kiosk, halaman HP, dan panel operator.
> Baca ini SEBELUM mengubah `app/static/`. (Diletakkan di `app/`, bukan akar repo, karena aplikasi ini disalin utuh ke Jetson Nano.)

## Arah estetika
Label daur ulang yang tegas: kertas krem, tinta hijau tua, satu aksen lime. Angka poin besar dan berat agar terbaca dari 1–2 m, gaya tiket/struk (garis pemisah, perforasi), tanpa kartu bersarang.

## Dial
- DESIGN_VARIANCE: 7 / 10 (kiosk harus tetap terbaca cepat)
- MOTION_INTENSITY: 5 / 10 (Jetson Nano: hanya transform/opacity, tanpa blur/backdrop-filter)
- VISUAL_DENSITY: 4 / 10

## Huruf
- Display: Bricolage Grotesque 500–800 (judul, angka poin; `font-variant-numeric: tabular-nums`)
- Body: Geist 400–700
- Mono: Geist Mono (kode klaim, kode voucher)
- Dimuat dari Google Fonts; cadangan "Trebuchet MS"/"Segoe UI" bila luring.
Dilarang: Inter, Roboto, Arial sebagai huruf utama; serif.

## Warna (hex, bukan oklch: Chromium lama di Nano)
```css
--paper:#f3efe4  --paper-2:#e9e4d5  --paper-3:#ddd6c4   /* latar */
--ink:#16201b    --ink-2:#2c3832    --muted:#56635b      /* teks (muted ≥ 4.5:1 di paper) */
--line:#d3ccb9
--lime:#c4f03c   /* SATU aksen: tombol utama, poin, status diterima */
--coral:#e8603f  /* semantik: ditolak / galat saja, bukan aksen */
--panel:#1d2a23  /* panel gelap (statistik, tiket poin, dompet) */
```
Teks di atas lime/coral selalu `--ink`. Bayangan diwarnai hijau tinta (`rgba(22,32,27,…)`), tidak pernah hitam murni. Tanpa gradien ungu-biru, tanpa teks bergradien.

## Gerak
- Easing: `cubic-bezier(.16,1,.3,1)`; "snap" untuk kartu keputusan/tiket: `cubic-bezier(.3,1.3,.5,1)`.
- Masuk berurutan `.rise` (jeda 0,06–0,08 s). Angka poin `countUp` (ease-out quart).
- Loop terus-menerus hanya: botol hiasan (drift 9 s), halo tombol Mulai, garis pindai kamera.
- `prefers-reduced-motion` mematikan semuanya.

## Tata letak
- Kiosk lanskap tanpa gulir, 1024×600 s.d. 1920×1080; ukuran mengikuti `vh` lewat `clamp()`. Layar ≤ 640 px tinggi menyembunyikan teks sekunder.
- Layar awal: split 1,55 : 1 (teks kiri, panel statistik gelap kanan). Aturan = daftar bernomor 3 kolom bergaris, bukan 3 kartu.
- Sesi: kamera 1,5 : 1 sisi; kartu keputusan besar muncul di bawah kamera 2,6 s (diterima) / 4,2 s (ditolak).
- Ringkasan = alur bertahap (`data-step`): kepala (judul + poin besar) tetap, di bawahnya satu langkah saja:
  `pilih` (dua kartu besar setara: lime "Ambil poin untuk saya", panel gelap "Donasikan ke yayasan") → `ambil` (3 langkah bernomor + QR) | `donasi` (3 kartu yayasan + tombol konfirmasi) → `selesai` (lencana + progres yayasan). Kaki = hitung mundur yang selalu menyebut akibatnya (mis. "poin otomatis didonasikan ke …").
  Donasi tidak boleh tersembunyi sebagai opsi kecil; dua pilihan harus sama besar.
  Waktu habis tidak pernah memindahkan poin: langkah `titip` ("Poinmu tetap aman" + kode klaim), dan langkah `hp` ("Lanjutkan di HP", 180 s) saat HP sudah membuka QR.
- HP: satu kolom, maks 480 px, gutter 18 px, sasaran sentuh ≥ 44 px, lembar bawah untuk konfirmasi.

## Komponen
`.btn` (pil; `.btn-lime` utama, `.btn-ghost`, `.btn-quiet`), `.chip`, `.bar` (progres scaleX), `.ticket` (tiket poin gelap), `.choice` (pilihan besar HP), `.row-item`, `.sheet`, `.sk` (skeleton). Ikon garis buatan sendiri di `icons.js` (stroke 1,75). Tanpa emoji.

## Suara & isi
- Bahasa Indonesia sehari-hari, kalimat pendek, kata kerja konkret ("Mulai setor botol", "Simpan ke akun", "Donasikan 60 poin").
- Pesan tolak selalu berisi apa yang salah + apa yang harus dilakukan.
- Semua hadiah/kampanye ditandai simulasi; kode voucher berawalan `DEMO-`.
- Bunyi WebAudio pendek: naik = diterima, turun = ditolak (tanpa berkas audio).

## Aksesibilitas
Kontras teks ≥ 4.5:1, `:focus-visible` di semua kontrol, label nyata di formulir, `aria-live` untuk hasil botol dan galat.

## Terakhir diperbarui
2026-10-21 — pembuatan awal kiosk, halaman HP, panel operator.
2026-10-22 — ringkasan sesi diubah jadi dua pilihan besar (ambil / donasi ke yayasan) atas masukan user: donasi sebelumnya kurang terlihat.
2026-10-22 — waktu habis tidak lagi mendonasikan poin (langkah `titip` & `hp`), kode klaim 7 hari.
