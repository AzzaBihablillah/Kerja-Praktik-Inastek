#!/bin/bash
# Cek status RVM dalam satu jendela (ikon "4 Cek RVM"). Hanya membaca, tidak mengubah apa pun.
J=$(python3 - <<'EOF'
import json, urllib.request
try:
    h = json.loads(urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=3).read().decode())
except Exception:
    print("APLIKASI|BELUM JALAN|klik ikon 1 atau 2 dulu"); raise SystemExit
m, p = h.get("mesin", {}), h.get("plc", {})
st = m.get("status", "?")
nilai = "OK" if st == "siap" else ("TUNGGU" if st == "memuat" else "MASALAH")
print("APLIKASI|%s|status: %s %s" % (nilai, st, m.get("pesan", "") or ""))
fps = m.get("fps", 0) or 0
print("KAMERA|%s|kamera %s, %.1f gambar/detik" % ("OK" if fps > 5 else "MASALAH", m.get("kamera"), fps))
if p.get("aktif"):
    print("PLC|%s|%s" % ("OK" if p.get("terhubung") else "MASALAH", p.get("ket", "")))
else:
    print("PLC|MATI (mode uji)|konveyor tidak dikendalikan; pakai ikon 1 untuk belt")
EOF
)
LOG=$(ls -t /home/jetson/rvm/expo/log/mulai_*.txt 2>/dev/null | head -1)
SER=$(grep -m1 "talang ESP32" "$LOG" 2>/dev/null | sed 's/.*talang ESP32: //')
case "$SER" in /dev/*) T="TALANG ESP32|OK|$SER";; *) T="TALANG ESP32|TIDAK ADA|keputusan talang hanya dicatat";; esac
OC=$(dmesg 2>/dev/null | grep -c "OC ALARM")
UP=$(uptime -p)
TXT=$(printf "%s\n%s\nDAYA|%s|menyala %s; peringatan arus (OC) sejak nyala: %s (wajar bila tidak restart)\n" "$J" "$T" "OK" "$UP" "$OC")
echo "$TXT" | awk -F'|' '{printf "%-13s %-16s %s\n",$1,$2,$3}' > /tmp/cek_rvm.txt
zenity --text-info --title="Cek RVM" --width=640 --height=260 --font="monospace 10" --filename=/tmp/cek_rvm.txt 2>/dev/null || cat /tmp/cek_rvm.txt
