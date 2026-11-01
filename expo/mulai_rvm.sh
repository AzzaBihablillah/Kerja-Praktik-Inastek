#!/bin/bash
# Mulai RVM di Jetson Nano (Capstone Expo). Dipanggil oleh ikon desktop.
#   mulai_rvm.sh plc        -> konveyor dikendalikan PLC (pakai HANYA bila area belt aman)
#   mulai_rvm.sh tanpa_plc  -> uji tanpa menggerakkan konveyor
# Kamera & ESP32 (talang) dicari otomatis setiap kali dijalankan. Log: ~/rvm/expo/log/
MODE="${1:-plc}"
DIR=/home/jetson/rvm/expo
mkdir -p "$DIR/log"
LOG="$DIR/log/mulai_$(date +%Y%m%d_%H%M%S).txt"
exec > >(tee -a "$LOG") 2>&1
echo "== Mulai RVM $(date '+%F %T')  mode=$MODE"

pesan() { zenity --"$1" --width=420 --text="$2" 2>/dev/null || notify-send "RVM" "$2"; }

# 1) container
if ! docker start rvm_expo >/dev/null; then
    pesan error "Container Docker rvm_expo tidak bisa dijalankan.\nLihat log: $LOG"; exit 1
fi
docker exec rvm_expo pkill -f "python3 -m app" 2>/dev/null && sleep 2      # sisa sesi sebelumnya

# 2) kamera: utamakan Logitech C922, selain itu kamera pertama
CAM=""
for f in /sys/class/video4linux/video*/name; do
    [ -e "$f" ] || continue
    n=$(basename "$(dirname "$f")"); n=${n#video}
    if grep -qi "c922\|logitech" "$f"; then CAM=$n; break; fi
    [ -z "$CAM" ] && CAM=$n
done
if [ -z "$CAM" ]; then
    pesan error "Kamera belum tercolok.\nColokkan kamera USB (C922), tunggu 5 detik, lalu klik ikon lagi."; exit 1
fi
echo "kamera: /dev/video$CAM ($(cat /sys/class/video4linux/video$CAM/name 2>/dev/null))"

# 3) talang ESP32 (CP2102 = ttyUSB, sebagian papan = ttyACM). Tidak ada -> keputusan talang hanya dicatat
SER=$(ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null | head -1)
echo "talang ESP32: ${SER:-(tidak ada, hanya dicatat)}"

# 4) alamat untuk QR klaim (HP pengunjung harus satu jaringan dengan Jetson)
IP=$(hostname -I | awk '{print $1}')
echo "alamat: http://$IP:8000  (operator: http://$IP:8000/operator)"

PLC=""; [ "$MODE" = "plc" ] && PLC="--plc"
MEREK=""; E=/home/jetson/rvm/datasets/3_model_final/ekspor/merek_coca_v2_kandidat.engine
[ -f "$E" ] && MEREK=/rvm/datasets/3_model_final/ekspor/merek_coca_v2_kandidat.engine
docker exec -d -e RVM_SORTIR_SERIAL="$SER" -e RVM_BRAND_MODEL="$MEREK" rvm_expo sh -c \
    "cd /rvm && exec python3 -m app --backend tensorrt --profil nano --cam $CAM $PLC --url http://$IP:8000 >> /rvm/expo/log/app_$(date +%Y%m%d).txt 2>&1"

# 5) tunggu server siap (model dimuat di latar; layar kiosk menampilkan status)
for i in $(seq 1 120); do
    (exec 3<>/dev/tcp/127.0.0.1/8000) 2>/dev/null && break
    sleep 1
done
if ! (exec 3<>/dev/tcp/127.0.0.1/8000) 2>/dev/null; then
    pesan error "Aplikasi tidak menyala dalam 2 menit.\nLihat log: $DIR/log/app_$(date +%Y%m%d).txt"; exit 1
fi
echo "server siap $(date '+%T')"

# 6) layar kiosk penuh (keluar: Alt+F4)
pkill -f "chromium-browser.*localhost:8000" 2>/dev/null
chromium-browser --kiosk --noerrdialogs --disable-infobars --disable-session-crashed-bubble --incognito \
    --password-store=basic --lang=id --disable-features=Translate,TranslateUI "http://localhost:8000/" >/dev/null 2>&1 &
echo "== selesai memulai"
