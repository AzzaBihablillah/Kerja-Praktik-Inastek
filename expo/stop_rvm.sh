#!/bin/bash
# Hentikan RVM dengan aman: layar kiosk ditutup, lalu aplikasi berhenti (aplikasi mematikan konveyor & menutup PLC saat berhenti).
DIR=/home/jetson/rvm/expo; mkdir -p "$DIR/log"
echo "== Stop RVM $(date '+%F %T')" >> "$DIR/log/stop.txt"
pkill -f "chromium-browser.*localhost:8000" 2>/dev/null
docker exec rvm_expo pkill -TERM -f "python3 -m app" 2>/dev/null
for i in $(seq 1 20); do
    docker exec rvm_expo pgrep -f "python3 -m app" >/dev/null 2>&1 || break
    sleep 1
done
docker exec rvm_expo pkill -KILL -f "python3 -m app" 2>/dev/null     # cadangan bila tidak berhenti dalam 20 dtk
echo "   selesai $(date '+%T')" >> "$DIR/log/stop.txt"
zenity --info --width=300 --timeout=5 --text="RVM sudah berhenti." 2>/dev/null || notify-send "RVM" "RVM sudah berhenti."
