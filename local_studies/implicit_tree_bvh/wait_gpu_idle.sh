#!/usr/bin/env bash
# Block until no other compute process uses the GPU (polls every 5 s).
while true; do
    apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
    if [ "$apps" -eq 0 ] && [ "$util" -lt 10 ]; then
        echo "GPU idle"
        exit 0
    fi
    sleep 5
done
