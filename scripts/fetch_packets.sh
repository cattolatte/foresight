#!/bin/bash
# Packet-level tables. The statement requires both feature levels and names TTL
# variance, window size, fragment flags and retransmission counts -- none of
# which exist in the flow table. HuggingFace throttles large unauthenticated
# transfers, so this resumes rather than restarting.
set -u
cd "$(dirname "$0")/.."
B=https://huggingface.co/datasets/rdpahalavan/CIC-IDS2017/resolve/main/Packet-Bytes
mkdir -p data/packets
for i in 1 2 3 4; do
  T=data/packets/packets_$i.parquet
  [ -f "$T" ] && continue
  for try in $(seq 1 40); do
    have=$(stat -f%z "$T.part" 2>/dev/null || echo 0)
    curl -sSL -C - --retry 3 --speed-limit 10000 --speed-time 120 \
         -o "$T.part" "$B/Packet_Bytes_File_$i.parquet" && mv "$T.part" "$T" && break
    echo "shard $i attempt $try stopped at $((have/1000000)) MB"
  done
  echo "shard $i done"
done
echo "PACKETS DONE"
