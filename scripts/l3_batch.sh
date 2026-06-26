#!/bin/bash
# L3 批量回填 — count=10, 从 start=165 跑到 start=255
# 每个 batch 10 天, 9 批覆盖 2025-06~10 缺口
set -e
PYTHON=/Users/mariusto/project/quant/.venv/bin/python
DIR=/Users/mariusto/project/superquant

for start in 165 175 185 195 205 215 225 235 245 255; do
    echo "=== $(date '+%H:%M:%S') L3 start=$start count=10 ==="
    cd $DIR && $PYTHON ml/build_features.py --l3 --start $start --count 10
    echo "=== $(date '+%H:%M:%S') batch $start done ==="
done
echo "=== $(date '+%H:%M:%S') ALL DONE ==="
