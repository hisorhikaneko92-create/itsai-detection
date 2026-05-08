#!/usr/bin/env bash
set -u

TARGET=9000
OUT=data/MainData/by_source/_new_pure_human_cc_9k.csv
MAX_ITERS=80   # safety cap

for i in $(seq 1 $MAX_ITERS); do
    # Disk check before each iteration — abort if low
    AVAIL_GB=$(df --output=avail -BG /root | tail -1 | tr -d 'G ')
    if [ "$AVAIL_GB" -lt 20 ]; then
        echo "ABORT: only ${AVAIL_GB}G free on /root, need >=20G"
        break
    fi

    # Current row count (existing - 1 for header)
    if [ -f "$OUT" ]; then
        CURRENT=$(($(wc -l < "$OUT") - 1))
    else
        CURRENT=0
    fi
    
    if [ "$CURRENT" -ge "$TARGET" ]; then
        echo "=== TARGET HIT: $CURRENT / $TARGET. Stopping. ==="
        break
    fi
    
    echo ""
    echo "=== iter $i: have $CURRENT/$TARGET, ${AVAIL_GB}G free ==="
    
    # Vary seed per iteration so we sample different segments
    python scripts/fetch_cc_pure_human.py \
        --output "$OUT" \
        --target "$TARGET" \
        --num-segments 1 \
        --seed $((42 + i))
    
    # Aggressive cache cleanup
    echo "--- cleaning cache ---"
    rm -rf cc_net/cache/*.warc.wet.gz cc_net/tmp_segments/* 2>/dev/null
    # If processed segments leave residue elsewhere, also wipe sub-dirs
    find cc_net/cache -name "*.gz" -size +100M -delete 2>/dev/null
    
    df -h /root | tail -1
done

if [ -f "$OUT" ]; then
    FINAL=$(($(wc -l < "$OUT") - 1))
    echo ""
    echo "=== FINAL: $FINAL rows in $OUT ==="
fi
