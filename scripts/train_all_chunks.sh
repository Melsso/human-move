#!/usr/bin/env bash
set -uo pipefail

BUCKETS="${1:-}"
EPOCHS="${2:-10}"

make sync

if [ -z "$BUCKETS" ]; then
    bucket_list=$(ls data/processed/ 2>/dev/null \
        | grep -E '^bucket_.+_[0-9]+\.npz$' \
        | sed -E 's/^bucket_(.+)_[0-9]+\.npz$/\1/' \
        | sort -u)
else
    bucket_list=$(echo "$BUCKETS" | tr ',' '\n')
fi

if [ -z "$bucket_list" ]; then
    echo "error: no bucket_<name>_<chunk>.npz files found under data/processed/ -- run 'make prepare' first" >&2
    exit 1
fi

echo "training buckets: $(echo "$bucket_list" | tr '\n' ' ')"

any_bucket_trained=0

for bucket in $bucket_list; do
    echo ""
    echo "########## bucket: $bucket ##########"

    chunk_numbers=$(ls data/processed/ 2>/dev/null \
        | grep -E "^bucket_${bucket}_[0-9]+\.npz$" \
        | sed -E "s/^bucket_${bucket}_([0-9]+)\.npz$/\1/" \
        | sort -n)

    if [ -z "$chunk_numbers" ]; then
        echo "no chunk files found for bucket $bucket, skipping"
        continue
    fi

    resume_arg=""
    last_checkpoint=""

    for chunk in $chunk_numbers; do
        npz="data/processed/bucket_${bucket}_${chunk}.npz"
        out_dir="training/checkpoints/bucket_${bucket}_${chunk}"

        echo ""
        echo "=== bucket $bucket, chunk $chunk ==="
        uv run --package chess-training python -m chess_training.train \
            "$npz" --out-dir "$out_dir" \
            --epochs "$EPOCHS" --batch-size 256 --lr 1e-3 \
            $resume_arg

        checkpoint="$out_dir/best.pt"
        if [ ! -f "$checkpoint" ]; then
            echo "error: expected $checkpoint to exist after training bucket $bucket chunk $chunk, aborting" >&2
            exit 1
        fi
        resume_arg="--resume-from $checkpoint"
        last_checkpoint="$checkpoint"
    done

    echo ""
    echo "=== bucket $bucket final checkpoint: $last_checkpoint ==="
    any_bucket_trained=1
done

if [ "$any_bucket_trained" -eq 0 ]; then
    echo "error: none of the requested buckets had any chunk files to train on" >&2
    exit 1
fi