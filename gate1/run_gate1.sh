#!/usr/bin/env bash
# Snapshot Engine Gate 1 on one GPU. Run stages one at a time, in order, and look at each result:
#
#   ./gate1/run_gate1.sh test        # 2-4 min, CPU: every script end to end on tiny random models
#   ./gate1/run_gate1.sh count       # 1-2 min: how many captioned clips are long enough (pick FRAMES)
#   ./gate1/run_gate1.sh data        # hours: VAE latents + T5 text cache (resumable; rerun if interrupted)
#   ./gate1/run_gate1.sh unmixer     # ~1-3 h: learned code + unmixer, and the learned compressor (arm c)
#   ./gate1/run_gate1.sh zeroshot    # minutes: STOP here if it prints FAIL
#   ./gate1/run_gate1.sh train       # ~7-12 h: fine-tune Wan 1.3B on snapshot sequences
#   ./gate1/run_gate1.sh control     # optional, same cost: fine-tune on ordinary frames (same token count)
#   ./gate1/run_gate1.sh sample      # ~3 h: snapshot model, stock Wan 50 steps, stock Wan at equal compute
#   ./gate1/run_gate1.sh eval        # ~10-20 min: speed, seams, motion, CLIP score, side-by-side videos
#
# Settings come from environment variables (defaults below), e.g.
#   FRAMES=49 MAX_CLIPS=15000 ./gate1/run_gate1.sh data
# Every stage logs to $RUNS/logs/<stage>.log.
set -euo pipefail
cd "$(dirname "$0")/.."                      # the snapshot_gate folder (gate1/ and snapshot_gate.py live here)
ulimit -n 65536 2>/dev/null || true

MODEL=${MODEL:-Wan-AI/Wan2.1-T2V-1.3B-Diffusers}
VIDEOS=${VIDEOS:-clips}
CAPTIONS=${CAPTIONS:-openvid/data/train/OpenVid-1M.csv}
DATA=${DATA:-gate1_data}
RUNS=${RUNS:-gate1_runs}
EVAL=${EVAL:-gate1_eval}
FRAMES=${FRAMES:-81}
MAX_CLIPS=${MAX_CLIPS:-20000}
STEPS=${STEPS:-12500}
GRAD_ACCUM=${GRAD_ACCUM:-8}
N_EVAL=${N_EVAL:-100}
SAMPLE_STEPS=${SAMPLE_STEPS:-50}
DEVICE=${DEVICE:-cuda}

mkdir -p "$RUNS/logs" "$EVAL"
log() { tee -a "$RUNS/logs/$1.log"; }

for stage in "$@"; do
  echo "=== stage: $stage ($(date -u +%H:%M) UTC) ==="
  case "$stage" in
    test)
      python gate1/test_gate1.py 2>&1 | log test ;;
    count)
      python gate1/prepare_data.py --stage count --videos "$VIDEOS" --captions "$CAPTIONS" --out "$DATA" 2>&1 | log count ;;
    data)
      python gate1/prepare_data.py --videos "$VIDEOS" --captions "$CAPTIONS" --out "$DATA" --model "$MODEL" \
        --frames "$FRAMES" --max-clips "$MAX_CLIPS" --n-eval "$N_EVAL" --device "$DEVICE" 2>&1 | log data ;;
    unmixer)
      python gate1/train_unmixer.py --data "$DATA" --arm learned    --out "$RUNS/unmixer_learned.pt" --device "$DEVICE" 2>&1 | log unmixer
      python gate1/train_unmixer.py --data "$DATA" --arm learned_ae --out "$RUNS/unmixer_ae.pt"      --device "$DEVICE" 2>&1 | log unmixer ;;
    zeroshot)
      python gate1/zero_shot.py --data "$DATA" --unmixer "$RUNS/unmixer_learned.pt" --ae "$RUNS/unmixer_ae.pt" \
        --model "$MODEL" --out "$RUNS/zero_shot.json" --device "$DEVICE" 2>&1 | log zeroshot ;;
    train)
      python gate1/train.py --data "$DATA" --unmixer "$RUNS/unmixer_learned.pt" --mode snapshot --model "$MODEL" \
        --out "$RUNS/wan_snapshot" --steps "$STEPS" --grad-accum "$GRAD_ACCUM" --resume --device "$DEVICE" 2>&1 | log train ;;
    control)
      python gate1/train.py --data "$DATA" --mode frames --model "$MODEL" \
        --out "$RUNS/wan_frames" --steps "$STEPS" --grad-accum "$GRAD_ACCUM" --resume --device "$DEVICE" 2>&1 | log control ;;
    sample)
      P="$DATA/eval_prompts.jsonl"
      python gate1/sample.py --system snapshot --model "$MODEL" --transformer "$RUNS/wan_snapshot/final" \
        --unmixer "$RUNS/unmixer_learned.pt" --prompts "$P" --n "$N_EVAL" --num-frames "$FRAMES" \
        --steps "$SAMPLE_STEPS" --out "$EVAL/snapshot" --device "$DEVICE" 2>&1 | log sample
      python gate1/sample.py --system wan --model "$MODEL" --prompts "$P" --n "$N_EVAL" --num-frames "$FRAMES" \
        --steps "$SAMPLE_STEPS" --out "$EVAL/wan50" --device "$DEVICE" 2>&1 | log sample
      # equal compute: stock Wan with as many steps as fit in the snapshot model's denoising time
      EQ=$(python -c "import json; s=json.load(open('$EVAL/snapshot/summary.json')); w=json.load(open('$EVAL/wan50/summary.json')); print(max(1, round($SAMPLE_STEPS * s['denoise_s'] / w['denoise_s'])))")
      echo "equal-compute steps for stock Wan: $EQ" | log sample
      python gate1/sample.py --system wan --model "$MODEL" --prompts "$P" --n "$N_EVAL" --num-frames "$FRAMES" \
        --steps "$EQ" --out "$EVAL/wan_eq" --device "$DEVICE" 2>&1 | log sample ;;
    eval)
      python gate1/evaluate.py --runs snapshot="$EVAL/snapshot" wan50="$EVAL/wan50" wan_eq="$EVAL/wan_eq" \
        --reference wan50 --equal-compute wan_eq --clip --out "$EVAL/report" 2>&1 | log eval ;;
    *)
      echo "unknown stage '$stage' (test count data unmixer zeroshot train control sample eval)"; exit 1 ;;
  esac
done
