# Snapshot Engine, Gate 0: reconstruction, speed and KV-cache experiment

Three questions, one run:

1. **Quality.** Can C consecutive video latent frames be recovered from **one** coded snapshot, and is a
   fixed linear code (with A Aᵀ = I) competitive with a learned compressor at the same ratio?
2. **Speed.** How much transformer compute does generating snapshots instead of frames save,
   analytically at Wan-14B scale and measured on a real transformer?
3. **KV cache.** How much smaller is the KV cache for causal (streaming) video generation,
   analytically and measured from the cache tensors actually allocated?

## Files

| File | What it does |
|---|---|
| `run_all.sh` | Runs everything below in order and writes `report/report.md` and `report/benchmark.png` |
| `snapshot_gate.py` | Quality: codes, unmixer, learned-compressor baseline, data loaders, metrics, decision rule |
| `benchmark.py` | Speed and KV: analytic production model, measured step time, measured streaming KV cache, joined report |
| `test_snapshot_gate.py`, `test_benchmark.py` | 16 tests: orthonormal codes, exact projection, noise mapping, data, KV bytes = formula, decision rule |
| `merge_results.py` | Merges several `snapshot_gate.py` result files and re-applies the decision rule |
| `encode_wan_latents.py` | Turns a folder of videos into normalised Wan 2.1 VAE latents (**untested here**: no GPU or weights) |

## Quality arms (all at the same compression C: C frames → one frame-sized latent)

| Arm | Encoder | Decoder |
|---|---|---|
| `avg` | plain temporal average (no coding) | unmixer + exact projection |
| `dcwalsh` | DC-plus-Walsh code, "a Bayer filter for time" | unmixer + exact projection |
| `cacti` | CACTI-style binary mask shifted one position per frame, normalised | unmixer + exact projection |
| `randpm` | i.i.d. random ±1 code | unmixer + exact projection |
| `learned` | learned code, normalised so A Aᵀ = I still holds | unmixer + exact projection |
| `learned_ae` | learned **nonlinear, position-aware** encoder (the competitor) | same U-Net, no projection |

The learned compressor sees each position's place in a 4×4 tile, so it can learn spatially multiplexed
codes too; it is at least as expressive as every coded arm.

## Run

```bash
pip install torch numpy matplotlib imageio[pyav]
./run_all.sh                                   # synthetic data on CPU, ~30-40 min

# the real gate on Wan 2.1 VAE latents (GPU)
python encode_wan_latents.py --videos my_videos/ --out latents/ --height 480 --width 832
DATA=npy LATENTS=latents/ DEVICE=cuda STEPS=20000 SIZE=32 WIDTH=64 BATCH=32 ./run_all.sh

# measured speed on a Wan-1.3B-shaped transformer (GPU)
python benchmark.py --gate-results runs/gate0.json --out report_13b --device cuda \
    --width 1536 --layers 30 --heads 12 --tokens-per-frame 1560 --latent-frames 21
```

## Decision rule (pre-registered; edit the constants before running, not after)

GO at a given C when, for the best coded arm:

1. its SNR is within `MARGIN_VS_LEARNED_DB[C]` of `learned_ae` (0.5 / 1.0 / 1.5 / 2.0 dB for C = 2 / 4 / 8 / 16);
2. on the most-moving third of clips, its motion SNR beats `avg` by at least 3 dB (the code carries motion);
3. optionally, its SNR is above `--floor-db`.

## What the smoke run on a 2-core CPU showed (synthetic data: a pipeline check, not evidence)

32 × 32 synthetic textures, 2,000 training steps per arm, 256 held-out clips.

| C | Speed-up per clip, Wan-14B 720p (analytic) | Measured step speed-up (small transformer) | KV cache, 60 s of video, Wan-14B (analytic) | Best coded SNR | Learned compressor SNR | Averaging SNR | Gate |
|---|---|---|---|---|---|---|---|
| 1 | 1.0× | 1.0× | 708 GB | – | – | – | – |
| 4 | 6.9× | 9.5× | 177 GB | 14.1 dB | 10.7 dB | 8.1 dB | GO |
| 8 | 17.1× | 22.0× | 88 GB | 11.6 dB | 6.3 dB | 5.1 dB | NO-GO (motion check fails) |

The measured streaming KV cache shrinks exactly C-fold (28.3 MB → 7.1 MB → 3.5 MB at C = 1, 4, 8).
Absolute SNRs are low because the synthetic textures are noise-like and the models tiny. The real
answer needs Wan latents, a GPU and longer training.
