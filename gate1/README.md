# Snapshot Engine, Gate 1: generate snapshots with a fine-tuned Wan 2.1 T2V 1.3B

Gate 0 showed that C = 4 latent frames can be recovered from one coded snapshot almost as well as
from a learned compressor. Gate 1 asks the real question: **can a video model generate snapshots well
enough that the unmixed videos look close to stock Wan, at several times less denoising compute?**

Everything runs on one GPU (developed for a single MI300X, 192 GB). Unzip this folder inside your
`snapshot_gate` directory, next to `snapshot_gate.py` (the scripts import the code and unmixer classes from it).

## How a clip is represented

The Wan VAE turns an 81-frame, 480x832 video into 21 latent frames of 16 x 60 x 104. Latent frame 0
encodes video frame 0 on its own, so it passes through uncoded; the other 20 are coded in groups of 4:

| Video | Latent frames | What the transformer sees | Tokens | Stock Wan tokens |
|---|---|---|---|---|
| 81 frames, 5 s | 21 = z0 + 20 | z0 + 5 snapshots = 6 frames | 9,360 | 32,760 |
| 49 frames, 3 s | 13 = z0 + 12 | z0 + 3 snapshots = 4 frames | 6,240 | 20,280 |

Snapshots are scaled per channel to unit variance before diffusion (a linear map, so Gaussian noise stays
Gaussian). The transformer itself is not modified.

## Files

| File | Step | What it does |
|---|---|---|
| `test_gate1.py` | 0 | Runs every script end to end on CPU with tiny random Wan models (offline, 2-4 min) |
| `prepare_data.py` | 2 | Captioned clips -> Wan-normalised VAE latents + cached T5 embeddings + eval prompts (resumable) |
| `train_unmixer.py` | 3 | Learned code + unmixer (Gate 0 recipe) in Wan's latent space, and the learned compressor |
| `zero_shot.py` | 1 | Stock Wan's loss on frames vs snapshots vs learned-compressor latents vs plain subsampling |
| `train.py` | 5 | Fine-tunes Wan 1.3B on snapshot sequences (or ordinary frames, as a control) |
| `sample.py` | 6 | Generates eval videos and times text / denoise / unmix / decode separately |
| `evaluate.py` | 7 | Speed-ups, seam ratio, motion, CLIP score, decision checks, side-by-side videos |
| `common.py` | | Shared helpers: coding, scaling, unmixing, Wan normalisation, video I/O |
| `run_gate1.sh` | | Runs each stage with sensible defaults and logs to `gate1_runs/logs/` |

## 0. Setup (once)

```bash
cd ~/users/danny/snapshot_gate && source venv/bin/activate
unzip gate1.zip                                   # creates gate1/ next to snapshot_gate.py
pip install -U "diffusers>=0.35" transformers accelerate sentencepiece ftfy av pillow
python -c "import torch; print(torch.version.hip, torch.cuda.is_available())"   # must still show HIP + True

# Wan 2.1 T2V 1.3B in diffusers format (transformer, VAE, T5 text encoder, tokenizer; tens of GB)
hf download Wan-AI/Wan2.1-T2V-1.3B-Diffusers --local-dir models/Wan2.1-T2V-1.3B-Diffusers
export MODEL=models/Wan2.1-T2V-1.3B-Diffusers

# OpenVid captions (separate from the video zips). List the CSVs, then download the main one:
python -c "from huggingface_hub import list_repo_files as l; print([f for f in l('nkp37/OpenVid-1M', repo_type='dataset') if f.endswith('.csv')])"
hf download nkp37/OpenVid-1M data/train/OpenVid-1M.csv --repo-type dataset --local-dir openvid
export CAPTIONS=openvid/data/train/OpenVid-1M.csv

./gate1/run_gate1.sh test                         # must end with ALL GATE 1 SCRIPTS RAN END TO END
```

If `pip install` replaced your ROCm torch (the check prints `None False`), reinstall it from the ROCm index
as before.

## Run, stage by stage

Each stage is one command. Look at the output before starting the next.

```bash
./gate1/run_gate1.sh count        # how many clips have >= 81 frames; if under ~10,000, use FRAMES=49 below
./gate1/run_gate1.sh data         # or: FRAMES=49 MAX_CLIPS=20000 ./gate1/run_gate1.sh data
./gate1/run_gate1.sh unmixer      # prints held-out SNR; expect roughly Gate 0's ~13-14 dB
./gate1/run_gate1.sh zeroshot     # PASS -> continue. FAIL -> stop: codes have no generation advantage
./gate1/run_gate1.sh train        # long; resumable (rerun the same command after an interruption)
./gate1/run_gate1.sh control      # optional diagnostic, same cost as train
./gate1/run_gate1.sh sample
./gate1/run_gate1.sh eval
```

Run the long stages inside `tmux`. Use the same `FRAMES` for every stage after `data`.

### Expected time and disk on one MI300X (estimates; every script prints its own ETA)

| Stage | Time | Disk |
|---|---|---|
| data (20,000 clips at 81 frames) | about 6-12 h, mostly VAE encoding | ~85 GB latents + ~15 GB text |
| unmixer (both arms) | 1-3 h | small |
| zeroshot (200 clips) | ~10-20 min | small |
| train (12,500 steps x 8 clips = 100k clips) | ~7-12 h | ~5 GB per checkpoint + 16 GB resume state |
| control | same as train | same |
| sample (100 prompts x 3 systems) | ~3 h (stock Wan at 50 steps is about a minute per video) | small |
| eval | 10-20 min | small |

Fewer clips (`MAX_CLIPS=10000`) halves the data time; 49-frame clips cut it by ~40%.

## What each result means

**Zero-shot (step 1).** Same noise, same caption, four inputs with the same number of token frames:
`frames` (a short ordinary clip, the reference), `snapshot`, `learned_ae` (Gate 0's learned compressor) and
`subsample` (every 4th latent frame, no coding). PASS needs, averaged over noise levels,
snapshot loss <= 1.5 x frames loss **and** <= 0.9 x learned-compressor loss. The thresholds are constants at
the top of `zero_shot.py`; change them before running, not after. A FAIL means fixed codes give no head start
over learning a new compressed latent, and the cheaper route to the same speed-up is a learned temporal
compressor.

**Training (step 5).** `gate1_runs/wan_snapshot/log.jsonl` has the loss curve. With `control` you can compare
it with fine-tuning on ordinary frames at the same token count: a snapshot loss far above the control's says
snapshot space is much harder to model.

**Evaluation (step 7).** `gate1_eval/report/report.md` has, per system: seconds per stage, speed-ups vs stock
Wan at 50 steps, motion, seam ratio and CLIP score. The automatic checks (constants at the top of `evaluate.py`):

| Check | GO if |
|---|---|
| Denoising speed-up vs stock Wan, 50 steps | >= 5x |
| End-to-end speed-up | >= 3x |
| Seam ratio (jump at snapshot-group boundaries) | <= 1.10 x stock Wan's |
| Motion (mean frame difference) | >= 0.90 x stock Wan's |
| CLIP score (prompt alignment) | >= stock Wan's - 1.0 |
| CLIP score vs stock Wan at equal compute | higher |

Then complete the rule by eye and with VBench:

* Watch `gate1_eval/report/side_by_side/*.mp4` (snapshot | stock 50 steps | stock equal compute). Look for a
  flicker or jump every 16 frames (snapshot-group boundaries), smeared motion, and frozen backgrounds.
* VBench on the three `videos/` folders, in custom-input mode, with the dimensions that need no reference
  prompts: subject consistency, background consistency, motion smoothness, dynamic degree, aesthetic quality,
  imaging quality, temporal flickering (`pip install vbench`; follow its README for custom inputs).
  GO if the snapshot model's average is within 3% of stock Wan at 50 steps, its dynamic degree is at least
  90% of stock Wan's, and it beats stock Wan at equal compute.

## What to send back

`gate1_runs/logs/*.log`, `gate1_runs/zero_shot.json`, `gate1_eval/report/report.md`, and a few side-by-side
videos.
