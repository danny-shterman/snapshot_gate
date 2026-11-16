"""Gate 1, step 2: captioned clips -> Wan-normalised VAE latents + cached T5 text embeddings.

Stages (run all, or one at a time with --stage):
  count    (separate, quick) how many captioned clips have at least 33/49/81/121 frames; use it to pick --frames
  latents  decode the first --frames frames of each captioned video, encode with the Wan 2.1 VAE,
           normalise with Wan's latents_mean / latents_std, save as float16 [16, F, H/8, W/8].
           Resumable: clips already encoded are skipped. Unreadable or too-short videos are skipped.
  text     encode every caption (plus "" for classifier-free guidance and the negative prompt)
           with Wan's UMT5 text encoder; saves the unpadded embedding per clip as float16.
  prompts  writes eval_prompts.jsonl: --n-eval held-out captions for sampling and evaluation.

Output layout (--out):
  latents/<id>.npy   text/<id>.npy   text/_empty.npy   text/_negative.npy
  manifest.jsonl     {"id", "latent", "caption", "split": "train"|"test", "video"}
  eval_prompts.jsonl

Example:
  python gate1/prepare_data.py --videos clips/ --captions openvid/data/train/OpenVid-1M.csv \
      --out gate1_data --frames 81 --max-clips 20000
"""
from __future__ import annotations
import argparse, csv, glob, json, os, random, sys, time
import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DEFAULT_MODEL, device_default, is_test_id, to_wan_space, wan_stats  # noqa: E402

NEGATIVE_PROMPT = ("blurry, low quality, distorted, deformed, static, frozen, jittery, flickering, "
                   "watermark, text, subtitles, overexposed, underexposed")


# --------------------------------------------------------------------------- captions
def load_captions(path: str, video_col: str | None, caption_col: str | None) -> dict[str, str]:
    csv.field_size_limit(10 ** 8)
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        cols = reader.fieldnames or []
        vc = video_col or next((c for c in cols if c.lower() in ("video", "video_path", "path", "file", "filename",
                                                                  "video_id", "videoid")), None)
        cc = caption_col or next((c for c in cols if c.lower() in ("caption", "text", "prompt", "description")), None)
        if not vc or not cc:
            sys.exit(f"could not find video/caption columns in {cols}; pass --video-col and --caption-col")
        caps, n_rows = {}, 0
        for row in reader:
            n_rows += 1
            name = os.path.basename(str(row[vc]).strip())
            cap = str(row[cc]).strip()
            if name and cap:
                caps[name] = cap
                caps[os.path.splitext(name)[0]] = cap
    print(f"captions: {n_rows:,} rows from {path} (video column '{vc}', caption column '{cc}')")
    return caps


# --------------------------------------------------------------------------- video
def read_frames(path: str, n: int) -> np.ndarray | None:
    """First n RGB frames as uint8 [n, H, W, 3], or None if unreadable or shorter than n."""
    import av
    try:
        frames = []
        with av.open(path) as c:
            s = c.streams.video[0]
            s.thread_type = "AUTO"
            for fr in c.decode(s):
                frames.append(fr.to_ndarray(format="rgb24"))
                if len(frames) == n:
                    break
        return np.stack(frames) if len(frames) == n else None
    except Exception:
        return None


def to_model_input(frames: np.ndarray, height: int, width: int) -> torch.Tensor:
    """uint8 [T,H,W,3] -> float [1,3,T,height,width] in [-1, 1], resized (cover) and centre-cropped."""
    x = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 127.5 - 1      # [T,3,H,W]
    H, W = x.shape[-2:]
    if (H, W) != (height, width):
        s = max(height / H, width / W)
        x = F.interpolate(x, size=(round(H * s), round(W * s)), mode="bilinear", align_corners=False, antialias=True)
        y0, x0 = (x.shape[-2] - height) // 2, (x.shape[-1] - width) // 2
        x = x[..., y0:y0 + height, x0:x0 + width]
    return x.permute(1, 0, 2, 3)[None].contiguous()


# --------------------------------------------------------------------------- stages
def stage_latents(a):
    from diffusers import AutoencoderKLWan
    dev = a.device
    caps = load_captions(a.captions, a.video_col, a.caption_col)
    vids = sorted(sum([glob.glob(os.path.join(a.videos, "**", e), recursive=True)
                       for e in ("*.mp4", "*.mov", "*.webm", "*.mkv", "*.avi")], []))
    vids = [v for v in vids if os.path.basename(v) in caps or os.path.splitext(os.path.basename(v))[0] in caps]
    random.Random(0).shuffle(vids)
    print(f"videos with captions: {len(vids):,}")
    assert (a.frames - 1) % 4 == 0, "--frames must be 4n+1 (81, 49, ...)"

    os.makedirs(os.path.join(a.out, "latents"), exist_ok=True)
    part = os.path.join(a.out, "manifest_partial.jsonl")
    done = {json.loads(l)["id"] for l in open(part)} if os.path.exists(part) else set()
    skipped = {l.strip() for l in open(part + ".skipped")} if os.path.exists(part + ".skipped") else set()
    have = len(done)
    print(f"already encoded: {have:,}; previously skipped: {len(skipped):,}")

    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[a.vae_dtype]
    vae = AutoencoderKLWan.from_pretrained(a.model, subfolder="vae", torch_dtype=dtype).to(dev).eval()
    mean, std = wan_stats(vae.config)
    t0, new = time.time(), 0
    with open(part, "a") as out, open(part + ".skipped", "a") as skip:
        for v in vids:
            if have >= a.max_clips:
                break
            cid = os.path.splitext(os.path.basename(v))[0]
            if cid in done or cid in skipped:
                continue
            fr = read_frames(v, a.frames)
            if fr is None:
                skip.write(cid + "\n"); skip.flush(); skipped.add(cid)
                continue
            x = to_model_input(fr, a.height, a.width).to(dev, dtype)
            with torch.no_grad():
                z = vae.encode(x).latent_dist.mode().float()
            z = to_wan_space(z, mean, std)[0].cpu().numpy().astype(np.float16)     # [16, F, H/8, W/8]
            rel = os.path.join("latents", cid + ".npy")
            np.save(os.path.join(a.out, rel), z)
            cap = caps.get(os.path.basename(v)) or caps[cid]
            out.write(json.dumps(dict(id=cid, latent=rel, caption=cap, video=os.path.abspath(v),
                                      split="test" if is_test_id(cid, a.test_percent) else "train")) + "\n")
            out.flush()
            have += 1; new += 1
            if new % 50 == 0:
                rate = new / (time.time() - t0)
                print(f"  encoded {have:,}/{a.max_clips:,}  ({rate:.2f} clips/s, "
                      f"~{(a.max_clips - have) / max(rate, 1e-9) / 3600:.1f} h left)  skipped {len(skipped):,}",
                      flush=True)
    # final manifest: dedupe, keep only files that exist
    rows, seen = [], set()
    for l in open(part):
        r = json.loads(l)
        if r["id"] not in seen and os.path.exists(os.path.join(a.out, r["latent"])):
            seen.add(r["id"]); rows.append(r)
    with open(os.path.join(a.out, "manifest.jsonl"), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    n_test = sum(r["split"] == "test" for r in rows)
    print(f"manifest: {len(rows):,} clips ({len(rows) - n_test:,} train, {n_test:,} test); "
          f"latent shape {np.load(os.path.join(a.out, rows[0]['latent'])).shape}")


def stage_count(a):
    """How many captioned clips are long enough for each clip length (reads container metadata only)."""
    import av
    caps = load_captions(a.captions, a.video_col, a.caption_col)
    vids = sorted(sum([glob.glob(os.path.join(a.videos, "**", e), recursive=True)
                       for e in ("*.mp4", "*.mov", "*.webm", "*.mkv", "*.avi")], []))
    vids = [v for v in vids if os.path.basename(v) in caps or os.path.splitext(os.path.basename(v))[0] in caps]
    random.Random(0).shuffle(vids)
    sample = vids[: a.count_sample]
    lens = []
    for v in sample:
        try:
            with av.open(v) as c:
                s = c.streams.video[0]
                n = s.frames or int(float(s.duration * s.time_base) * float(s.average_rate))
            lens.append(n)
        except Exception:
            lens.append(0)
    lens = np.array(lens)
    print(f"captioned videos: {len(vids):,}; frame counts measured on {len(sample):,}")
    for f in (33, 49, 81, 121):
        share = float((lens >= f).mean()) if len(lens) else 0.0
        print(f"  >= {f:3d} frames: {share:6.1%}  -> about {int(share * len(vids)):,} usable clips")


def stage_text(a):
    from transformers import AutoTokenizer, UMT5EncoderModel
    try:
        from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    except Exception:                                   # older/newer diffusers: fall back to plain strip
        prompt_clean = lambda s: " ".join(s.split())   # noqa: E731
    dev = a.device
    rows = [json.loads(l) for l in open(os.path.join(a.out, "manifest.jsonl"))]
    tdir = os.path.join(a.out, "text"); os.makedirs(tdir, exist_ok=True)
    todo = [(r["id"], r["caption"]) for r in rows if not os.path.exists(os.path.join(tdir, r["id"] + ".npy"))]
    todo += [(k, v) for k, v in (("_empty", ""), ("_negative", NEGATIVE_PROMPT))
             if not os.path.exists(os.path.join(tdir, k + ".npy"))]
    print(f"text embeddings to compute: {len(todo):,}")
    if not todo:
        return
    tok = AutoTokenizer.from_pretrained(a.model, subfolder="tokenizer")
    enc = UMT5EncoderModel.from_pretrained(a.model, subfolder="text_encoder", torch_dtype=torch.bfloat16).to(dev).eval()
    for i in range(0, len(todo), a.text_batch):
        chunk = todo[i:i + a.text_batch]
        t = tok([prompt_clean(c) for _, c in chunk], padding="max_length", max_length=512, truncation=True,
                add_special_tokens=True, return_attention_mask=True, return_tensors="pt")
        with torch.no_grad():
            h = enc(t.input_ids.to(dev), t.attention_mask.to(dev)).last_hidden_state.float().cpu()
        lens = t.attention_mask.sum(1)
        for (cid, _), e, n in zip(chunk, h, lens):
            np.save(os.path.join(tdir, cid + ".npy"), e[: int(n)].numpy().astype(np.float16))
        if (i // a.text_batch) % 50 == 0:
            print(f"  text {min(i + a.text_batch, len(todo)):,}/{len(todo):,}", flush=True)


def stage_prompts(a):
    rows = [json.loads(l) for l in open(os.path.join(a.out, "manifest.jsonl"))]
    test = [r for r in rows if r["split"] == "test"][: a.n_eval]
    with open(os.path.join(a.out, "eval_prompts.jsonl"), "w") as fh:
        for r in test:
            fh.write(json.dumps(dict(id=r["id"], caption=r["caption"], video=r.get("video"))) + "\n")
    print(f"eval prompts: {len(test)} held-out captions -> {os.path.join(a.out, 'eval_prompts.jsonl')}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--videos", help="folder of clips (searched recursively), already 16 fps")
    p.add_argument("--captions", help="CSV with a video file-name column and a caption column (OpenVid format)")
    p.add_argument("--video-col", default=None); p.add_argument("--caption-col", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--frames", type=int, default=81, help="video frames per clip, 4n+1")
    p.add_argument("--height", type=int, default=480); p.add_argument("--width", type=int, default=832)
    p.add_argument("--max-clips", type=int, default=20000)
    p.add_argument("--vae-dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--text-batch", type=int, default=32)
    p.add_argument("--n-eval", type=int, default=100)
    p.add_argument("--test-percent", type=int, default=3, help="held-out share of clips, by hash of the clip id")
    p.add_argument("--stage", choices=["all", "count", "latents", "text", "prompts"], default="all")
    p.add_argument("--count-sample", type=int, default=2000, help="videos to inspect for --stage count")
    p.add_argument("--device", default=device_default())
    a = p.parse_args(argv)
    if a.stage == "count":
        if not (a.videos and a.captions):
            p.error("--videos and --captions are required for the count stage")
        return stage_count(a)
    if a.stage in ("all", "latents"):
        if not (a.videos and a.captions):
            p.error("--videos and --captions are required for the latents stage")
        stage_latents(a)
    if a.stage in ("all", "text"):
        stage_text(a)
    if a.stage in ("all", "prompts"):
        stage_prompts(a)


if __name__ == "__main__":
    main()
