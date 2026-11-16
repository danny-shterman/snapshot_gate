"""Gate 1, step 7: compare the snapshot model with stock Wan, apply the decision rule, write side-by-sides.

Inputs are folders written by sample.py (same prompts, same order). Computes, per video:
  motion       mean absolute difference between consecutive frames (proxy for VBench dynamic degree)
  seam ratio   frame difference across snapshot-group boundaries / elsewhere. Unmixing errors show up
               as a jump every C*4 video frames (every 16 frames at C = 4); 1.0 means no visible seam.
               Computed at the same frame positions for every system, so stock Wan gives the baseline.
  CLIP score   (with --clip) mean image-text cosine x 100 over 8 frames: prompt alignment
and, per run, the mean stage timings from summary.json.

Decision rule (pre-registered; edit BEFORE generating videos, not after). Automatic checks:
  denoise speed-up vs reference           >= MIN_DENOISE_SPEEDUP
  end-to-end speed-up vs reference        >= MIN_TOTAL_SPEEDUP
  seam ratio                              <= reference seam ratio x MAX_SEAM_VS_REF
  motion                                  >= reference motion x MIN_MOTION_VS_REF
  CLIP score (if --clip)                  >= reference - MAX_CLIP_DROP
  CLIP score vs equal-compute run         >  equal-compute run's
VBench quality (README) and watching the side-by-sides complete the rule.

Example:
  python gate1/evaluate.py --runs snapshot=gate1_eval/snapshot wan50=gate1_eval/wan50 wan_eq=gate1_eval/wan_eq \
      --reference wan50 --equal-compute wan_eq --clip --out gate1_eval/report
"""
from __future__ import annotations
import argparse, glob, json, os, statistics, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import read_video, write_video  # noqa: E402

MIN_DENOISE_SPEEDUP = 5.0
MIN_TOTAL_SPEEDUP = 3.0
MAX_SEAM_VS_REF = 1.10
MIN_MOTION_VS_REF = 0.90
MAX_CLIP_DROP = 1.0


def frame_metrics(frames: np.ndarray, group_len: int) -> dict:
    x = frames[:, ::2, ::2].astype(np.float32) / 255.0                       # half resolution is plenty
    d = np.abs(np.diff(x, axis=0)).mean(axis=(1, 2, 3))                      # d[t]: frame t -> t+1
    seams = [t for t in range(group_len, len(d), group_len)]                # between groups (not the z0 edge)
    rest = [t for t in range(1, len(d)) if t not in seams]
    seam = float(np.mean(d[seams]) / max(np.mean(d[rest]), 1e-8)) if seams and rest else float("nan")
    return dict(motion=float(d.mean()), seam_ratio=seam, first_frame_jump=float(d[0] / max(np.mean(d[1:]), 1e-8)))


class Clip:
    def __init__(self, name, device):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch, self.dev = torch, device
        self.m = CLIPModel.from_pretrained(name).to(device).eval()
        self.p = CLIPProcessor.from_pretrained(name)

    def score(self, frames, text):
        torch = self.torch
        idx = np.linspace(0, len(frames) - 1, 8).round().astype(int)
        inp = self.p(text=[text], images=[frames[i] for i in idx], return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            out = self.m(**{k: v.to(self.dev) for k, v in inp.items()})
        im = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        tx = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
        return float((im @ tx.T).mean() * 100)


def side_by_side(paths, labels, out, height=240):
    from PIL import Image, ImageDraw
    vids = [read_video(p) for p in paths]
    T = min(len(v) for v in vids)
    tiles = []
    for v, lab in zip(vids, labels):
        H, W = v.shape[1:3]
        w = int(round(W * height / H / 2)) * 2
        fr = []
        for f in v[:T]:
            im = Image.fromarray(f).resize((w, height))
            ImageDraw.Draw(im).text((8, 6), lab, fill=(255, 255, 0))
            fr.append(np.asarray(im))
        tiles.append(np.stack(fr))
    write_video(np.concatenate(tiles, axis=2), out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True, help="name=folder pairs written by sample.py")
    p.add_argument("--reference", required=True, help="run name of stock Wan at full steps")
    p.add_argument("--equal-compute", default=None, help="run name of stock Wan at reduced steps")
    p.add_argument("--C", type=int, default=4)
    p.add_argument("--vae-temporal", type=int, default=4)
    p.add_argument("--clip", action="store_true", help="compute CLIP scores (downloads a CLIP model)")
    p.add_argument("--clip-model", default="openai/clip-vit-large-patch14")
    p.add_argument("--grid", type=int, default=20, help="side-by-side videos to write")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default=None)
    a = p.parse_args(argv)

    runs = dict(r.split("=", 1) for r in a.runs)
    assert a.reference in runs, "--reference must be one of the run names"
    os.makedirs(a.out, exist_ok=True)
    group_len = a.C * a.vae_temporal
    clip = None
    if a.clip:
        import torch
        clip = Clip(a.clip_model, a.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    per, summ = {}, {}
    for name, folder in runs.items():
        timing = {json.loads(l)["k"]: json.loads(l) for l in open(os.path.join(folder, "timing.jsonl"))}
        rows = []
        for path in sorted(glob.glob(os.path.join(folder, "videos", "*.mp4"))):
            k = int(os.path.basename(path).split("_")[0])
            fr = read_video(path)
            m = frame_metrics(fr, group_len)
            if clip:
                m["clip"] = clip.score(fr, timing[k]["caption"])
            rows.append(dict(k=k, path=path, **m))
        per[name] = rows
        s = json.load(open(os.path.join(folder, "summary.json")))
        cfg = json.load(open(os.path.join(folder, "config.json")))
        summ[name] = dict(system=cfg["system"], steps=cfg["steps"], n=len(rows),
                          **{k: s[k] for k in ("text_s", "denoise_s", "unmix_s", "decode_s", "total_s")},
                          **{m: statistics.mean(r[m] for r in rows) for m in rows[0] if m not in ("k", "path")})
        print(f"{name}: {len(rows)} videos measured", flush=True)

    ref = summ[a.reference]
    cand = next((n for n, s in summ.items() if s["system"] == "snapshot"), None)
    checks = {}
    if cand:
        c = summ[cand]
        c["denoise_speedup"] = ref["denoise_s"] / c["denoise_s"]
        c["total_speedup"] = ref["total_s"] / c["total_s"]
        checks["denoise_speedup"] = c["denoise_speedup"] >= MIN_DENOISE_SPEEDUP
        checks["total_speedup"] = c["total_speedup"] >= MIN_TOTAL_SPEEDUP
        checks["no_visible_seams"] = c["seam_ratio"] <= ref["seam_ratio"] * MAX_SEAM_VS_REF
        checks["keeps_motion"] = c["motion"] >= ref["motion"] * MIN_MOTION_VS_REF
        if clip:
            checks["prompt_alignment"] = c["clip"] >= ref["clip"] - MAX_CLIP_DROP
            if a.equal_compute in summ:
                checks["beats_equal_compute"] = c["clip"] > summ[a.equal_compute]["clip"]
    for n, s in summ.items():
        if n != a.reference:
            s.setdefault("denoise_speedup", ref["denoise_s"] / s["denoise_s"])
            s.setdefault("total_speedup", ref["total_s"] / s["total_s"])

    cols = ["steps", "denoise_s", "unmix_s", "decode_s", "text_s", "total_s", "denoise_speedup", "total_speedup",
            "motion", "seam_ratio", "first_frame_jump"] + (["clip"] if clip else [])
    L = ["# Gate 1 evaluation", "", "| run | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for n, s in summ.items():
        L.append(f"| {n} | " + " | ".join("-" if s.get(k) is None else
                                          (f"{s[k]:.3f}" if isinstance(s.get(k), float) else str(s[k])) for k in cols) + " |")
    if cand:
        L += ["", f"## Decision (automatic part, candidate: {cand}, reference: {a.reference})", ""]
        L += [f"- {k}: {'PASS' if v else 'FAIL'}" for k, v in checks.items()]
        L += ["", f"Automatic checks: {'ALL PASS' if all(checks.values()) else 'NOT ALL PASS'}. "
              "Complete the rule with VBench and the side-by-side videos (see README)."]
    report = "\n".join(L)
    open(os.path.join(a.out, "report.md"), "w").write(report)
    json.dump(dict(summary=summ, per_video=per, checks=checks), open(os.path.join(a.out, "report.json"), "w"), indent=1)
    print(report)

    if a.grid:
        names = list(runs)
        by_k = {n: {r["k"]: r["path"] for r in per[n]} for n in names}
        common_k = sorted(set.intersection(*[set(v) for v in by_k.values()]))[: a.grid]
        os.makedirs(os.path.join(a.out, "side_by_side"), exist_ok=True)
        for k in common_k:
            side_by_side([by_k[n][k] for n in names], names, os.path.join(a.out, "side_by_side", f"{k:03d}.mp4"))
        print(f"side-by-side videos: {len(common_k)} in {os.path.join(a.out, 'side_by_side')}")


if __name__ == "__main__":
    main()
