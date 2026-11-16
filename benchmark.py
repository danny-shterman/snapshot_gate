"""Snapshot Engine: performance and KV-cache benchmark, joined with the Gate 0 quality results.

Three parts, all for compression factors C (C latent frames -> one snapshot):

  A. Analytic, production scale (default: Wan 2.1 14B class, 720p, 81 frames):
     tokens, FLOPs per denoising step, estimated seconds per clip on an accelerator,
     and KV-cache bytes for causal/streaming generation (per second of video and for a long video).
  B. Measured, on a real transformer you can run anywhere:
     B1  bidirectional clip generation: wall-clock per denoising step at tokens(C)
     B2  causal streaming with a real KV cache: cache bytes actually allocated, and time per
         second of generated video.
  C. Joined report: reconstruction quality (from snapshot_gate.py results) next to speed-up and
     KV reduction, so the quality cost of each C is visible beside its gain.

Run:
  python benchmark.py --gate-results runs/gate0.json --out report/            # CPU, small model
  python benchmark.py --gate-results runs/gate0.json --out report/ --device cuda --width 1536 \
      --layers 30 --heads 12 --tokens-per-frame 1560                           # Wan-1.3B-shaped, GPU
"""
from __future__ import annotations
import argparse, json, math, os, statistics, time
import torch
from torch import nn
from torch.nn import functional as F

# --------------------------------------------------------------------------- A. analytic
PRESETS = {
    # tokens_per_frame: latent positions after 8x8 VAE and 2x2 patchify; latent_fps = fps / 4 (Wan-VAE)
    "wan14b_720p": dict(name="Wan 2.1 14B class, 1280x720", params=14e9, layers=40, width=5120,
                        tokens_per_frame=3600, latent_frames=21, latent_fps=4.0),
    "wan1.3b_480p": dict(name="Wan 2.1 1.3B class, 832x480", params=1.3e9, layers=30, width=1536,
                         tokens_per_frame=1560, latent_frames=21, latent_fps=4.0),
}


def analytic(p: dict, Cs, steps=50, guidance=2, flops_per_s=2.25e15, utilisation=0.4,
             kv_bytes_per_elem=2, long_video_s=60.0) -> list[dict]:
    """Per-C cost model. KV assumes full multi-head attention (no GQA), K and V cached per layer."""
    rows, base = [], None
    kv_per_token = p["layers"] * 2 * p["width"] * kv_bytes_per_elem
    for C in Cs:
        snaps = math.ceil(p["latent_frames"] / C)
        T = snaps * p["tokens_per_frame"]
        w = 2 * p["params"] * T
        a = 4 * p["layers"] * p["width"] * T * T
        clip_s = steps * guidance * (w + a) / (flops_per_s * utilisation)
        tok_per_video_s = p["tokens_per_frame"] * p["latent_fps"] / C
        row = dict(C=C, snapshots=snaps, tokens=T, flops_step=w + a, attention_share=a / (w + a),
                   est_seconds_per_clip=clip_s,
                   kv_bytes_per_video_second=tok_per_video_s * kv_per_token,
                   kv_bytes_long_video=tok_per_video_s * kv_per_token * long_video_s)
        base = base or row
        row["speedup"] = base["est_seconds_per_clip"] / clip_s
        row["kv_reduction"] = base["kv_bytes_per_video_second"] / row["kv_bytes_per_video_second"]
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- B. measured
class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h, self.n1, self.n2 = heads, nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def split(self, x):
        B, T, D = x.shape
        return x.view(B, T, self.h, D // self.h).transpose(1, 2)

    def forward(self, x, cache=None):
        """cache: dict with 'k','v' [B, h, T_past, dh] or None. Returns x and new k, v."""
        q, k, v = self.qkv(self.n1(x)).chunk(3, -1)
        q, k, v = self.split(q), self.split(k), self.split(v)
        if cache is not None:
            # block-causal streaming: attend to all past chunks + the whole current chunk
            if cache.get("k") is not None:
                k = torch.cat([cache["k"], k], 2); v = torch.cat([cache["v"], v], 2)
            cache["k"], cache["v"] = k, v
        y = F.scaled_dot_product_attention(q, k, v)
        x = x + self.o(y.transpose(1, 2).reshape(x.shape))
        return x + self.mlp(self.n2(x))


class TinyDiT(nn.Module):
    """DiT-shaped stack (attention + 4x MLP). Conditioning omitted: it does not change cost scaling."""
    def __init__(self, width, layers, heads, ch=16):
        super().__init__()
        self.inp, self.out = nn.Linear(ch, width), nn.Linear(width, ch)
        self.blocks = nn.ModuleList(Block(width, heads) for _ in range(layers))

    def forward(self, x, caches=None):
        h = self.inp(x)
        for i, b in enumerate(self.blocks):
            h = b(h, None if caches is None else caches[i])
        return self.out(h)


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def measure_bidirectional(model, Cs, tokens_per_frame, latent_frames, dev, dtype, repeats=5):
    """Wall-clock of one denoising step (one forward pass) on a whole clip at tokens(C)."""
    rows, base = [], None
    for C in Cs:
        T = math.ceil(latent_frames / C) * tokens_per_frame
        x = torch.randn(1, T, 16, device=dev, dtype=dtype)
        model(x); _sync(dev)                                 # warm-up
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter(); model(x); _sync(dev); times.append(time.perf_counter() - t0)
        ms = 1000 * statistics.median(times)
        base = base or ms
        rows.append(dict(C=C, tokens=T, ms_per_step=ms, speedup=base / ms))
    return rows


@torch.no_grad()
def measure_streaming(model, Cs, tokens_per_frame, latent_frames, latent_fps, dev, dtype):
    """Causal generation, one chunk at a time with a real KV cache.
    A chunk is one latent frame (C=1) or one snapshot (C>1). Reports cache bytes actually held."""
    rows, base = [], None
    for C in Cs:
        caches = [dict(k=None, v=None) for _ in model.blocks]
        n_chunks = math.ceil(latent_frames / C)
        x = torch.randn(1, tokens_per_frame, 16, device=dev, dtype=dtype)
        model(x, [dict(k=None, v=None) for _ in model.blocks]); _sync(dev)   # warm-up
        t0 = time.perf_counter()
        for _ in range(n_chunks):
            model(x, caches)
        _sync(dev)
        sec = time.perf_counter() - t0
        kv = sum(c["k"].numel() * c["k"].element_size() + c["v"].numel() * c["v"].element_size()
                 for c in caches)
        video_s = latent_frames / latent_fps
        row = dict(C=C, chunks=n_chunks, cached_tokens=caches[0]["k"].shape[2], kv_bytes=kv,
                   kv_bytes_per_video_second=kv / video_s, gen_seconds_per_video_second=sec / video_s)
        base = base or row
        row["kv_reduction"] = base["kv_bytes"] / kv
        row["speedup"] = base["gen_seconds_per_video_second"] / row["gen_seconds_per_video_second"]
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- C. joined report
def quality_by_C(path):
    if not path or not os.path.exists(path):
        return {}
    d = json.load(open(path))
    out = {}
    for C, arms in d["results"].items():
        coded = {k: v["snr_db"] for k, v in arms.items() if k in ("dcwalsh", "cacti", "randpm", "learned")}
        best = max(coded, key=coded.get) if coded else None
        out[int(C)] = dict(best_coded_arm=best, best_coded_snr_db=coded.get(best),
                           avg_snr_db=arms.get("avg", {}).get("snr_db"),
                           learned_ae_snr_db=arms.get("learned_ae", {}).get("snr_db"),
                           go=(d.get("decisions", {}).get(str(C)) or d.get("decisions", {}).get(C) or {}).get("go"))
    return out


def gb(x):
    return f"{x / 1e9:,.1f} GB"


def fmt_report(preset, ana, bi, st, qual, meta) -> str:
    L = [f"# Snapshot Engine: quality, speed and KV-cache benchmark", "",
         f"Production model (analytic): **{preset['name']}**, {preset['latent_frames']} latent frames, "
         f"{preset['tokens_per_frame']:,} tokens per latent frame, {meta['steps']} steps x {meta['guidance']} guidance, "
         f"{meta['flops_per_s'] / 1e15:.2f} PFLOPS at {meta['utilisation']:.0%} utilisation, KV in "
         f"{meta['kv_bytes'] * 8}-bit.", "",
         "## Combined: what each C buys and what it costs", "",
         "| C | transformer speed-up (analytic) | measured step speed-up | KV per video second | KV reduction | "
         "best coded SNR dB | learned-compressor SNR dB | averaging SNR dB |",
         "|---|---|---|---|---|---|---|---|"]
    bi_by = {r["C"]: r for r in bi}
    for r in ana:
        q = qual.get(r["C"], {})
        f = lambda v: "-" if v is None else f"{v:.2f}"
        L.append(f"| {r['C']} | {r['speedup']:.1f}x | "
                 f"{bi_by[r['C']]['speedup']:.1f}x | {gb(r['kv_bytes_per_video_second'])} | {r['kv_reduction']:.1f}x | "
                 f"{f(q.get('best_coded_snr_db'))} ({q.get('best_coded_arm') or '-'}) | "
                 f"{f(q.get('learned_ae_snr_db'))} | {f(q.get('avg_snr_db'))} |"
                 if r["C"] in bi_by else
                 f"| {r['C']} | {r['speedup']:.1f}x | - | {gb(r['kv_bytes_per_video_second'])} | "
                 f"{r['kv_reduction']:.1f}x | {f(q.get('best_coded_snr_db'))} | {f(q.get('learned_ae_snr_db'))} | "
                 f"{f(q.get('avg_snr_db'))} |")
    L += ["", "## A. Production scale (analytic)", "",
          "| C | snapshots | tokens | PFLOP per step | attention share | est. s per clip | speed-up | "
          f"KV per video second | KV for {meta['long_video_s']:.0f} s of video |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in ana:
        L.append(f"| {r['C']} | {r['snapshots']} | {r['tokens']:,} | {r['flops_step'] / 1e15:.2f} | "
                 f"{r['attention_share']:.0%} | {r['est_seconds_per_clip']:.0f} | {r['speedup']:.1f}x | "
                 f"{gb(r['kv_bytes_per_video_second'])} | {gb(r['kv_bytes_long_video'])} |")
    L += ["", f"## B1. Measured: one denoising step on a real transformer ({meta['measured_model']}, {meta['device']})", "",
          "| C | tokens | ms per step | speed-up |", "|---|---|---|---|"]
    for r in bi:
        L.append(f"| {r['C']} | {r['tokens']:,} | {r['ms_per_step']:.1f} | {r['speedup']:.1f}x |")
    L += ["", f"## B2. Measured: causal streaming with a real KV cache ({meta['measured_model']}, {meta['device']})", "",
          "| C | chunks | cached tokens | KV bytes held | KV reduction | s to generate 1 s of video | speed-up |",
          "|---|---|---|---|---|---|---|"]
    for r in st:
        L.append(f"| {r['C']} | {r['chunks']} | {r['cached_tokens']:,} | {r['kv_bytes'] / 1e6:,.1f} MB | "
                 f"{r['kv_reduction']:.1f}x | {r['gen_seconds_per_video_second']:.3f} | {r['speedup']:.1f}x |")
    L += ["", "Notes: analytic times are estimates, not measurements. Measured numbers come from the small "
          "transformer named above on this machine. Quality comes from the Gate 0 results file; on synthetic "
          "data it is a pipeline check, not evidence. VAE decoding, the unmixer and the text encoder are excluded "
          "from all speed figures; they do not shrink with C. Measured step speed-ups can exceed the analytic "
          "ones when the small model spends a larger share of its time in attention than the production model "
          "does. KV reduction falls short of C only when the last snapshot is partly empty (for example 24 latent "
          "frames at C = 16 make 2 snapshots); in a long stream it approaches C."]
    return "\n".join(L)


def plot(ana, st, qual, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    blue, grey, ink, quiet = "#2a78d6", "#b9b8b1", "#111111", "#52514e"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), dpi=160)
    ax = axes[0]
    Cs = [r["C"] for r in ana]
    sp = [r["speedup"] for r in ana]
    ax.bar([str(c) for c in Cs], sp, color=grey)
    for i, r in enumerate(ana):
        q = qual.get(r["C"], {})
        lab = f"{r['speedup']:.1f}x"
        if q.get("best_coded_snr_db") is not None:
            lab += f"\nSNR {q['best_coded_snr_db']:.1f} dB"
        ax.text(i, r["speedup"] + 0.5, lab, ha="center", va="bottom", color=ink, fontsize=8.5)
    ax.set_ylim(0, max(sp) * 1.22)
    ax.set_xlabel("C (latent frames per snapshot)")
    ax.set_ylabel("transformer speed-up per clip (analytic)")
    ax.set_title("Speed-up per clip, with best coded reconstruction SNR", loc="left", fontsize=10, color=ink)
    ax2 = axes[1]
    secs = [0, 30, 60, 90, 120]
    for r in ana:
        if r["C"] in (1, 4, 8):
            ys = [r["kv_bytes_per_video_second"] * s / 1e9 for s in secs]
            ax2.plot(secs, ys, marker="o", lw=2, color=blue if r["C"] > 1 else grey,
                     alpha=1.0 if r["C"] == 8 else 0.6)
            ax2.text(secs[-1] * 1.02, ys[-1], f"C = {r['C']}", va="center", fontsize=9, color=ink)
    ax2.set_ylim(0, None)
    ax2.set_xlim(0, secs[-1] * 1.12)
    ax2.set_xlabel("seconds of video in context")
    ax2.set_ylabel("KV cache, GB (bf16, full attention)")
    ax2.set_title("KV cache for causal generation shrinks C-fold", loc="left", fontsize=10, color=ink)
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
        a.grid(axis="y", color="#e6e5e0", lw=0.6)
    fig.tight_layout()
    fig.savefig(path)


# --------------------------------------------------------------------------- main
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--C", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    p.add_argument("--preset", choices=list(PRESETS), default="wan14b_720p")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance", type=int, default=2)
    p.add_argument("--flops-per-s", type=float, default=2.25e15, help="B200 dense BF16")
    p.add_argument("--utilisation", type=float, default=0.4)
    p.add_argument("--kv-bytes", type=int, default=2, help="bytes per cached element (bf16 = 2)")
    p.add_argument("--long-video-s", type=float, default=60.0)
    # measured transformer (defaults run on a laptop CPU in about a minute)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--tokens-per-frame", type=int, default=144)
    p.add_argument("--latent-frames", type=int, default=24)
    p.add_argument("--latent-fps", type=float, default=4.0)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--gate-results", default=None, help="JSON from snapshot_gate.py or merge_results.py")
    p.add_argument("--out", default="report")
    a = p.parse_args(argv)

    os.makedirs(a.out, exist_ok=True)
    preset = PRESETS[a.preset]
    ana = analytic(preset, a.C, a.steps, a.guidance, a.flops_per_s, a.utilisation, a.kv_bytes, a.long_video_s)

    dev = torch.device(a.device)
    dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
    torch.manual_seed(0)
    model = TinyDiT(a.width, a.layers, a.heads).to(dev, dtype).eval()
    Cm = [c for c in a.C if c <= a.latent_frames]
    bi = measure_bidirectional(model, Cm, a.tokens_per_frame, a.latent_frames, dev, dtype, a.repeats)
    st = measure_streaming(model, Cm, a.tokens_per_frame, a.latent_frames, a.latent_fps, dev, dtype)
    qual = quality_by_C(a.gate_results)

    meta = dict(steps=a.steps, guidance=a.guidance, flops_per_s=a.flops_per_s, utilisation=a.utilisation,
                kv_bytes=a.kv_bytes, long_video_s=a.long_video_s, device=str(dev),
                measured_model=f"{a.layers} layers, width {a.width}, {a.heads} heads, "
                               f"{a.tokens_per_frame} tokens/latent frame, {a.latent_frames} latent frames, {dtype}".replace("torch.", ""))
    report = fmt_report(preset, ana, bi, st, qual, meta)
    open(os.path.join(a.out, "report.md"), "w").write(report)
    json.dump(dict(meta=meta, preset=preset, analytic=ana, measured_step=bi, measured_streaming=st, quality=qual),
              open(os.path.join(a.out, "benchmark.json"), "w"), indent=1)
    plot(ana, st, qual, os.path.join(a.out, "benchmark.png"))
    print(report)
    return ana, bi, st


if __name__ == "__main__":
    main()
