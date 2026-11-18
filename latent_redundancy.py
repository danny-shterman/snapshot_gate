"""How much temporal redundancy is left in Wan VAE latents, and which snapshot sizes C it can support.

No training, no GPU: reads latents you already encoded and measures, per clip,

  1. MSD between consecutive latent frames: mean((z_t - z_{t-1})^2), per frame pair and per clip
  2. MSD at lag k (z_t vs z_{t+k}, k = 1..16): how fast latent frames stop resembling each other
  3. For each C: the C x C temporal second-moment matrix over snapshot groups (latent frames 1..C,
     C+1..2C, ... as in Gate 1; latent frame 0 is excluded because it passes through uncoded), and from it
       - energy share of the top eigenvalue (how much of a C-frame group one number per position can hold)
       - linear SNR: best single linear code shared by all positions + best linear decoder
         = 10 log10(tr S / (tr S - lambda_1))
       - average-code SNR: plain temporal average + best linear decoder
       - static SNR: Gate 0's "static_no_network" (copy the average into every frame)
     all for every clip and for the most-moving third of clips (the clips Gate 0's motion check uses).

The linear SNR is what a fixed linear code gets with no learned prior. Gate 0's coded arms add spatially
varying codes and a nonlinear unmixer, so their SNR should land above it; the gap at C = 4, where Gate 0
was run, tells you how much the unmixer adds.

Inputs (auto-detected):
  Gate 1 data folder   --data gate1_data       uses manifest.jsonl; latents are [16, F, H, W]
  folder of .npy       --data latents/         Gate 0 layout [T, 16, H, W] (override with --layout)

Outputs in --out: latent_redundancy.png (4 panels), per_clip.csv, summary.json, summary.md.

Run (CPU; reading the files dominates, about 4 MB per 81-frame clip):
  python latent_redundancy.py --data gate1_data --out redundancy --max-clips 2000 --workers 8
  python latent_redundancy.py --data gate1_data --out redundancy --gate0 runs/gate0.json   # overlay Gate 0
"""
from __future__ import annotations
import argparse, csv, glob, json, math, os, random, sys
from multiprocessing import Pool

import numpy as np

MAX_LAG = 16
C_VALUES = (2, 3, 4, 5, 6, 8, 10, 16)


# --------------------------------------------------------------------------- inputs
def list_clips(data: str, split: str) -> list[tuple[str, str]]:
    """[(clip id, path)] from a Gate 1 manifest, or every .npy in a folder."""
    man = os.path.join(data, "manifest.jsonl")
    if os.path.exists(man):
        rows = [json.loads(l) for l in open(man)]
        if split != "all":
            rows = [r for r in rows if r.get("split") == split]
        return [(r["id"], os.path.join(data, r["latent"])) for r in rows]
    files = sorted(glob.glob(os.path.join(data, "*.npy")))
    return [(os.path.splitext(os.path.basename(f))[0], f) for f in files]


def load_tchw(path: str, layout: str) -> np.ndarray:
    """Latent clip as float32 [T, ch, H, W]."""
    z = np.load(path).astype(np.float32)
    if z.ndim == 5:                       # [1, ch, T, H, W] or [1, T, ch, H, W]
        z = z[0]
    if layout == "cthw":
        z = z.transpose(1, 0, 2, 3)
    return z


# --------------------------------------------------------------------------- per clip
def analyse(job):
    cid, path, layout, skip_first, c_values = job
    try:
        z = load_tchw(path, layout)                      # [T, ch, H, W]
    except Exception as e:                               # unreadable file: report and skip
        return dict(id=cid, error=str(e))
    T = z.shape[0]
    energy = float((z ** 2).mean())
    start = 1 if skip_first and T > 2 else 0
    zc = z[start:]                                       # coded frames only
    n = zc.shape[0]

    pair = ((zc[1:] - zc[:-1]) ** 2).mean(axis=(1, 2, 3))                     # [n-1]
    lag = np.full(MAX_LAG, np.nan)
    for k in range(1, min(MAX_LAG, n - 1) + 1):
        lag[k - 1] = float(((zc[k:] - zc[:-k]) ** 2).mean())
    first_pair = float(((z[1] - z[0]) ** 2).mean()) if skip_first and T > 1 else float("nan")

    # second-moment matrices over non-overlapping groups (Gate 1's grouping), summed, not averaged,
    # so clips can be pooled exactly later
    S = {}
    for C in c_values:
        g = n // C
        if g == 0:
            continue
        x = zc[: g * C].reshape(g, C, -1).transpose(1, 0, 2).reshape(C, -1)   # [C, samples]
        S[C] = (x @ x.T, x.shape[1])
    return dict(id=cid, frames=T, energy=energy, pair=pair, lag=lag, first_pair=first_pair, S=S)


# --------------------------------------------------------------------------- bounds from S
def bounds(S: np.ndarray) -> dict:
    """SNRs (dB) for one pooled C x C second-moment matrix."""
    C = S.shape[0]
    tr = float(np.trace(S))
    lam = np.linalg.eigvalsh(S)[::-1]
    u = np.ones(C) / math.sqrt(C)
    uSu = float(u @ S @ u)
    uS2u = float(u @ S @ S @ u)
    db = lambda err: float(10 * math.log10(tr / max(err, 1e-12 * tr)))
    return dict(top_share=float(lam[0] / tr),
                linear_snr_db=db(tr - lam[0]),               # best shared linear code + LMMSE decoder
                avg_lmmse_snr_db=db(tr - uS2u / uSu),        # temporal average + LMMSE decoder
                static_snr_db=db(tr - uSu))                  # Gate 0 static_no_network


def pooled(results: list[dict], C: int) -> dict | None:
    mats = [r["S"][C] for r in results if C in r["S"]]
    if not mats:
        return None
    S = sum(m for m, _ in mats) / sum(k for _, k in mats)
    out = bounds(S)
    out["clips"] = len(mats)
    return out


# --------------------------------------------------------------------------- Gate 0 overlay
def load_gate0(path: str) -> dict:
    """{C: {"static": dB, "best_coded": dB, "best_arm": name, "learned_ae": dB}} from gate0 results JSON."""
    res = json.load(open(path)).get("results", {})
    out = {}
    for c, arms in res.items():
        coded = {k: v["snr_db"] for k, v in arms.items() if k in ("dcwalsh", "cacti", "randpm", "learned")}
        d = {}
        if "static_no_network" in arms:
            d["static"] = arms["static_no_network"]["snr_db"]
        if coded:
            d["best_arm"] = max(coded, key=coded.get)
            d["best_coded"] = coded[d["best_arm"]]
        if "learned_ae" in arms:
            d["learned_ae"] = arms["learned_ae"]["snr_db"]
        out[int(c)] = d
    return out


# --------------------------------------------------------------------------- plots
def plot(results, high, table, gate0, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    clip_msd = np.array([r["pair"].mean() for r in results])
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    # 1. per-clip consecutive MSD
    a = ax[0, 0]
    a.hist(clip_msd, bins=60, color="#4C72B0", alpha=0.85)
    thr = np.quantile(clip_msd, 2 / 3)
    a.axvline(thr, color="#C44E52", ls="--", label=f"most-moving third starts at {thr:.3f}")
    a.axvline(np.median(clip_msd), color="k", ls=":", label=f"median {np.median(clip_msd):.3f}")
    a.set(title="Per-clip MSD between consecutive latent frames", xlabel="mean (z_t - z_t-1)^2",
          ylabel="clips")
    a.legend(fontsize=8)

    # 2. MSD along the clip
    a = ax[0, 1]
    L = min(len(r["pair"]) for r in results)
    P = np.stack([r["pair"][:L] for r in results])
    x = np.arange(1, L + 1)
    a.plot(x, np.median(P, 0), color="#4C72B0", label="median clip")
    a.fill_between(x, np.quantile(P, .25, 0), np.quantile(P, .75, 0), color="#4C72B0", alpha=.2,
                   label="25-75% of clips")
    fp = np.array([r["first_pair"] for r in results])
    if np.isfinite(fp).any():
        a.scatter([0], [np.nanmedian(fp)], color="#C44E52", zorder=3,
                  label="z0 -> z1 (z0 is encoded alone)")
    a.set(title="Consecutive-frame MSD along the clip", xlabel="frame pair (z_t, z_t+1), t counted from z1",
          ylabel="MSD")
    a.legend(fontsize=8)

    # 3. MSD vs lag
    a = ax[1, 0]
    lag_all = np.nanmean(np.stack([r["lag"] for r in results]), 0)
    lag_hi = np.nanmean(np.stack([r["lag"] for r in high]), 0)
    e = np.mean([r["energy"] for r in results])
    k = np.arange(1, MAX_LAG + 1)
    a.plot(k, lag_all / (2 * e), "o-", color="#4C72B0", label="all clips")
    a.plot(k, lag_hi / (2 * e), "s-", color="#C44E52", label="most-moving third")
    a.axhline(1.0, color="gray", ls=":", label="uncorrelated frames")
    a.set(title="How fast latent frames decorrelate", xlabel="lag k (latent frames; 1 latent = 4 video frames)",
          ylabel="MSD at lag k / (2 x mean energy)  ~  1 - correlation")
    a.set_ylim(bottom=0)
    a.legend(fontsize=8)

    # 4. SNR vs C
    a = ax[1, 1]
    Cs = sorted(table)
    for key, lab, st in (("linear_snr_db", "best linear code (no prior)", "o-"),
                         ("static_snr_db", "static copy (Gate 0 static)", "v:")):
        a.plot(Cs, [table[c]["all"][key] for c in Cs], st, color="#4C72B0", label=f"{lab}, all")
        a.plot(Cs, [table[c]["high"][key] for c in Cs], st, color="#C44E52", label=f"{lab}, most-moving")
    if gate0:
        gc = sorted(c for c in gate0 if "best_coded" in gate0[c])
        if gc:
            a.plot(gc, [gate0[c]["best_coded"] for c in gc], "*", ms=14, color="#55A868",
                   label="Gate 0 measured, best coded arm")
        gc = sorted(c for c in gate0 if "learned_ae" in gate0[c])
        if gc:
            a.plot(gc, [gate0[c]["learned_ae"] for c in gc], "P", ms=10, color="#8172B2",
                   label="Gate 0 measured, learned compressor")
    a.set(title="Reconstruction SNR vs snapshot size C", xlabel="C (latent frames per snapshot)",
          ylabel="SNR dB", xscale="log")
    a.set_xticks(Cs); a.set_xticklabels([str(c) for c in Cs])
    a.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- main
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="Gate 1 data folder (with manifest.jsonl) or folder of .npy")
    p.add_argument("--out", default="redundancy")
    p.add_argument("--layout", choices=["auto", "tchw", "cthw"], default="auto",
                   help="auto: Gate 1 manifest -> cthw ([16, F, H, W]); plain .npy folder -> tchw")
    p.add_argument("--split", choices=["all", "train", "test"], default="all")
    p.add_argument("--max-clips", type=int, default=2000, help="random sample; 0 = every clip")
    p.add_argument("--C", type=int, nargs="+", default=list(C_VALUES))
    p.add_argument("--keep-first", action="store_true", help="include latent frame 0 (default: skip it)")
    p.add_argument("--gate0", default=None, help="Gate 0 results JSON to overlay")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    clips = list_clips(a.data, a.split)
    if not clips:
        sys.exit(f"no latents found in {a.data}")
    layout = a.layout
    if layout == "auto":
        layout = "cthw" if os.path.exists(os.path.join(a.data, "manifest.jsonl")) else "tchw"
    random.Random(a.seed).shuffle(clips)
    if a.max_clips:
        clips = clips[: a.max_clips]
    shape = load_tchw(clips[0][1], layout).shape
    print(f"{len(clips):,} clips, layout {layout}, first clip as [T, ch, H, W] = {list(shape)}", flush=True)
    if shape[1] != 16:
        print(f"warning: {shape[1]} channels, not 16; check --layout", flush=True)

    jobs = [(cid, path, layout, not a.keep_first, tuple(a.C)) for cid, path in clips]
    results = []
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(analyse, jobs, chunksize=4), 1):
            results.append(r)
            if i % 200 == 0 or i == len(jobs):
                print(f"  {i:,}/{len(jobs):,}", flush=True)
    bad = [r for r in results if "error" in r]
    results = [r for r in results if "error" not in r and len(r["pair"]) > 0]
    if bad:
        print(f"skipped {len(bad)} unreadable clips (first: {bad[0]['id']}: {bad[0]['error']})")
    if len(results) < 3:
        sys.exit("too few readable clips")

    clip_msd = np.array([r["pair"].mean() for r in results])
    thr = float(np.quantile(clip_msd, 2 / 3))
    high = [r for r, m in zip(results, clip_msd) if m >= thr]

    table = {}
    for C in a.C:
        al, hi = pooled(results, C), pooled(high, C)
        if al and hi:
            table[C] = dict(all=al, high=hi)
    gate0 = load_gate0(a.gate0) if a.gate0 else {}

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "per_clip.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "latent_frames", "mean_energy", "msd_consecutive_mean", "msd_consecutive_max",
                    "msd_z0_z1", "most_moving_third"] + [f"msd_lag{k}" for k in range(1, MAX_LAG + 1)])
        for r, m in zip(results, clip_msd):
            w.writerow([r["id"], r["frames"], f"{r['energy']:.5f}", f"{m:.5f}", f"{r['pair'].max():.5f}",
                        f"{r['first_pair']:.5f}", int(m >= thr)] + [f"{v:.5f}" for v in r["lag"]])

    summary = dict(clips=len(results), layout=layout, skip_first=not a.keep_first,
                   msd_consecutive=dict(median=float(np.median(clip_msd)), mean=float(clip_msd.mean()),
                                        p90=float(np.quantile(clip_msd, .9)), high_motion_threshold=thr),
                   mean_energy=float(np.mean([r["energy"] for r in results])),
                   by_C={str(c): v for c, v in table.items()}, gate0={str(c): v for c, v in gate0.items()})
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)

    lines = [f"# Latent redundancy: {len(results):,} clips from `{a.data}`", "",
             f"Consecutive-frame MSD per clip: median {np.median(clip_msd):.4f}, "
             f"90th percentile {np.quantile(clip_msd, .9):.4f}; mean frame energy {summary['mean_energy']:.4f}. "
             f"Most-moving third: clips with MSD >= {thr:.4f}.", "",
             "| C | top-eigenvalue share (all / most-moving) | linear SNR dB | average + linear decoder SNR dB "
             "| static SNR dB | Gate 0 best coded dB | Gate 0 learned compressor dB |",
             "|---|---|---|---|---|---|---|"]
    for C in sorted(table):
        t, g = table[C], gate0.get(C, {})
        f = lambda key: f"{t['all'][key]:.2f} / {t['high'][key]:.2f}"
        lines.append(f"| {C} | {t['all']['top_share']:.1%} / {t['high']['top_share']:.1%} | {f('linear_snr_db')} "
                     f"| {f('avg_lmmse_snr_db')} | {f('static_snr_db')} "
                     f"| {g.get('best_coded', float('nan')):.2f} | {g.get('learned_ae', float('nan')):.2f} |")
    lines += ["", "SNR columns: all clips / most-moving third. Linear SNR = best single linear code shared by "
              "all positions with the best linear decoder: what a fixed code gets with no learned prior."]
    md = "\n".join(lines)
    open(os.path.join(a.out, "summary.md"), "w").write(md + "\n")
    plot(results, high, table, gate0, os.path.join(a.out, "latent_redundancy.png"))
    print(md)
    print(f"\nwrote {a.out}/latent_redundancy.png, per_clip.csv, summary.json, summary.md")


if __name__ == "__main__":
    main()
