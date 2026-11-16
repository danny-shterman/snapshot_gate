"""Gate 1, step 1 (Step B): does stock Wan already "understand" snapshot sequences? No training.

For held-out captioned clips, computes stock Wan 2.1's flow-matching loss on four inputs that all
have the same number of token frames K = 1 + n (6 for 81-frame clips):

  (a) frames     the first K consecutive latent frames (a short ordinary clip): the reference
  (b) snapshot   z0 + n coded snapshots of the whole clip, scaled to unit variance
  (c) learned_ae z0 + n learned-compressor latents of the whole clip, scaled the same way
  (d) subsample  z0, z4, z8, ... (plain temporal decimation at the same compression, no coding)

All four use the same noise, the same noise levels and the same caption, so the comparison is paired.
Lower loss = the input looks more like data the model already models well.

Decision rule (pre-registered; edit BEFORE running, not after):
  PASS if, averaged over noise levels,  loss(snapshot) / loss(frames)     <= SNAP_OVER_FRAMES_MAX
                                  and   loss(snapshot) / loss(learned_ae) <= SNAP_OVER_AE_MAX

Example:
  python gate1/zero_shot.py --data gate1_data --unmixer gate1_runs/unmixer_learned.pt \
      --ae gate1_runs/unmixer_ae.pt --out gate1_runs/zero_shot.json
"""
from __future__ import annotations
import argparse, json, os, sys, time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (DEFAULT_MODEL, ae_encode_sequence, device_default, encode_snapshots, load_arm,  # noqa: E402
                    load_latent, load_text, n_snapshots, read_manifest, scale_snapshots)

SNAP_OVER_FRAMES_MAX = 1.5
SNAP_OVER_AE_MAX = 0.9
ARMS = ["frames", "snapshot", "learned_ae", "subsample"]


def build_inputs(z, coded, ae, C):
    """z [1,16,F,H,W] Wan-normalised -> dict of [1,16,K,H,W] inputs."""
    F_ = z.shape[2]
    n = n_snapshots(F_, C); K = 1 + n
    out = dict(frames=z[:, :, :K])
    T = coded.code(z.shape[-2], z.shape[-1], z.device)
    out["snapshot"] = scale_snapshots(encode_snapshots(z, T), coded._snap_mean, coded._snap_std)
    if ae is not None:
        out["learned_ae"] = scale_snapshots(ae_encode_sequence(z, ae), ae._snap_mean, ae._snap_std)
    out["subsample"] = z[:, :, [0] + [C * k for k in range(1, n + 1)]]
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--unmixer", required=True, help="learned-code checkpoint from train_unmixer.py")
    p.add_argument("--ae", default=None, help="learned-compressor checkpoint (train_unmixer.py --arm learned_ae)")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--n-clips", type=int, default=200)
    p.add_argument("--t", type=float, nargs="+", default=[0.2, 0.5, 0.8], help="noise levels sigma in (0,1)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="zero_shot.json")
    p.add_argument("--device", default=device_default())
    a = p.parse_args(argv)

    from diffusers import WanTransformer3DModel
    dev = a.device
    dtype = torch.bfloat16 if dev.startswith("cuda") else torch.float32
    coded, ck = load_arm(a.unmixer, dev)
    coded._snap_mean, coded._snap_std = ck["snap_mean"].to(dev), ck["snap_std"].to(dev)
    C = ck["C"]
    ae = None
    if a.ae:
        ae, ck2 = load_arm(a.ae, dev)
        ae._snap_mean, ae._snap_std = ck2["snap_mean"].to(dev), ck2["snap_std"].to(dev)
    arms = [k for k in ARMS if k != "learned_ae" or ae is not None]
    model = WanTransformer3DModel.from_pretrained(a.model, subfolder="transformer", torch_dtype=dtype).to(dev).eval()

    rows = read_manifest(a.data, "test")[: a.n_clips]
    gen = torch.Generator(device="cpu").manual_seed(a.seed)
    losses = {k: {t: [] for t in a.t} for k in arms}
    t0 = time.time()
    with torch.no_grad():
        for i, r in enumerate(rows):
            z = load_latent(a.data, r)[None].to(dev)
            emb = load_text(a.data, r["id"])[None].to(dev, dtype)
            x0s = build_inputs(z, coded, ae, C)
            for t in a.t:
                eps = torch.randn(x0s["frames"].shape, generator=gen).to(dev)       # shared across arms
                for k in arms:
                    x0 = x0s[k]
                    xt = (1 - t) * x0 + t * eps
                    v = model(hidden_states=xt.to(dtype), timestep=torch.tensor([t * 1000.0], device=dev),
                              encoder_hidden_states=emb, return_dict=False)[0].float()
                    losses[k][t].append(float(torch.mean((v - (eps - x0)) ** 2)))
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(rows)} clips  ({(time.time() - t0) / (i + 1):.1f} s/clip)", flush=True)

    mean = {k: {t: sum(v) / len(v) for t, v in d.items()} for k, d in losses.items()}
    avg = {k: sum(d.values()) / len(d) for k, d in mean.items()}
    ratio = lambda num, den: sum(mean[num][t] / mean[den][t] for t in a.t) / len(a.t)  # noqa: E731
    checks = {"snapshot_close_to_frames": ratio("snapshot", "frames") <= SNAP_OVER_FRAMES_MAX}
    if ae is not None:
        checks["snapshot_better_than_learned_ae"] = ratio("snapshot", "learned_ae") <= SNAP_OVER_AE_MAX
    res = dict(n_clips=len(rows), t=a.t, mean_loss=mean, avg_loss=avg,
               ratio_snapshot_over_frames=ratio("snapshot", "frames"),
               ratio_snapshot_over_learned_ae=ratio("snapshot", "learned_ae") if ae is not None else None,
               ratio_subsample_over_frames=ratio("subsample", "frames"),
               thresholds=dict(SNAP_OVER_FRAMES_MAX=SNAP_OVER_FRAMES_MAX, SNAP_OVER_AE_MAX=SNAP_OVER_AE_MAX),
               checks=checks, go=all(checks.values()))
    json.dump(res, open(a.out, "w"), indent=1)

    print("\n| input | " + " | ".join(f"loss @ t={t}" for t in a.t) + " | average |")
    print("|---|" + "---|" * (len(a.t) + 1))
    for k in arms:
        print(f"| {k} | " + " | ".join(f"{mean[k][t]:.4f}" for t in a.t) + f" | {avg[k]:.4f} |")
    print(f"\nsnapshot / frames      = {res['ratio_snapshot_over_frames']:.3f}  (pass <= {SNAP_OVER_FRAMES_MAX})")
    if ae is not None:
        print(f"snapshot / learned_ae  = {res['ratio_snapshot_over_learned_ae']:.3f}  (pass <= {SNAP_OVER_AE_MAX})")
    print(f"subsample / frames     = {res['ratio_subsample_over_frames']:.3f}  (reference: no coding)")
    print(f"\nZero-shot decision: {'PASS' if res['go'] else 'FAIL'}  {checks}")


if __name__ == "__main__":
    main()
