"""Gate 1, step 3: train the snapshot code + unmixer (or the learned compressor) on Wan-normalised latents.

This is the Gate 0 winning recipe (learned linear code, frozen for the first 5k steps, then trained
at 0.1x the learning rate) retrained in the latent space the Wan transformer actually uses, on groups
aligned with how Gate 1 codes a clip (latent frames 1-4, 5-8, ...). It also:
  * adds small noise to the snapshot during training (--snap-noise), because generated snapshots
    are never exact, so the unmixer should not trust them blindly;
  * measures per-channel snapshot statistics after training (used to scale snapshots to unit
    variance for diffusion) and stores them in the checkpoint;
  * reports held-out SNR on full frames, comparable to Gate 0's numbers.

--arm learned_ae trains Gate 0's learned compressor the same way; zero_shot.py uses it as arm (c).

Example:
  python gate1/train_unmixer.py --data gate1_data --arm learned    --out gate1_runs/unmixer_learned.pt
  python gate1/train_unmixer.py --data gate1_data --arm learned_ae --out gate1_runs/unmixer_ae.pt
"""
from __future__ import annotations
import argparse, math, os, random, sys, time
import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (device_default, encode_snapshots, ae_encode_sequence, group_frames, read_manifest,  # noqa: E402
                    load_latent, sg, unmix_sequence)


class GroupCrops(torch.utils.data.IterableDataset):
    """Endless random (clip, aligned group, crop) samples -> f [C, 16, s, s]."""
    def __init__(self, data_dir, rows, C, size, seed):
        self.d, self.rows, self.C, self.s, self.seed = data_dir, rows, C, size, seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed + (info.id if info else 0) * 1000003)
        while True:
            r = self.rows[rng.randrange(len(self.rows))]
            z = np.load(os.path.join(self.d, r["latent"]), mmap_mode="r")          # [16, F, H, W]
            n = (z.shape[1] - 1) // self.C
            j = rng.randrange(n)
            H, W = z.shape[2:]
            s = min(self.s, H, W)
            y0, x0 = rng.randrange(H - s + 1), rng.randrange(W - s + 1)
            f = np.asarray(z[:, 1 + j * self.C: 1 + (j + 1) * self.C, y0:y0 + s, x0:x0 + s], dtype=np.float32)
            del z
            yield torch.from_numpy(f).permute(1, 0, 2, 3).contiguous()           # [C, 16, s, s]


def snr_db(ref, est):
    return float(10 * torch.log10(ref.var() / F.mse_loss(est, ref).clamp_min(1e-12)))


@torch.no_grad()
def evaluate_full(model, kind, rows, data_dir, C, device, max_clips):
    """Reconstruct every aligned group of held-out clips at full resolution."""
    refs, ests = [], []
    for r in rows[:max_clips]:
        z = load_latent(data_dir, r)[None].to(device)                               # [1,16,F,H,W]
        f = group_frames(z, C)[0]                                                   # [n,C,16,H,W]
        if kind == "learned_ae":
            est = model(f)
        else:
            est = unmix_sequence(encode_snapshots(z, model.code(z.shape[-2], z.shape[-1], device)), model)
            est = group_frames(est, C)[0]
        refs.append(f.cpu()); ests.append(est.cpu())
    f, e = torch.cat(refs), torch.cat(ests)
    d_ref, d_est = f[:, 1:] - f[:, :-1], e[:, 1:] - e[:, :-1]
    m = d_ref.pow(2).flatten(1).mean(1)
    hi = m >= m.quantile(2 / 3)
    return dict(snr_db=snr_db(f, e), motion_snr_db=snr_db(d_ref, d_est),
                high_motion_snr_db=snr_db(f[hi], e[hi]), high_motion_motion_snr_db=snr_db(d_ref[hi], d_est[hi]),
                groups=len(f))


@torch.no_grad()
def snapshot_stats(model, kind, rows, data_dir, C, device, max_clips):
    """Per-channel mean/std of what the transformer will see in place of frames 1..F-1."""
    s1 = s2 = 0.0; n = 0
    for r in rows[:max_clips]:
        z = load_latent(data_dir, r)[None].to(device)
        if kind == "learned_ae":
            seq = ae_encode_sequence(z, model)
        else:
            seq = encode_snapshots(z, model.code(z.shape[-2], z.shape[-1], device))
        g = seq[0, :, 1:].flatten(1).double()                                       # [16, n*H*W]
        s1 = s1 + g.sum(1); s2 = s2 + (g * g).sum(1); n += g.shape[1]
    mean = s1 / n
    std = (s2 / n - mean ** 2).clamp_min(1e-12).sqrt()
    return mean.float().cpu(), std.float().cpu()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--arm", choices=["learned", "learned_ae"], default="learned")
    p.add_argument("--out", required=True, help="checkpoint path (.pt)")
    p.add_argument("--C", type=int, default=4)
    p.add_argument("--crop", type=int, default=64)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--steps", type=int, default=40000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--code-lr-mult", type=float, default=0.1)
    p.add_argument("--code-freeze-steps", type=int, default=5000)
    p.add_argument("--snap-noise", type=float, default=0.05,
                   help="std of noise added to the snapshot (Wan-normalised units) on half the samples; 0 = off")
    p.add_argument("--eval-clips", type=int, default=200)
    p.add_argument("--stats-clips", type=int, default=500)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=1000)
    p.add_argument("--device", default=device_default())
    a = p.parse_args(argv)

    train, test = read_manifest(a.data, "train"), read_manifest(a.data, "test")
    assert train and test, "manifest needs both train and test clips"
    torch.manual_seed(a.seed)
    model = sg.make_arm(a.arm, a.C, 16, a.width, seed=a.seed).to(a.device)
    code = [model.learned.w] if getattr(model, "learned", None) is not None else []
    rest = [q for q in model.parameters() if all(q is not c for c in code)]
    groups = [dict(params=rest)] + ([dict(params=code, lr=a.lr * a.code_lr_mult, weight_decay=0.0)] if code else [])
    opt = torch.optim.AdamW(groups, lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    dl = torch.utils.data.DataLoader(GroupCrops(a.data, train, a.C, a.crop, a.seed), batch_size=a.batch,
                                     num_workers=a.num_workers, pin_memory=a.device.startswith("cuda"),
                                     persistent_workers=a.num_workers > 0)
    it = iter(dl)
    g_noise = torch.Generator(device=a.device).manual_seed(a.seed + 7)
    t0, run = time.time(), None
    print(f"training {a.arm}: C={a.C} width={a.width} crop={a.crop} steps={a.steps} on {len(train):,} clips", flush=True)
    for step in range(a.steps):
        if code:
            code[0].requires_grad_(step >= a.code_freeze_steps)
        f = next(it).to(a.device, non_blocking=True)                               # [B,C,16,s,s]
        if a.arm == "learned_ae":
            est = model(f)
        else:
            g, T = model.encode(f)
            if a.snap_noise > 0:
                keep = (torch.rand(len(g), 1, 1, 1, device=a.device, generator=g_noise) < 0.5).float()
                g = g + keep * a.snap_noise * torch.randn(g.shape, device=a.device, generator=g_noise)
            est = model.decode(g, T)
        loss = F.mse_loss(est, f)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        run = loss.item() if run is None else 0.98 * run + 0.02 * loss.item()
        if a.log_every and (step + 1) % a.log_every == 0:
            el = time.time() - t0
            print(f"  step {step + 1:6d}  loss {run:.4f}  {(step + 1) / el:.1f} it/s  "
                  f"~{(a.steps - step - 1) / ((step + 1) / el) / 60:.0f} min left", flush=True)

    model.eval()
    res = evaluate_full(model, a.arm, test, a.data, a.C, a.device, a.eval_clips)
    smean, sstd = snapshot_stats(model, a.arm, train, a.data, a.C, a.device, a.stats_clips)
    print(f"held-out ({res['groups']} groups, full frames): SNR {res['snr_db']:.2f} dB | motion SNR "
          f"{res['motion_snr_db']:.2f} dB | high-motion motion SNR {res['high_motion_motion_snr_db']:.2f} dB")
    print(f"snapshot stats: mean {smean.abs().mean():.3f} (avg |.|), std {sstd.min():.3f}-{sstd.max():.3f}")
    ck = dict(state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()}, kind=a.arm, C=a.C, ch=16,
              width=a.width, a=0.7071, seed=a.seed, snap_mean=smean, snap_std=sstd, eval=res, args=vars(a),
              train_sec=round(time.time() - t0))
    if code:
        with torch.no_grad():
            ck["code_tile"] = model.learned(model.learned.tile, model.learned.tile).cpu()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save(ck, a.out)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
