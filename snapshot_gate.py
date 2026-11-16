"""Snapshot Engine, Gate 0: can C latent frames be recovered from one coded snapshot?

Compares, at equal compression C (C frames -> one frame-sized latent):
  avg         plain temporal average + unmixer          (no coding; lower bound)
  dcwalsh     DC-plus-Walsh code + unmixer              ("Bayer filter for time")
  cacti       CACTI-style shifted binary mask + unmixer (normalised per position)
  randpm      i.i.d. random +-1 code + unmixer
  learned     learned code (normalised, so A A^T = I)   + unmixer
  learned_ae  learned nonlinear encoder + same decoder  (the learned-compression competitor)

Every coded arm satisfies A A^T = I and ends with the exact data-consistency projection
f <- f + A^T (g - A f). Data: synthetic moving textures, a folder of .npy arrays
(pixels or VAE latents, shape [T, ch, H, W]), or a folder of videos.

The learned code can collapse to plain averaging early in training (once the snapshot is an
average it carries no motion, so nothing pushes it back). --code-freeze-steps keeps the code
fixed while the unmixer warms up, and --code-lr-mult trains it at a lower learning rate.
The defaults (0 and 1.0) reproduce the original behaviour exactly.

Run:  python snapshot_gate.py --data synthetic --C 4 8 --steps 3000 --device cuda
      python snapshot_gate.py --data npy --path latents/ --C 4 8 --steps 20000 --device cuda
      python snapshot_gate.py --data npy --path latents/ --C 4 --arms learned --width 128 --steps 40000 \
          --code-lr-mult 0.1 --code-freeze-steps 5000 --device cuda
"""
from __future__ import annotations
import argparse, glob, json, math, os, time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

# --------------------------------------------------------------------------- codes
ARMS = ["avg", "dcwalsh", "cacti", "randpm", "learned", "learned_ae"]


def walsh(C: int) -> torch.Tensor:
    """Sylvester Hadamard matrix with rows scaled to unit length. Row 0 is the DC row."""
    assert C >= 2 and C & (C - 1) == 0, "C must be a power of two"
    Hm = torch.ones(1, 1)
    while Hm.shape[0] < C:
        Hm = torch.cat([torch.cat([Hm, Hm], 1), torch.cat([Hm, -Hm], 1)], 0)
    return Hm / math.sqrt(C)


def fixed_code(kind: str, C: int, H: int, W: int, a: float = 0.7071, seed: int = 0) -> torch.Tensor:
    """Returns T [C, H, W] with unit norm along C at every position (shared across channels)."""
    g = torch.Generator().manual_seed(seed)
    if kind == "avg":
        T = torch.ones(C, H, W) / math.sqrt(C)
    elif kind == "dcwalsh":
        Wh = walsh(C)                                    # [C, C], row 0 = DC
        b = math.sqrt(max(0.0, 1 - a * a))
        y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
        r = 1 + (x + 2 * y) % (C - 1)                    # non-DC row per position
        T = a * Wh[0].view(C, 1, 1) + b * Wh[r].permute(2, 0, 1)
    elif kind == "cacti":
        base = (torch.rand(H, W + C, generator=g) < 0.5).float()
        T = torch.stack([base[:, k:k + W] for k in range(C)])  # mask shifted 1 position per frame
        n = T.norm(dim=0, keepdim=True)
        T = torch.where(n > 0, T / n.clamp_min(1e-8), torch.ones_like(T) / math.sqrt(C))
    elif kind == "randpm":
        T = (torch.randint(0, 2, (C, H, W), generator=g).float() * 2 - 1) / math.sqrt(C)
    else:
        raise ValueError(kind)
    return T


class LearnedCode(nn.Module):
    """Free code per position inside a tile, normalised to unit length (A A^T = I)."""
    def __init__(self, C: int, tile: int = 4, seed: int = 0):
        super().__init__()
        init = fixed_code("dcwalsh", C, tile, tile, seed=seed)
        self.w = nn.Parameter(init + 0.01 * torch.randn(C, tile, tile))
        self.tile = tile

    def forward(self, H: int, W: int) -> torch.Tensor:
        t = self.w / self.w.norm(dim=0, keepdim=True).clamp_min(1e-6)
        reps = (math.ceil(H / self.tile), math.ceil(W / self.tile))
        return t.repeat(1, *reps)[:, :H, :W]


def snapshot(f: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """f [B, C, ch, H, W], T [C, H, W] -> g [B, ch, H, W] = sum_k T_k * f_k."""
    return (f * T[None, :, None]).sum(1)


def project(fhat: torch.Tensor, g: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """Exact projection onto {f : A f = g} (valid because A A^T = I)."""
    return fhat + T[None, :, None] * (g - snapshot(fhat, T))[:, None]


# --------------------------------------------------------------------------- networks
def block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(min(8, cout), cout),
                         nn.SiLU(), nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(min(8, cout), cout),
                         nn.SiLU())


class UNet(nn.Module):
    """Small 2-level U-Net. Purely local: the receptive field is a few dozen positions."""
    def __init__(self, cin: int, cout: int, width: int = 32):
        super().__init__()
        w = width
        self.e1, self.e2, self.mid = block(cin, w), block(w, 2 * w), block(2 * w, 4 * w)
        self.u2, self.d2 = nn.ConvTranspose2d(4 * w, 2 * w, 2, 2), block(4 * w, 2 * w)
        self.u1, self.d1 = nn.ConvTranspose2d(2 * w, w, 2, 2), block(2 * w, w)
        self.out = nn.Conv2d(w, cout, 1)

    def forward(self, x):
        H, W = x.shape[-2:]
        ph, pw = (-H) % 4, (-W) % 4
        x = F.pad(x, (0, pw, 0, ph), mode="replicate")
        e1 = self.e1(x); e2 = self.e2(F.avg_pool2d(e1, 2)); m = self.mid(F.avg_pool2d(e2, 2))
        d2 = self.d2(torch.cat([self.u2(m), e2], 1)); d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.out(d1)[..., :H, :W]


class CodedArm(nn.Module):
    """Fixed or learned code, local unmixer that also sees the code maps, exact projection."""
    def __init__(self, kind: str, C: int, ch: int, width: int, a: float, seed: int):
        super().__init__()
        self.kind, self.C, self.ch, self.a, self.seed = kind, C, ch, a, seed
        self.learned = LearnedCode(C, seed=seed) if kind == "learned" else None
        self.net = UNet(ch + C, C * ch, width)
        self._cache = {}

    def code(self, H, W, device):
        if self.learned is not None:
            return self.learned(H, W).to(device)
        key = (H, W, str(device))
        if key not in self._cache:
            self._cache[key] = fixed_code(self.kind, self.C, H, W, self.a, self.seed).to(device)
        return self._cache[key]

    def encode(self, f):
        T = self.code(f.shape[-2], f.shape[-1], f.device)
        return snapshot(f, T), T

    def decode(self, g, T):
        B, ch, H, W = g.shape
        # init: back-projection A^T g, then a local network predicts the rest
        init = T[None, :, None] * g[:, None]
        x = torch.cat([g, T[None].expand(B, -1, -1, -1)], 1)
        fhat = init + self.net(x).view(B, self.C, ch, H, W)
        return project(fhat, g, T)

    def forward(self, f):
        g, T = self.encode(f)
        return self.decode(g, T)


def tile_position(H: int, W: int, tile: int, device) -> torch.Tensor:
    """One-hot position inside a tile x tile cell: [tile*tile, H, W]."""
    y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
    idx = (y % tile) * tile + (x % tile)
    return F.one_hot(idx, tile * tile).permute(2, 0, 1).float()


class LearnedAE(nn.Module):
    """Learned temporal compressor at the same ratio: C frames -> one ch-channel latent.

    Encoder and decoder both see the position inside a 4x4 tile, so the encoder CAN learn
    position-dependent (spatially multiplexed) codes like the coded arms. It is therefore at
    least as expressive as every coded arm: nonlinear and position-aware."""
    def __init__(self, C: int, ch: int, width: int, tile: int = 4):
        super().__init__()
        self.C, self.ch, self.tile = C, ch, tile
        P = tile * tile
        self.enc = nn.Sequential(block(C * ch + P, width), nn.Conv2d(width, ch, 1))
        self.dec = UNet(ch + P, C * ch, width)

    def forward(self, f):
        B, C, ch, H, W = f.shape
        pos = tile_position(H, W, self.tile, f.device)[None].expand(B, -1, -1, -1)
        z = self.enc(torch.cat([f.reshape(B, C * ch, H, W), pos], 1))
        return self.dec(torch.cat([z, pos], 1)).view(B, C, ch, H, W)


def make_arm(kind, C, ch, width, a=0.7071, seed=0):
    return LearnedAE(C, ch, width) if kind == "learned_ae" else CodedArm(kind, C, ch, width, a, seed)


# --------------------------------------------------------------------------- data
def _value_noise(g, h, w, cells):
    n = torch.rand(1, 1, cells, cells, generator=g)
    return F.interpolate(n, size=(h, w), mode="bicubic", align_corners=False)[0, 0]


def synthetic_clip(T: int, size: int, ch: int = 3, max_speed: float = 3.0, g=None) -> torch.Tensor:
    """Textured background with optional camera pan, plus 1-3 textured moving shapes.
    Returns [T, ch, size, size] in roughly [-1, 1]. Speeds in pixels per frame."""
    big = 2 * size
    bg = torch.stack([_value_noise(g, big, big, (4, 8, 16)[c % 3]) for c in range(ch)]) * 2 - 1
    pan = (torch.rand(2, generator=g) * 2 - 1) * max_speed * 0.5
    shapes = []
    for _ in range(int(torch.randint(1, 4, (1,), generator=g))):
        r = float(torch.rand(1, generator=g) * size * 0.15 + size * 0.08)
        p = torch.rand(2, generator=g) * size
        v = (torch.rand(2, generator=g) * 2 - 1) * max_speed
        tex = torch.stack([_value_noise(g, size, size, 6) for _ in range(ch)]) * 2 - 1
        shapes.append((r, p, v, tex, bool(torch.rand(1, generator=g) < 0.5)))
    yy, xx = torch.meshgrid(torch.arange(size).float(), torch.arange(size).float(), indexing="ij")
    out = []
    for t in range(T):
        oy = int(size / 2 + pan[0] * t) % size; ox = int(size / 2 + pan[1] * t) % size
        fr = bg[:, oy:oy + size, ox:ox + size].clone()
        for r, p, v, tex, square in shapes:
            cy, cx = p[0] + v[0] * t, p[1] + v[1] * t
            m = ((yy - cy).abs() < r) & ((xx - cx).abs() < r) if square else \
                ((yy - cy) ** 2 + (xx - cx) ** 2) < r * r
            fr = torch.where(m[None], tex, fr)
        out.append(fr)
    return torch.stack(out)


class ClipSource:
    """Yields batches f [B, C, ch, size, size]."""
    def __init__(self, args):
        self.args, self.files = args, []
        if args.data == "npy":
            self.files = sorted(glob.glob(os.path.join(args.path, "*.npy")))
            assert self.files, f"no .npy files in {args.path}"
            self.arrays = [np.load(f, mmap_mode="r") for f in self.files]
        elif args.data == "video":
            import imageio.v3 as iio
            self.files = sorted(sum([glob.glob(os.path.join(args.path, e))
                                     for e in ("*.mp4", "*.mov", "*.avi", "*.webm")], []))
            assert self.files, f"no videos in {args.path}"
            self.arrays = []
            for f in self.files:
                v = iio.imread(f, plugin="pyav")                         # [T, H, W, 3] uint8
                self.arrays.append((v.astype(np.float32) / 127.5 - 1).transpose(0, 3, 1, 2))
        if self.files:
            ch = self.arrays[0].shape[1]
            samp = np.concatenate([np.asarray(a[: min(8, len(a))], dtype=np.float64)
                                   .reshape(-1, ch, *a.shape[2:])
                                   .transpose(1, 0, 2, 3).reshape(ch, -1) for a in self.arrays[:16]], 1)
            self.mean = torch.tensor(samp.mean(1), dtype=torch.float32).view(1, 1, ch, 1, 1)
            self.std = torch.tensor(samp.std(1) + 1e-6, dtype=torch.float32).view(1, 1, ch, 1, 1)
            self.ch = ch
        else:
            self.ch = args.channels

    def batch(self, B: int, C: int, g: torch.Generator, split: str = "train") -> torch.Tensor:
        a, s = self.args, self.args.size
        if a.data == "synthetic":
            return torch.stack([synthetic_clip(C, s, self.ch, a.max_speed, g) for _ in range(B)])
        n = len(self.arrays)
        idx = [i for i in range(n) if (i % 10 == 0) == (split == "test")] or list(range(n))
        clips = []
        for _ in range(B):
            arr = self.arrays[idx[int(torch.randint(0, len(idx), (1,), generator=g))]]
            Tn, ch, H, W = arr.shape
            # Wan-style VAEs encode the first frame on its own; skip that latent by default
            start = 1 if (a.data == "npy" and getattr(a, "skip_first", True) and Tn > C) else 0
            t0 = int(torch.randint(start, max(start + 1, Tn - C + 1), (1,), generator=g))
            y0 = int(torch.randint(0, max(1, H - s + 1), (1,), generator=g))
            x0 = int(torch.randint(0, max(1, W - s + 1), (1,), generator=g))
            c = np.asarray(arr[t0:t0 + C, :, y0:y0 + s, x0:x0 + s], dtype=np.float32)
            clips.append(torch.from_numpy(c))
        f = torch.stack(clips)
        return (f - self.mean) / self.std


# --------------------------------------------------------------------------- metrics
def snr_db(ref: torch.Tensor, est: torch.Tensor) -> float:
    return float(10 * torch.log10(ref.var() / F.mse_loss(est, ref).clamp_min(1e-12)))


def motion(f: torch.Tensor) -> torch.Tensor:
    """Per-clip motion energy: mean squared frame difference. f [B, C, ...] -> [B]."""
    return (f[:, 1:] - f[:, :-1]).pow(2).flatten(1).mean(1)


@torch.no_grad()
def evaluate(model, f: torch.Tensor, bs: int = 16) -> dict:
    model.eval()
    est = torch.cat([model(f[i:i + bs]) for i in range(0, len(f), bs)])
    model.train()
    d_ref, d_est = f[:, 1:] - f[:, :-1], est[:, 1:] - est[:, :-1]
    m = motion(f)
    hi = m >= m.quantile(2 / 3)
    return dict(snr_db=snr_db(f, est), motion_snr_db=snr_db(d_ref, d_est),
                high_motion_snr_db=snr_db(f[hi], est[hi]),
                high_motion_motion_snr_db=snr_db(d_ref[hi], d_est[hi]),
                mse=float(F.mse_loss(est, f)))


@torch.no_grad()
def static_baseline(f: torch.Tensor, C: int, a: float) -> dict:
    """No network: assume nothing moves (every frame = scaled DC snapshot)."""
    T = fixed_code("avg", C, f.shape[-2], f.shape[-1]).to(f.device)
    g = snapshot(f, T)
    est = (g / math.sqrt(C))[:, None].expand_as(f)
    return dict(snr_db=snr_db(f, est), motion_snr_db=float("-inf"))


# --------------------------------------------------------------------------- decision rule
# Pre-registered thresholds (proposals: change them BEFORE looking at results, not after).
MARGIN_VS_LEARNED_DB = {2: 0.5, 4: 1.0, 8: 1.5, 16: 2.0}
MIN_GAIN_OVER_AVG_HIGH_MOTION_DB = 3.0


def decide(results: dict, C: int, floor_db: float | None = None) -> dict:
    r = results[C]
    coded = {k: v for k, v in r.items() if k in ("dcwalsh", "cacti", "randpm", "learned")}
    if not coded:                       # e.g. a run with only baseline arms; merge files first
        return dict(best_coded_arm=None, checks={}, go=None)
    best = max(coded, key=lambda k: coded[k]["snr_db"])
    checks = {}
    if "learned_ae" in r:
        margin = MARGIN_VS_LEARNED_DB.get(C, 2.0)
        checks["close_to_learned_compressor"] = coded[best]["snr_db"] >= r["learned_ae"]["snr_db"] - margin
    if "avg" in r:
        checks["coding_carries_motion"] = (coded[best]["high_motion_motion_snr_db"]
                                           >= r["avg"]["high_motion_motion_snr_db"]
                                           + MIN_GAIN_OVER_AVG_HIGH_MOTION_DB)
    if floor_db is not None:
        checks["above_absolute_floor"] = coded[best]["snr_db"] >= floor_db
    return dict(best_coded_arm=best, checks=checks, go=bool(checks) and all(checks.values()))


# --------------------------------------------------------------------------- main
def train_arm(kind, C, src, args, test_f, device):
    torch.manual_seed(args.seed)
    model = make_arm(kind, C, src.ch, args.width, args.a, args.seed).to(device)
    # The learned code gets its own parameter group: its own learning rate, no weight decay,
    # and it can be frozen for the first --code-freeze-steps steps while the unmixer warms up.
    code = [model.learned.w] if getattr(model, "learned", None) is not None else []
    rest = [p for p in model.parameters() if all(p is not q for q in code)]
    groups = [dict(params=rest)]
    if code:
        groups.append(dict(params=code, lr=args.lr * args.code_lr_mult, weight_decay=0.0))
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    g = torch.Generator().manual_seed(args.seed + 1)
    t0 = time.time()
    for step in range(args.steps):
        if code:
            code[0].requires_grad_(step >= args.code_freeze_steps)
        f = src.batch(args.batch, C, g).to(device)
        loss = F.mse_loss(model(f), f)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if args.log_every and (step + 1) % args.log_every == 0:
            print(f"  C={C} {kind:10s} step {step + 1:6d}  loss {loss.item():.4f}", flush=True)
    res = evaluate(model, test_f.to(device))
    res.update(params=sum(p.numel() for p in model.parameters()), train_sec=round(time.time() - t0, 1))
    return res


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", choices=["synthetic", "npy", "video"], default="synthetic")
    p.add_argument("--path", default=None)
    p.add_argument("--C", type=int, nargs="+", default=[4, 8])
    p.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    p.add_argument("--size", type=int, default=64, help="spatial crop (positions)")
    p.add_argument("--channels", type=int, default=3, help="synthetic only")
    p.add_argument("--max-speed", type=float, default=3.0, help="synthetic only, px/frame")
    p.add_argument("--a", type=float, default=0.7071, help="DC share of the dcwalsh code")
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--code-lr-mult", type=float, default=1.0,
                   help="learned arm only: code learning rate = lr x this (e.g. 0.1)")
    p.add_argument("--code-freeze-steps", type=int, default=0,
                   help="learned arm only: keep the code fixed for this many steps (e.g. 5000)")
    p.add_argument("--test-clips", type=int, default=256)
    p.add_argument("--floor-db", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--out", default="gate0_results.json")
    args = p.parse_args(argv)

    src = ClipSource(args)
    results, decisions = {}, {}
    for C in args.C:
        gt = torch.Generator().manual_seed(10_000 + C)
        test_f = src.batch(args.test_clips, C, gt, split="test")
        results[C] = {"static_no_network": static_baseline(test_f, C, args.a)}
        for kind in args.arms:
            print(f"training C={C} arm={kind}", flush=True)
            results[C][kind] = train_arm(kind, C, src, args, test_f, args.device)
            r = results[C][kind]
            print(f"  -> SNR {r['snr_db']:.2f} dB | motion SNR {r['motion_snr_db']:.2f} dB | "
                  f"high-motion SNR {r['high_motion_snr_db']:.2f} dB | params {r['params']:,}", flush=True)
            with open(args.out, "w") as fh:          # save after every arm: nothing is lost
                json.dump(dict(args=vars(args), results=results, decisions=decisions), fh, indent=1)
        decisions[C] = decide(results, C, args.floor_db)
        print(f"C={C} decision: {decisions[C]}", flush=True)

    with open(args.out, "w") as fh:
        json.dump(dict(args=vars(args), results=results, decisions=decisions), fh, indent=1)
    print(report(results, decisions))
    return results, decisions


def report(results, decisions) -> str:
    lines = ["", "| C | arm | SNR dB | motion SNR dB | high-motion SNR dB | high-motion motion SNR dB | params |",
             "|---|---|---|---|---|---|---|"]
    for C, r in results.items():
        for k, v in r.items():
            if k == "static_no_network":
                lines.append(f"| {C} | static (no network) | {v['snr_db']:.2f} | - | - | - | 0 |")
            else:
                lines.append(f"| {C} | {k} | {v['snr_db']:.2f} | {v['motion_snr_db']:.2f} | "
                             f"{v['high_motion_snr_db']:.2f} | {v['high_motion_motion_snr_db']:.2f} | "
                             f"{v['params']:,} |")
        d = decisions[C]
        if d["go"] is None:
            continue
        lines.append(f"| {C} | decision | {'GO' if d['go'] else 'NO-GO'} "
                     f"(best coded: {d['best_coded_arm']}; {d['checks']}) | | | | |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
