"""Gate 1, step 5: fine-tune Wan 2.1 T2V 1.3B to generate snapshot sequences (or frames, as the control).

  --mode snapshot  x0 = [z0, S1..Sn]: latent frame 0 uncoded + n coded snapshots, scaled to unit variance
  --mode frames    x0 = the first 1+n latent frames (same token count; control for "is snapshot space harder?")

Objective: Wan's own rectified-flow loss. x_t = (1 - s) x0 + s eps, target = eps - x0, with the timestep
sigma s drawn from a shifted uniform distribution (--flow-shift). Captions are dropped 10% of the time
(replaced by the empty-prompt embedding) so classifier-free guidance keeps working.

The transformer is not modified: input/output stay 16 channels and the sequence is just shorter in time.
Temporal RoPE uses plain consecutive indices 0..n.

Memory: fp32 master weights + AdamW states ~21 GB for 1.3B, plus activations (gradient checkpointing on
by default). Saves the transformer (diffusers format, about 5 GB) every --save-every steps and at the end,
and a resumable training state (weights, optimizer, step; about 16 GB, latest only) as train_state.pt.

Example:
  python gate1/train.py --data gate1_data --unmixer gate1_runs/unmixer_learned.pt --mode snapshot \
      --out gate1_runs/wan_snapshot --steps 12500 --grad-accum 8
"""
from __future__ import annotations
import argparse, json, math, os, random, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (DEFAULT_MODEL, TEXT_LEN, device_default, encode_snapshots, load_arm, load_text,  # noqa: E402
                    n_snapshots, read_manifest, scale_snapshots)


class Clips(torch.utils.data.Dataset):
    def __init__(self, data_dir, rows):
        self.d, self.rows = data_dir, rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        z = torch.from_numpy(np.load(os.path.join(self.d, r["latent"])).astype(np.float32))   # [16,F,H,W]
        return z, load_text(self.d, r["id"])


def sample_sigma(n, shift, gen, device):
    u = torch.rand(n, generator=gen, device="cpu").to(device)
    return shift * u / (1 + (shift - 1) * u)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--unmixer", default=None, help="learned-code checkpoint (required for --mode snapshot)")
    p.add_argument("--mode", choices=["snapshot", "frames"], default="snapshot")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--init", default=None, help="start from this transformer folder instead of the stock one")
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=12500, help="optimizer steps")
    p.add_argument("--batch", type=int, default=1, help="clips per micro-batch")
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--flow-shift", type=float, default=3.0)
    p.add_argument("--cfg-drop", type=float, default=0.1)
    p.add_argument("--no-grad-ckpt", action="store_true")
    p.add_argument("--save-every", type=int, default=2500)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true", help="continue from <out>/train_state.pt")
    p.add_argument("--device", default=device_default())
    a = p.parse_args(argv)

    from diffusers import WanTransformer3DModel
    dev = a.device
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed); random.seed(a.seed)

    C = smean = sstd = coded = None
    if a.mode == "snapshot":
        assert a.unmixer, "--unmixer is required for --mode snapshot"
        coded, ck = load_arm(a.unmixer, dev)
        for q in coded.parameters():
            q.requires_grad_(False)
        C, smean, sstd = ck["C"], ck["snap_mean"].to(dev), ck["snap_std"].to(dev)
    else:
        C = 4

    src = a.init or a.model
    sub = None if a.init else "transformer"
    model = WanTransformer3DModel.from_pretrained(src, subfolder=sub, torch_dtype=torch.float32).to(dev)
    if not a.no_grad_ckpt:
        model.enable_gradient_checkpointing()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.999), weight_decay=a.weight_decay)
    lr_at = lambda s: a.lr * min(1.0, (s + 1) / max(1, a.warmup))  # noqa: E731  warm-up then constant
    step = 0
    state_path = os.path.join(a.out, "train_state.pt")
    if a.resume and os.path.exists(state_path):
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); step = st["step"]
        print(f"resumed at step {step}")

    rows = read_manifest(a.data, "train")
    dl = torch.utils.data.DataLoader(Clips(a.data, rows), batch_size=a.batch, shuffle=True, drop_last=True,
                                     num_workers=a.num_workers, pin_memory=dev.startswith("cuda"),
                                     persistent_workers=a.num_workers > 0)
    empty = load_text(a.data, "_empty").to(dev)
    gen = torch.Generator().manual_seed(a.seed + step)
    json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=1)
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    autocast = torch.autocast(device_type="cuda" if dev.startswith("cuda") else "cpu", dtype=torch.bfloat16,
                              enabled=dev.startswith("cuda"))

    def batches():
        while True:
            for b in dl:
                yield b

    it = batches()
    t0, run, seen = time.time(), None, 0
    print(f"fine-tuning ({a.mode}) on {len(rows):,} clips: {a.steps} steps x {a.batch * a.grad_accum} clips", flush=True)
    while step < a.steps:
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for _ in range(a.grad_accum):
            z, emb = next(it)
            z, emb = z.to(dev, non_blocking=True), emb.to(dev, non_blocking=True)
            K = 1 + n_snapshots(z.shape[2], C)
            if a.mode == "snapshot":
                with torch.no_grad():
                    T = coded.code(z.shape[-2], z.shape[-1], dev)
                    x0 = scale_snapshots(encode_snapshots(z, T), smean, sstd)
            else:
                x0 = z[:, :, :K]
            drop = torch.rand(len(z), generator=gen) < a.cfg_drop
            if drop.any():
                emb = emb.clone(); emb[drop.to(dev)] = empty
            s = sample_sigma(len(z), a.flow_shift, gen, dev)
            eps = torch.randn(x0.shape, generator=gen).to(dev)
            xt = (1 - s.view(-1, 1, 1, 1, 1)) * x0 + s.view(-1, 1, 1, 1, 1) * eps
            with autocast:
                v = model(hidden_states=xt, timestep=s * 1000.0, encoder_hidden_states=emb, return_dict=False)[0]
            loss = torch.mean((v.float() - (eps - x0)) ** 2) / a.grad_accum
            loss.backward()
            tot += loss.item()
            seen += len(z)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        run = tot if run is None else 0.98 * run + 0.02 * tot
        if step % a.log_every == 0 or step == a.steps:
            el = time.time() - t0
            mem = torch.cuda.max_memory_allocated() / 2 ** 30 if dev.startswith("cuda") else 0
            rec = dict(step=step, loss=tot, loss_ema=run, grad_norm=float(gn), lr=lr_at(step - 1),
                       clips_per_s=seen / el, mem_gb=round(mem, 1))
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(f"  step {step:6d}/{a.steps}  loss {tot:.4f} (ema {run:.4f})  gnorm {float(gn):.2f}  "
                  f"{seen / el:.2f} clips/s  ~{(a.steps - step) * a.grad_accum * a.batch / max(seen / el, 1e-9) / 3600:.1f} h left"
                  f"  peak {mem:.0f} GB", flush=True)
        if step % a.save_every == 0 or step == a.steps:
            model.save_pretrained(os.path.join(a.out, f"step{step:06d}"))        # fp32, ~5 GB
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step), state_path)
            print(f"  saved {os.path.join(a.out, f'step{step:06d}')}", flush=True)
    final = os.path.join(a.out, "final")
    if os.path.lexists(final):
        os.remove(final)
    os.symlink(f"step{step:06d}", final)
    print(f"done; final transformer: {final}")


if __name__ == "__main__":
    main()
