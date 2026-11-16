"""Shared helpers for Snapshot Engine Gate 1.

Conventions
-----------
* A latent clip is [B, 16, F, H, W] in Wan's own normalised space: (z - latents_mean) / latents_std,
  exactly what the Wan transformer was trained on. F = 1 + C*n (81 video frames -> F = 21).
* Latent frame 0 encodes video frame 0 on its own (causal Wan VAE). It passes through uncoded.
  The remaining C*n frames are coded in groups of C into n snapshots, so the transformer sees
  a "snapshot sequence" of 1 + n token frames: [z0, S1, ..., Sn].
* Snapshots are rescaled per channel to unit variance ("snapshot stats") before diffusion and
  scaled back before unmixing. The scaling is linear, so Gaussian noise stays Gaussian.
* The code and the unmixer come from train_unmixer.py (the Gate 0 learned-code arm, retrained
  on Wan-normalised latents). Code and network classes are imported from ../snapshot_gate.py.
"""
from __future__ import annotations
import hashlib, json, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))      # snapshot_gate.py sits one level up
import snapshot_gate as sg                     # noqa: E402

DEFAULT_MODEL = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
TEXT_LEN = 512                                 # Wan pads T5 embeddings to 512 tokens


# --------------------------------------------------------------------------- misc
def device_default() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"     # ROCm builds also report "cuda"


def is_test_id(clip_id: str, test_percent: int = 3) -> bool:
    """Deterministic held-out split by hash of the clip id."""
    return int(hashlib.md5(clip_id.encode()).hexdigest(), 16) % 100 < test_percent


def read_manifest(data_dir: str, split: str | None = None) -> list[dict]:
    path = os.path.join(data_dir, "manifest.jsonl")
    rows = [json.loads(l) for l in open(path) if l.strip()]
    if split is not None:
        rows = [r for r in rows if r["split"] == split]
    return rows


def load_latent(data_dir: str, row: dict) -> torch.Tensor:
    """[16, F, H, W] float32, Wan-normalised."""
    return torch.from_numpy(np.load(os.path.join(data_dir, row["latent"])).astype(np.float32))


def load_text(data_dir: str, name: str) -> torch.Tensor:
    """Cached T5 embedding, zero-padded to [TEXT_LEN, 4096] like Wan's pipeline does."""
    e = torch.from_numpy(np.load(os.path.join(data_dir, "text", name + ".npy")).astype(np.float32))
    out = torch.zeros(TEXT_LEN, e.shape[1])
    out[: min(len(e), TEXT_LEN)] = e[:TEXT_LEN]
    return out


def sync(device) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def write_video(frames: np.ndarray, path: str, fps: int = 16, crf: int = 18) -> None:
    """frames: [T,H,W,3] uint8 or float in [0,1] -> H.264 mp4 (yuv420p) via PyAV."""
    import av
    if frames.dtype != np.uint8:
        frames = (np.clip(frames, 0, 1) * 255 + 0.5).astype(np.uint8)
    T, H, W, _ = frames.shape
    H2, W2 = H - H % 2, W - W % 2
    with av.open(path, "w") as c:
        s = c.add_stream("libx264", rate=fps)
        s.width, s.height, s.pix_fmt = W2, H2, "yuv420p"
        s.options = {"crf": str(crf), "preset": "veryfast"}
        for f in frames:
            for pkt in s.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(f[:H2, :W2]), format="rgb24")):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)


def read_video(path: str, max_frames: int | None = None) -> np.ndarray:
    """mp4 -> [T,H,W,3] uint8."""
    import av
    out = []
    with av.open(path) as c:
        for fr in c.decode(video=0):
            out.append(fr.to_ndarray(format="rgb24"))
            if max_frames and len(out) >= max_frames:
                break
    return np.stack(out)


# --------------------------------------------------------------------------- Wan normalisation
def wan_stats(vae_config) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor(vae_config.latents_mean, dtype=torch.float32)
    std = torch.tensor(vae_config.latents_std, dtype=torch.float32)
    return mean, std


def to_wan_space(z: torch.Tensor, mean, std) -> torch.Tensor:
    """Raw VAE latent [B,16,F,H,W] -> Wan-normalised."""
    return (z - mean.view(1, -1, 1, 1, 1).to(z)) / std.view(1, -1, 1, 1, 1).to(z)


def from_wan_space(z: torch.Tensor, mean, std) -> torch.Tensor:
    return z * std.view(1, -1, 1, 1, 1).to(z) + mean.view(1, -1, 1, 1, 1).to(z)


# --------------------------------------------------------------------------- snapshot coding
def n_snapshots(F: int, C: int) -> int:
    assert (F - 1) % C == 0, f"latent frames F={F} must be 1 + C*n (C={C}); use 4n+1 video frames with n*4 % C == 0"
    return (F - 1) // C


def group_frames(z: torch.Tensor, C: int) -> torch.Tensor:
    """[B,ch,F,H,W] -> groups [B, n, C, ch, H, W] (frames 1..F-1)."""
    B, ch, F, H, W = z.shape
    n = n_snapshots(F, C)
    return z[:, :, 1:1 + C * n].reshape(B, ch, n, C, H, W).permute(0, 2, 3, 1, 4, 5)


def ungroup_frames(z0: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    """z0 [B,ch,1,H,W] + groups [B,n,C,ch,H,W] -> [B,ch,1+n*C,H,W]."""
    B, n, C, ch, H, W = f.shape
    return torch.cat([z0, f.permute(0, 3, 1, 2, 4, 5).reshape(B, ch, n * C, H, W)], 2)


def encode_snapshots(z: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """Frame latents [B,ch,F,H,W] -> raw snapshot sequence [B,ch,1+n,H,W] (unscaled)."""
    f = group_frames(z, T.shape[0])                          # [B,n,C,ch,H,W]
    g = (f * T[None, None, :, None]).sum(2)                  # [B,n,ch,H,W]
    return torch.cat([z[:, :, :1], g.permute(0, 2, 1, 3, 4)], 2)


def scale_snapshots(seq: torch.Tensor, smean: torch.Tensor, sstd: torch.Tensor) -> torch.Tensor:
    out = seq.clone()
    out[:, :, 1:] = (seq[:, :, 1:] - smean.view(1, -1, 1, 1, 1).to(seq)) / sstd.view(1, -1, 1, 1, 1).to(seq)
    return out


def unscale_snapshots(seq: torch.Tensor, smean: torch.Tensor, sstd: torch.Tensor) -> torch.Tensor:
    out = seq.clone()
    out[:, :, 1:] = seq[:, :, 1:] * sstd.view(1, -1, 1, 1, 1).to(seq) + smean.view(1, -1, 1, 1, 1).to(seq)
    return out


@torch.no_grad()
def unmix_sequence(seq: torch.Tensor, unmixer: "sg.CodedArm") -> torch.Tensor:
    """Raw (unscaled) snapshot sequence [B,ch,1+n,H,W] -> frame latents [B,ch,1+n*C,H,W]."""
    B, ch, K, H, W = seq.shape
    T = unmixer.code(H, W, seq.device).to(seq.dtype)
    g = seq[:, :, 1:].permute(0, 2, 1, 3, 4).reshape(B * (K - 1), ch, H, W)
    f = unmixer.decode(g, T).view(B, K - 1, unmixer.C, ch, H, W)
    return ungroup_frames(seq[:, :, :1], f)


# --------------------------------------------------------------------------- learned compressor (zero-shot arm c)
@torch.no_grad()
def ae_encode_sequence(z: torch.Tensor, ae: "sg.LearnedAE") -> torch.Tensor:
    """Frame latents -> z0 + learned-compressor latents, one per group: [B,ch,1+n,H,W] (unscaled)."""
    f = group_frames(z, ae.C)
    B, n, C, ch, H, W = f.shape
    x = f.reshape(B * n, C * ch, H, W)
    pos = sg.tile_position(H, W, ae.tile, z.device)[None].expand(B * n, -1, -1, -1).to(z.dtype)
    e = ae.enc(torch.cat([x, pos], 1)).view(B, n, ch, H, W)
    return torch.cat([z[:, :, :1], e.permute(0, 2, 1, 3, 4)], 2)


# --------------------------------------------------------------------------- checkpoints
def load_arm(path: str, device="cpu"):
    """Returns (model, ckpt). model is sg.CodedArm ('learned') or sg.LearnedAE ('learned_ae')."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = sg.make_arm(ck["kind"], ck["C"], ck["ch"], ck["width"], ck.get("a", 0.7071), ck.get("seed", 0))
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck
