"""Encode a folder of videos into normalised Wan 2.1 VAE latents for Gate 0.

UNTESTED in the authoring environment (no GPU, no model download). Written against the
diffusers AutoencoderKLWan API; check it on one video before a full run.

Output: one .npy per video chunk, shape [T_lat, 16, H/8, W/8], float16, normalised with the
VAE's own latents_mean / latents_std, which is what Wan's diffusion transformer sees.

    pip install diffusers transformers accelerate imageio[pyav]
    python encode_wan_latents.py --videos my_videos/ --out latents/ --height 480 --width 832

Then:  python snapshot_gate.py --data npy --path latents/ --C 4 8 --size 32 --steps 20000
"""
import argparse, glob, os
import numpy as np
import torch
import torch.nn.functional as F


def load_video(path, height, width, max_frames):
    import imageio.v3 as iio
    v = iio.imread(path, plugin="pyav")[:max_frames]                 # [T, H, W, 3] uint8
    x = torch.from_numpy(v).float().permute(3, 0, 1, 2) / 127.5 - 1  # [3, T, H, W] in [-1, 1]
    x = F.interpolate(x, size=(height, width), mode="bilinear", align_corners=False)
    T = (x.shape[1] - 1) // 4 * 4 + 1                                # Wan VAE wants 4n+1 frames
    return x[:, :T]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--videos", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--frames", type=int, default=81, help="frames per chunk (4n+1)")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    from diffusers import AutoencoderKLWan
    vae = AutoencoderKLWan.from_pretrained(a.model, subfolder="vae", torch_dtype=torch.float32)
    vae = vae.to(a.device).eval()
    mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(a.device)
    std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(a.device)
    os.makedirs(a.out, exist_ok=True)

    files = sorted(sum([glob.glob(os.path.join(a.videos, e)) for e in ("*.mp4", "*.mov", "*.webm", "*.avi")], []))
    for i, path in enumerate(files):
        x = load_video(path, a.height, a.width, max_frames=10_000)
        for c0 in range(0, x.shape[1] - a.frames + 1, a.frames - 1):
            chunk = x[:, c0:c0 + a.frames][None].to(a.device)
            with torch.no_grad():
                lat = vae.encode(chunk).latent_dist.mode()             # [1, 16, T_lat, h, w]
            lat = ((lat - mean) / std)[0].permute(1, 0, 2, 3)          # [T_lat, 16, h, w]
            name = f"{os.path.splitext(os.path.basename(path))[0]}_{c0:05d}.npy"
            np.save(os.path.join(a.out, name), lat.cpu().numpy().astype(np.float16))
        print(f"[{i + 1}/{len(files)}] {path}", flush=True)


if __name__ == "__main__":
    main()
