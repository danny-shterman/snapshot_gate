"""Gate 1, step 6: generate evaluation videos and time every stage.

Systems (same sampler, same text encoder, same VAE, same prompts and seeds):
  --system snapshot   fine-tuned transformer samples the snapshot sequence [z0, S1..Sn]
                      -> unscale -> unmixer -> 1 + n*C latent frames -> VAE decode
  --system wan        stock (or any) Wan transformer samples all latent frames -> VAE decode

Each video is timed in four stages, each bracketed by a device synchronize:
  text (T5 encode of prompt + negative prompt), denoise (the full sampling loop, CFG included),
  unmix (snapshot only), decode (VAE decode to pixels). One untimed warm-up video runs first.

Outputs in --out: videos/<k>_<id>.mp4, timing.jsonl (one row per video), summary.json (means).

Examples:
  python gate1/sample.py --system snapshot --transformer gate1_runs/wan_snapshot/final \
      --unmixer gate1_runs/unmixer_learned.pt --prompts gate1_data/eval_prompts.jsonl --out gate1_eval/snapshot
  python gate1/sample.py --system wan --prompts gate1_data/eval_prompts.jsonl --out gate1_eval/wan50
  python gate1/sample.py --system wan --steps 7 --prompts gate1_data/eval_prompts.jsonl --out gate1_eval/wan_eq
"""
from __future__ import annotations
import argparse, json, os, statistics, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (DEFAULT_MODEL, device_default, from_wan_space, load_arm, sync, unmix_sequence,  # noqa: E402
                    unscale_snapshots, wan_stats, write_video)
from prepare_data import NEGATIVE_PROMPT  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--system", choices=["snapshot", "wan"], required=True)
    p.add_argument("--model", default=DEFAULT_MODEL, help="base Wan pipeline (text encoder, VAE, scheduler)")
    p.add_argument("--transformer", default=None, help="transformer folder; default: the base model's own")
    p.add_argument("--unmixer", default=None, help="learned-code checkpoint (snapshot system)")
    p.add_argument("--prompts", required=True, help="eval_prompts.jsonl from prepare_data.py, or a .txt file")
    p.add_argument("--n", type=int, default=100, help="number of prompts")
    p.add_argument("--out", required=True)
    p.add_argument("--num-frames", type=int, default=81)
    p.add_argument("--height", type=int, default=480); p.add_argument("--width", type=int, default=832)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance", type=float, default=5.0)
    p.add_argument("--flow-shift", type=float, default=3.0)
    p.add_argument("--negative-prompt", default=NEGATIVE_PROMPT)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--save-latents", action="store_true")
    p.add_argument("--warmup", type=int, default=1, help="untimed warm-up generations")
    p.add_argument("--device", default=device_default())
    a = p.parse_args(argv)

    from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanPipeline, WanTransformer3DModel
    dev = a.device
    dtype = torch.bfloat16 if dev.startswith("cuda") else torch.float32
    os.makedirs(os.path.join(a.out, "videos"), exist_ok=True)

    tr = (WanTransformer3DModel.from_pretrained(a.transformer, torch_dtype=dtype) if a.transformer
          else WanTransformer3DModel.from_pretrained(a.model, subfolder="transformer", torch_dtype=dtype))
    vae = AutoencoderKLWan.from_pretrained(a.model, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(a.model, transformer=tr, vae=vae, torch_dtype=dtype).to(dev)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=a.flow_shift)
    pipe.set_progress_bar_config(disable=True)
    mean, std = wan_stats(vae.config)
    ts = pipe.vae_scale_factor_temporal

    F_lat = (a.num_frames - 1) // ts + 1
    unmixer = None
    if a.system == "snapshot":
        assert a.unmixer, "--unmixer is required for --system snapshot"
        unmixer, ck = load_arm(a.unmixer, dev)
        C = ck["C"]
        smean, sstd = ck["snap_mean"].to(dev), ck["snap_std"].to(dev)
        assert (F_lat - 1) % C == 0, f"{a.num_frames} frames -> {F_lat} latent frames, not 1 + {C}n"
        K = 1 + (F_lat - 1) // C
        gen_frames = (K - 1) * ts + 1          # what the pipeline must be asked for to get K latent frames
    else:
        K, gen_frames = F_lat, a.num_frames

    if a.prompts.endswith(".jsonl"):
        items = [json.loads(l) for l in open(a.prompts) if l.strip()]
    else:
        items = [dict(id=f"p{i:03d}", caption=l.strip()) for i, l in enumerate(open(a.prompts)) if l.strip()]
    items = items[: a.n]
    json.dump(dict(vars(a), token_frames=K, latent_frames=F_lat, pipeline_frames=gen_frames),
              open(os.path.join(a.out, "config.json"), "w"), indent=1)
    print(f"{a.system}: {len(items)} prompts, {a.steps} steps, transformer sees {K} latent frames "
          f"(video {a.num_frames} frames)", flush=True)

    def generate(cap, seed):
        tm = {}
        sync(dev); t = time.perf_counter()
        pe, ne = pipe.encode_prompt(prompt=cap, negative_prompt=a.negative_prompt, do_classifier_free_guidance=a.guidance > 1,
                                    max_sequence_length=512, device=dev)
        sync(dev); tm["text"] = time.perf_counter() - t
        g = torch.Generator(device=dev).manual_seed(seed)
        t = time.perf_counter()
        lat = pipe(prompt_embeds=pe, negative_prompt_embeds=ne, height=a.height, width=a.width,
                   num_frames=gen_frames, num_inference_steps=a.steps, guidance_scale=a.guidance,
                   generator=g, output_type="latent").frames                         # [1,16,K,h,w] Wan-normalised
        sync(dev); tm["denoise"] = time.perf_counter() - t
        t = time.perf_counter()
        if unmixer is not None:
            lat = unmix_sequence(unscale_snapshots(lat.float(), smean, sstd), unmixer)
        sync(dev); tm["unmix"] = time.perf_counter() - t
        t = time.perf_counter()
        with torch.no_grad():
            video = vae.decode(from_wan_space(lat.float(), mean, std).to(vae.dtype), return_dict=False)[0]
        frames = pipe.video_processor.postprocess_video(video, output_type="np")[0]
        sync(dev); tm["decode"] = time.perf_counter() - t
        tm["total"] = sum(tm.values())
        return frames, lat, tm

    for _ in range(a.warmup):
        generate(items[0]["caption"], a.seed)
    rows = []
    with open(os.path.join(a.out, "timing.jsonl"), "w") as log:
        for k, it in enumerate(items):
            frames, lat, tm = generate(it["caption"], a.seed + k)
            name = f"{k:03d}_{it['id']}"
            write_video(frames, os.path.join(a.out, "videos", name + ".mp4"), fps=a.fps)
            if a.save_latents:
                np.save(os.path.join(a.out, "videos", name + ".npy"), lat[0].cpu().numpy().astype(np.float16))
            rec = dict(k=k, id=it["id"], caption=it["caption"], frames=int(len(frames)), **{f"{s}_s": v for s, v in tm.items()})
            log.write(json.dumps(rec) + "\n"); log.flush(); rows.append(rec)
            print(f"  [{k + 1}/{len(items)}] " + "  ".join(f"{s} {v:.2f}s" for s, v in tm.items()), flush=True)

    summ = {f"{s}_s": statistics.mean(r[f"{s}_s"] for r in rows) for s in ("text", "denoise", "unmix", "decode", "total")}
    summ.update(system=a.system, steps=a.steps, n=len(rows), token_frames=K)
    json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print("mean per video: " + "  ".join(f"{k} {v:.2f}" for k, v in summ.items() if k.endswith("_s")))


if __name__ == "__main__":
    main()
