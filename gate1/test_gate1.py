"""Offline end-to-end test of every Gate 1 script, on CPU, with tiny random Wan models. ~2-4 minutes.

Builds a miniature Wan pipeline (random weights, same classes and file layout as the real one),
a few synthetic captioned videos, then runs prepare_data -> train_unmixer (both arms) -> zero_shot ->
train (snapshot, frames, resume) -> sample (snapshot, wan) -> evaluate. Checks shapes and outputs only;
the numbers are meaningless. Run it after unzipping to confirm the environment works:

  python gate1/test_gate1.py            # uses a temporary folder, removed afterwards
  python gate1/test_gate1.py --keep     # keep it for inspection
"""
from __future__ import annotations
import argparse, csv, json, os, shutil, subprocess, sys, tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def make_tiny_wan(path):
    import torch
    from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanPipeline, WanTransformer3DModel
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, UMT5Config, UMT5EncoderModel
    torch.manual_seed(0)
    words = "a the and of to in on with is are person dog car man woman runs walks red blue sky street".split()
    vocab = {"<pad>": 0, "</s>": 1, "<unk>": 2, **{w: i + 3 for i, w in enumerate(words)}}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>")); tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="<pad>", eos_token="</s>", unk_token="<unk>")
    te = UMT5EncoderModel(UMT5Config(vocab_size=len(vocab), d_model=32, d_kv=8, d_ff=64, num_layers=1, num_heads=4))
    vae = AutoencoderKLWan(base_dim=16, z_dim=16, dim_mult=[1, 2, 4, 4], num_res_blocks=1)
    tr = WanTransformer3DModel(num_attention_heads=2, attention_head_dim=8, text_dim=32, freq_dim=16, ffn_dim=32,
                               num_layers=2, rope_max_seq_len=32)
    sch = UniPCMultistepScheduler(prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=3.0)
    WanPipeline(tokenizer=tok, text_encoder=te, vae=vae, scheduler=sch, transformer=tr).save_pretrained(path)


def make_videos(folder, csv_path, n, frames, size):
    from common import write_video
    os.makedirs(folder, exist_ok=True)
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        bg = rng.integers(0, 255, 3)
        v = np.zeros((frames, size, size, 3), np.uint8) + bg.astype(np.uint8)
        x, y, vx, vy = rng.integers(8, size - 24, 2).tolist() + rng.integers(-2, 3, 2).tolist()
        for t in range(frames):
            v[t, max(0, y + vy * t):max(0, y + vy * t) + 16, max(0, x + vx * t):max(0, x + vx * t) + 16] = (255, 40, 40)
        name = f"clip{i:03d}.mp4"
        write_video(v, os.path.join(folder, name), fps=16)
        rows.append(dict(video=name, caption=f"a red car runs on the street {i}"))
    rows.append(dict(video="too_short.mp4", caption="a dog"))
    write_video(np.zeros((5, size, size, 3), np.uint8), os.path.join(folder, "too_short.mp4"))
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["video", "caption"]); w.writeheader(); w.writerows(rows)


def run(args, cwd):
    print("\n$ python " + " ".join(args), flush=True)
    r = subprocess.run([sys.executable] + args, cwd=cwd)
    assert r.returncode == 0, f"failed: {args[0]}"


def main():
    p = argparse.ArgumentParser(); p.add_argument("--keep", action="store_true"); a = p.parse_args()
    root = tempfile.mkdtemp(prefix="gate1_test_")
    g = lambda s: os.path.join(HERE, s)  # noqa: E731
    try:
        model = os.path.join(root, "tiny-wan"); make_tiny_wan(model)
        make_videos(os.path.join(root, "clips"), os.path.join(root, "caps.csv"), 24, 33, 64)
        d, r, e = (os.path.join(root, x) for x in ("data", "runs", "eval"))
        cpu = ["--device", "cpu"]
        run([g("prepare_data.py"), "--videos", os.path.join(root, "clips"), "--captions", os.path.join(root, "caps.csv"),
             "--out", d, "--model", model, "--frames", "33", "--height", "64", "--width", "64", "--max-clips", "24",
             "--vae-dtype", "fp32", "--test-percent", "25", "--n-eval", "3", *cpu], root)
        rows = [json.loads(l) for l in open(os.path.join(d, "manifest.jsonl"))]
        z = np.load(os.path.join(d, rows[0]["latent"]))
        assert z.shape == (16, 9, 8, 8), z.shape
        assert any(x["split"] == "test" for x in rows) and any(x["split"] == "train" for x in rows)
        assert os.path.exists(os.path.join(d, "text", "_empty.npy"))
        small = ["--crop", "8", "--width", "8", "--steps", "30", "--batch", "4", "--num-workers", "0",
                 "--code-freeze-steps", "10", "--eval-clips", "4", "--stats-clips", "8", "--log-every", "10", *cpu]
        run([g("train_unmixer.py"), "--data", d, "--arm", "learned", "--out", os.path.join(r, "unmixer_learned.pt"), *small], root)
        run([g("train_unmixer.py"), "--data", d, "--arm", "learned_ae", "--out", os.path.join(r, "unmixer_ae.pt"), *small], root)
        run([g("zero_shot.py"), "--data", d, "--unmixer", os.path.join(r, "unmixer_learned.pt"), "--ae",
             os.path.join(r, "unmixer_ae.pt"), "--model", model, "--n-clips", "3", "--out", os.path.join(r, "zs.json"), *cpu], root)
        tr = ["--data", d, "--model", model, "--steps", "4", "--grad-accum", "2", "--num-workers", "0",
              "--save-every", "2", "--log-every", "1", "--warmup", "2", *cpu]
        run([g("train.py"), "--mode", "snapshot", "--unmixer", os.path.join(r, "unmixer_learned.pt"),
             "--out", os.path.join(r, "wan_snapshot"), *tr], root)
        run([g("train.py"), "--mode", "snapshot", "--unmixer", os.path.join(r, "unmixer_learned.pt"),
             "--out", os.path.join(r, "wan_snapshot"), "--resume", *tr[:5], "6", *tr[6:]], root)
        run([g("train.py"), "--mode", "frames", "--out", os.path.join(r, "wan_frames"), *tr], root)
        smp = ["--model", model, "--prompts", os.path.join(d, "eval_prompts.jsonl"), "--n", "3", "--num-frames", "33",
               "--height", "64", "--width", "64", *cpu]
        run([g("sample.py"), "--system", "snapshot", "--transformer", os.path.join(r, "wan_snapshot", "final"),
             "--unmixer", os.path.join(r, "unmixer_learned.pt"), "--out", os.path.join(e, "snapshot"), "--steps", "3", *smp], root)
        run([g("sample.py"), "--system", "wan", "--out", os.path.join(e, "wan50"), "--steps", "3", *smp], root)
        run([g("sample.py"), "--system", "wan", "--out", os.path.join(e, "wan_eq"), "--steps", "1", *smp], root)
        run([g("evaluate.py"), "--runs", f"snapshot={e}/snapshot", f"wan50={e}/wan50", f"wan_eq={e}/wan_eq",
             "--reference", "wan50", "--equal-compute", "wan_eq", "--grid", "2", "--out", os.path.join(e, "report")], root)
        from common import read_video
        v = read_video(os.path.join(e, "snapshot", "videos", sorted(os.listdir(os.path.join(e, "snapshot", "videos")))[0]))
        assert v.shape[0] == 33 and v.shape[1:3] == (64, 64), v.shape
        print("\nALL GATE 1 SCRIPTS RAN END TO END")
    finally:
        if a.keep:
            print(f"kept {root}")
        else:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
