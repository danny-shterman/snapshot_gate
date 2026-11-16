python3 -m venv venv
source venv/bin/activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.2 \
 || pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/rocm7.2

python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# to run Gate 0
DEVICE=cuda ./run_all.sh

git clone https://github.com/Wan-Video/Wan2.1
grep -viE '^(torch|torchvision|flash_attn)' Wan2.1/requirements.txt > /tmp/wan_req.txt
pip install -r /tmp/wan_req.txt
python -c "import torch; print(torch.version.hip, torch.cuda.is_available())"   # must still show a HIP version and True

pip install -U huggingface_hub
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B Wan2.1_VAE.pth --local-dir ./Wan2.1-T2V-1.3B
# newer huggingface_hub versions name the command `hf download` instead
export PYTHONPATH=$PWD/Wan2.1:$PYTHONPATH

# Cut everything into 81-frame clips at 16 fps and 832×480, which is the regime Wan generates in. 81 frames give 21 latent frames per clip.
mkdir -p clips
for f in raw_videos/*.mp4; do
  b=$(basename "$f" .mp4)
  ffmpeg -loglevel error -i "$f" -an \
    -vf "fps=16,scale=832:480:force_original_aspect_ratio=increase,crop=832:480" \
    -f segment -segment_frames 81,162,243,324 -reset_timestamps 1 "clips/${b}_%03d.mp4"
done
