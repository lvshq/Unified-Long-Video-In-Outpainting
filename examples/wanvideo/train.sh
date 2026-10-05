#!/usr/bin/env bash
# Run from the repository root after installing requirements-train.txt.
# Usage: bash examples/wanvideo/train.sh data/example_dataset models/WanVideo/Wan2.1-T2V-14B outputs/training
set -euo pipefail
DATASET_DIR="${1:?Provide the dataset directory}"
MODEL_DIR="${2:?Provide the Wan2.1-T2V-14B directory}"
OUTPUT_DIR="${3:-outputs/training}"

python examples/wanvideo/train_wan_t2v.py \
  --task data_process --dataset_path "$DATASET_DIR" --output_path "$OUTPUT_DIR" \
  --text_encoder_path "$MODEL_DIR/models_t5_umt5-xxl-enc-bf16.pth" \
  --vae_path "$MODEL_DIR/Wan2.1_VAE.pth" \
  --num_frames 81 --height 240 --width 416 --tiled

DIT_PATHS=$(python - "$MODEL_DIR" <<'PYTHON'
import pathlib, sys
files = sorted(pathlib.Path(sys.argv[1]).glob("diffusion_pytorch_model-*-of-*.safetensors"))
if len(files) != 6:
    raise SystemExit("Expected six Wan2.1-T2V-14B DiT shards")
print(",".join(str(path) for path in files))
PYTHON
)
python examples/wanvideo/train_wan_t2v.py \
  --task train --train_architecture lora \
  --dataset_path "$DATASET_DIR" --output_path "$OUTPUT_DIR" --dit_path "$DIT_PATHS" \
  --steps_per_epoch 500 --max_epochs 4 --num_frames 81 --height 240 --width 416 \
  --learning_rate 1e-4 --lora_rank 16 --lora_alpha 16 --masked_loss_weight 0.9 \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" --accumulate_grad_batches 1 \
  --use_gradient_checkpointing --seed 1
