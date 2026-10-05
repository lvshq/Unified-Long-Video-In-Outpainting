# Local data

The complete InpaintBench dataset is not included in this release. Place your own
videos and masks here; downloaded and generated assets are ignored by Git.

Training expects `DATASET/metadata.csv` with `file_name,text` columns and videos
under `DATASET/train/`. See the main README for preprocessing instructions.

`configs/example_prompts.json` preserves available experiment prompts. It is not
a download manifest or a complete benchmark release.
