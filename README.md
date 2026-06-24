# AirLetters CNN+BiLSTM Gesture Recognition

PyTorch project for Qualcomm AirLetters gesture recognition using a CNN frame
encoder plus BiLSTM temporal encoder baseline. The initial workflow trains on a
balanced subset of the official splits, then can be scaled toward the full
paper-style setting.

The official dataset files remain unchanged at the project root:

- `videos/`
- `train.csv`
- `val.csv`
- `test.csv`

## Dataset Summary

- Videos: 161,652 `.mp4` files
- Train split: 128,745 samples
- Validation split: 16,480 samples
- Test split: 16,427 samples
- Classes: 38
- CSV columns: `id`, `filename`, `label`, `worker_id`, `video_duration`

## Project Structure

```text
air_letter_2/
├── checkpoints/
├── configs/
│   └── default.yaml
│   └── cnn_bilstm_subset.yaml
├── docs/
│   └── paper_implementation_notes.md
├── logs/
├── outputs/
├── src/
│   └── airletters/
│       ├── config.py
│       ├── data/
│       │   ├── dataloaders.py
│       │   ├── dataset.py
│       │   └── video.py
│       ├── models/
│       │   └── cnn_bilstm.py
│       ├── pipelines/
│       │   ├── evaluate.py
│       │   ├── infer.py
│       │   └── train.py
│       └── utils/
│           ├── checkpointing.py
│           ├── metrics.py
│           ├── reproducibility.py
│           └── train_eval.py
├── videos/
├── test.csv
├── train.csv
├── val.csv
└── requirements.txt
```

## Folder Purpose

- `configs/`: Dataset paths, model settings, and training parameters.
- `src/airletters/data/`: CSV split loading, OpenCV video decoding, and PyTorch
  dataloaders.
- `src/airletters/models/`: CNN+BiLSTM model implementation.
- `src/airletters/pipelines/`: Train, evaluate, and single-video inference
  commands.
- `src/airletters/utils/`: Checkpointing, metrics, seeding, and training loops.
- `checkpoints/`: Saved `latest.pt` and `best.pt` model checkpoints.
- `logs/`: Training metrics written as `metrics.jsonl`.
- `outputs/`: Reserved for predictions, reports, and later experiment outputs.

## Setup

Run these commands from the project root:

```powershell
cd C:\Users\shelk\Downloads\air_letter_2
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
$env:PYTHONPATH = "$PWD\src"
```

If PyTorch does not detect your NVIDIA GPU, reinstall PyTorch with the CUDA
command recommended by the official PyTorch install selector.

## Paper-Aligned Starting Point

The paper reports the following CNN+LSTM baseline settings for ResNet-50 +
LSTM: 48 frames, 224x224 frame size, ImageNet initialization, random resized
crop for training, center crop for evaluation, label smoothing `0.1`, Adam,
learning rate `1e-3`, and batch size `32`.

For this laptop-friendly first implementation, use:

```text
configs/cnn_bilstm_subset.yaml
```

It keeps the same baseline direction but starts smaller:

- ResNet-50 + BiLSTM
- 16 frames
- 112x112 crops
- balanced subset: 25 train videos/class, 5 val videos/class, 5 test videos/class
- Adam, LR `1e-3`, label smoothing `0.1`
- batch size `2`

More notes from the paper are in `docs/paper_implementation_notes.md`.

## Train on Subset

Start with a tiny smoke run to confirm everything works:

```powershell
python -m airletters.pipelines.train --config configs/cnn_bilstm_subset.yaml --epochs 1 --max-train-batches 2 --max-val-batches 2
```

Then train the subset:

```powershell
python -m airletters.pipelines.train --config configs/cnn_bilstm_subset.yaml
```

Training saves:

- `checkpoints/latest.pt`
- `checkpoints/best.pt`
- `logs/metrics.jsonl`

## Evaluate

```powershell
python -m airletters.pipelines.evaluate --config configs/cnn_bilstm_subset.yaml --checkpoint checkpoints\best.pt --split test --save-plots
```

Evaluation with `--save-plots` writes:

- `outputs/test/test_metrics.json`
- `outputs/test/test_confusion_matrix.png`
- `outputs/test/test_per_class_accuracy.png`

## Inference

```powershell
python -m airletters.pipelines.infer --config configs/cnn_bilstm_subset.yaml --checkpoint checkpoints\best.pt --video videos\00000000.mp4
```

## Scaling Up

After the subset run is stable, increase these in `configs/cnn_bilstm_subset.yaml`:

- `data.subset.train.samples_per_class`
- `data.video.num_frames`, moving toward `32` or `48`
- `data.video.image_size`, moving toward `224`
- `data.video.resize_short_edge`, moving toward `300`

The full paper setting is heavy for a 6 GB GPU, so increase one thing at a time.
