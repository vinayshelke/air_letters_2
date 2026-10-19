# AirLetters – Real-Time Air-Writing & Gesture Recognition

A deep learning pipeline for recognizing hand-written letters and digits drawn
in the air, built on the
[Qualcomm AirLetters dataset](https://github.com/quic/aimet-model-zoo).  
The final model achieves **86.4 % top-1 validation accuracy** across 28 classes
(A–Z + *Doing Nothing* + *Doing Other Things*) using a ResNet-18 frame encoder
and a 2-layer Bidirectional LSTM.

---

## Results

| Branch | Architecture | Frames | Train videos | Classes | Val Accuracy |
|---|---|---|---|---|---|
| `mediapipe+t` | MediaPipe landmarks + Transformer | 48 | 13 000 (500 / class) | 26 | ~65 % |
| `v1` *(this branch)* | ResNet-18 + BiLSTM | 48 | **15 680** (400 / class) | **28** | **86.4 %** |

---

## Architecture

### Branch: `v1` — ResNet-18 + BiLSTM (Final Model)

```
Input video
    │
    ▼
Motion-based frame sampling (48 frames)
    │
    ▼
ResNet-18 (ImageNet pre-trained)   ← per-frame spatial encoder
    │   [batch, 48, 512]
    ▼
2-layer Bidirectional LSTM
    │   [batch, 48, 512]
    ▼
Mean Pooling across time steps
    │   [batch, 512]
    ▼
Linear classifier  →  28 classes
```

**Key design choices:**
- **Motion-based temporal sampling** – selects the 48 most action-dense frames
  per video via frame-differencing, discarding idle background frames.
- **Mean pooling** – aggregates BiLSTM hidden states over the time axis,
  making the classifier robust to variable gesture speeds.
- **Two special classes** – *Doing Nothing* and *Doing Other Things* act as
  natural gesture delimiters in the live demo for word-level spelling.

---

### Branch: `mediapipe+t` — MediaPipe + Transformer (Baseline)

```
Input video
    │
    ▼
MediaPipe Hands (per-frame landmark extraction)
    │   21 keypoints × (x, y, z) + joint angles + velocity
    ▼
Transformer sequence encoder (48 frames of landmarks)
    │
    ▼
Linear classifier  →  26 classes (A–Z)
```

- Lightweight: processes hand landmarks instead of raw pixels — no GPU needed
  for feature extraction.
- Live demo on this branch used a **2.5-second recording timer** per letter
  (press key → record → predict).
- Reached ~65 % validation accuracy on 26 letter classes (500 videos / class).

---

## Dataset

[Qualcomm AirLetters](https://github.com/quic/aimet-model-zoo) – 161 652 MP4
clips from crowd-sourced workers.

| Split | Total clips | Used in v1 (subset) |
|---|---|---|
| Train | 128 745 | 11 200 (400 / class × 28) |
| Validation | 16 480 | 2 240 (80 / class × 28) |
| Test | 16 427 | 2 240 (80 / class × 28) |

**Classes (28):** A–Z + *Doing Nothing* + *Doing Other Things*

Place the raw dataset at the project root:

```
air_letter_2/
├── videos/          ← all MP4 clips
├── train.csv
├── val.csv
└── test.csv
```

---

## Project Structure

```
air_letter_2/
├── configs/
│   ├── letters.yaml          ← 28-class config (default)
│   └── digits.yaml           ← 12-class config (0–9 + specials)
├── src/airletters/
│   ├── config.py             ← YAML loading helpers
│   ├── data/
│   │   ├── dataset.py        ← CSV loading, class filtering, subset sampling
│   │   ├── dataloaders.py    ← PyTorch DataLoader factory
│   │   └── video.py          ← OpenCV decoding, motion sampling, augmentation
│   ├── models/
│   │   └── cnn_bilstm.py     ← ResNet-18 + BiLSTM model definition
│   ├── pipelines/
│   │   ├── train.py          ← Training loop (supports --resume)
│   │   ├── evaluate.py       ← Validation / test evaluation + plots
│   │   ├── infer.py          ← Single-video top-k inference
│   │   └── live_demo.py      ← Real-time webcam word-spelling demo
│   └── utils/
│       ├── checkpointing.py  ← save / load (model + optimizer + scheduler)
│       ├── plots.py          ← Training curves, confusion matrix, per-class bar
│       ├── reproducibility.py
│       └── train_eval.py     ← train_one_epoch / evaluate functions
├── checkpoints/              ← best.pt  latest.pt (git-ignored)
├── logs/                     ← metrics.jsonl (git-ignored)
├── outputs/                  ← PNG plots (git-ignored)
└── requirements.txt
```

---

## Git Sync & Setup

### On Current PC (Pushing Changes)
```powershell
git add .
git commit -m "Configure 38-class full dataset training, evaluation plots, and cache optimizations"
git push origin resnet38class
```

### On Other PC (Fetching & Checking Out This Branch)
If the repository already exists on the other PC but does not have the `resnet38class` branch yet:
```powershell
# 1. Fetch all remote branches from GitHub
git fetch origin

# 2. Switch to the new branch (creates local tracking branch automatically)
git checkout -b resnet38class origin/resnet38class
# (Or simply: git switch resnet38class)

# 3. Verify you are on the right branch
git branch
```

---

## Quickstart & Commands

Before running any script, make sure `src` is in your Python path:
```powershell
$env:PYTHONPATH="src"
```

### 1. Frame Caching (Pre-Extraction)
Pre-extracting motion-sampled frames to JPEG eliminates CPU video decompression bottlenecks during training:
```powershell
# Cache train, val, and test splits
$env:PYTHONPATH="src"; python scripts/cache_frames.py --config configs/letters.yaml

# Cache specific splits or override number of temporal variants
$env:PYTHONPATH="src"; python scripts/cache_frames.py --config configs/letters.yaml --splits train val --num-variants 1
```

### 2. Training
```powershell
# Train from scratch using config
$env:PYTHONPATH="src"; python src/airletters/pipelines/train.py --config configs/letters.yaml

# Resume training from checkpoint
$env:PYTHONPATH="src"; python src/airletters/pipelines/train.py --config configs/letters.yaml --resume checkpoints/latest.pt
```

### 3. Evaluation & Metrics
```powershell
# Evaluate model on test set (or val set) and generate metrics/plots
$env:PYTHONPATH="src"; python src/airletters/pipelines/evaluate.py --config configs/letters.yaml --checkpoint checkpoints/best.pt --split test
```

### 4. Single-Video Inference
```powershell
# Predict air gesture for an individual video file
$env:PYTHONPATH="src"; python src/airletters/pipelines/infer.py --config configs/letters.yaml --checkpoint checkpoints/best.pt --video path/to/sample.mp4 --top-k 3
```

### 5. Managing Frame Cache Size (Windows PowerShell)
```powershell
# Check current size of frames_cache (in GB)
(Get-ChildItem -Path frames_cache -Recurse | Measure-Object -Property Length -Sum).Sum / 1GB

# Check number of cached videos
(Get-ChildItem -Path frames_cache -Directory).Count

# Fast deletion of cache using robocopy mirror trick (avoids slow Windows MFT file-by-file deletion):
mkdir C:\empty_temp
robocopy C:\empty_temp frames_cache /MIR /MT:8
rmdir frames_cache
rmdir C:\empty_temp
```

---

## Live Demo — Letter Recognition

Run the real-time webcam demo to recognize air-written letters.

```powershell
python src/airletters/pipelines/live_demo.py --checkpoint checkpoints/best.pt
```

**How it works:**
1. Press any key to start a **2.5-second recording window**.
2. Draw a letter in the air during the window.
3. The model predicts the letter and displays it on screen with a confidence score.
4. Repeat for the next letter.

| Key | Action |
|---|---|
| `Q` | Quit |

---

## Configuration Reference

Both config files are fully commented. Key parameters:

| Parameter | Description |
|---|---|
| `class_filter` | `digits_with_special` / `letters_with_special` / `all` |
| `num_frames` | Frames sampled per video (48) |
| `sampling_strategy` | `motion` (recommended) or `uniform` |
| `samples_per_class` | Subset size per class for train / val / test |
| `scheduler` | `cosine` (CosineAnnealingLR) or `none` |
| `batch_size` | Reduce if CUDA out-of-memory |

---

## Requirements

```
torch
torchvision
opencv-python
numpy
pandas
PyYAML
tqdm
scikit-learn
matplotlib
```
