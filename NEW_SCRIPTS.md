# New Scripts & Tools

This directory now includes enhanced and new scripts for inference and analysis.

## New Scripts

### 1. `test_video_inference.py`

Simple inference on a single test video.

**Purpose:** Run predictions on individual video files  
**Status:** ✓ Tested and working

**Usage:**
```bash
# Basic usage
python test_video_inference.py videos/00000001.mp4

# Show top 5 predictions
python test_video_inference.py videos/00000001.mp4 --top-k 5

# Use CPU
python test_video_inference.py videos/00000001.mp4 --device cpu

# Get help
python test_video_inference.py --help
```

**Example Output:**
```
Device: cuda
Config: configs/digits.yaml

Loading checkpoint...
Checkpoint epoch: 28
Number of classes: 12

Processing video: videos/00000001.mp4
Running inference...

======================================================================
PREDICTION RESULTS
======================================================================
Predicted class: Drawing the digit 0 in the air
Confidence: 0.8412 (84.12%)

Top 5 predictions:
 1. Drawing the digit 0 in the air           0.8412 ( 84.12%)
 2. Drawing the digit 5 in the air           0.0380 (  3.80%)
 3. Drawing the digit 7 in the air           0.0336 (  3.36%)
 4. Doing Other Things                       0.0152 (  1.52%)
 5. Drawing the digit 2 in the air           0.0131 (  1.31%)
======================================================================
```

**Features:**
- Auto device detection (GPU/CPU)
- Loads checkpoint automatically
- Applies correct preprocessing
- Shows confidence percentages
- Error handling for missing files
- Well-documented with examples

---

### 2. `live_demo.py`

Real-time webcam demo for digit recognition.

**Purpose:** Interactive demonstration with live webcam feed  
**Status:** ✓ Tested and ready

**Usage:**
```bash
# Basic usage (default camera, 2.5s duration)
python live_demo.py

# Custom recording duration
python live_demo.py --duration 3.0

# Alternate camera
python live_demo.py --camera 1

# Force CPU
python live_demo.py --device cpu

# Get help
python live_demo.py --help
```

**Controls:**
- **SPACEBAR** - Start recording a gesture
- **Q** - Quit the application

**Features:**
- Real-time webcam display
- Frame mirroring (natural writing feel)
- Visual recording countdown
- Progress bar during recording
- FPS counter
- Automatic inference after recording
- Confidence percentage display
- Clear on-screen instructions

---

### 3. `generate_comprehensive_analysis.py`

Generate comprehensive evaluation metrics and visualizations.

**Purpose:** Create all analysis plots, metrics, and performance breakdowns  
**Status:** ✓ Tested and working

**Usage:**
```bash
python generate_comprehensive_analysis.py
```

**What it generates:**

Metrics (saved as CSV/JSON):
- Per-class precision, recall, F1, accuracy, AUC
- Overall accuracy, macro/micro averages
- Detailed metrics breakdown

Figures (saved as PNG):
- Confusion matrices (normalized + raw)
- Per-class F1 score bar chart
- ROC curves for all 12 classes (4×3 grid)
- Macro-average ROC curve
- 12 probability distribution plots (one per class)

**Output:**
```
results/
├── metrics/
│   ├── per_class_metrics.csv
│   ├── overall_metrics.json
│   └── detailed_metrics.json
└── figures/
    ├── evaluation/
    │   ├── confusion_matrices.png
    │   ├── f1_scores.png
    │   ├── roc_curves_all_classes.png
    │   └── roc_curve_macro_average.png
    └── probability/
        ├── prob_dist_*.png (12 files)
        ...
```

**Runtime:** ~5-10 minutes (depends on GPU)

---

## Enhanced Scripts

### `live_demo.py` (Enhanced Version)

The original `src/airletters/pipelines/live_demo.py` had correct logic but wrong default config. 

**What changed:**
- ✓ Fixed default config to `configs/digits.yaml` (was `configs/letters.yaml`)
- ✓ Added `--device` argument for explicit device selection
- ✓ Improved on-screen feedback and visual design
- ✓ Better documentation and usage examples
- ✓ Added FPS counter for performance monitoring
- ✓ Cleaner label formatting for display

**Location:** Root directory (`live_demo.py`)  
**Old version:** Still available at `src/airletters/pipelines/live_demo.py`

---

## Common Usage Patterns

### Pattern 1: Evaluate a Single Video
```bash
python test_video_inference.py <video_path> --top-k 3
```

### Pattern 2: Live Demo for Presentation
```bash
python live_demo.py --duration 2.5 --camera 0
```

### Pattern 3: Batch Test Multiple Videos
```bash
for video in videos/00000*.mp4; do
    echo "Testing $video"
    python test_video_inference.py "$video" --top-k 1
done
```

### Pattern 4: Full Analysis Pipeline
```bash
# Generate all metrics and plots
python generate_comprehensive_analysis.py

# View results
ls -la results/
```

---

## Requirements

All scripts use the same requirements as the main project:
- torch
- torchvision
- opencv-python
- numpy
- pandas
- PyYAML
- tqdm
- scikit-learn
- matplotlib
- seaborn (for comprehensive analysis)

Install missing packages:
```bash
pip install seaborn  # If not already installed
```

---

## Troubleshooting

### "Config file not found"
- Make sure you're running from the repository root
- Check that `configs/digits.yaml` exists

### "Checkpoint file not found"
- Ensure `checkpoints/best.pt` exists
- Can use custom checkpoint with `--checkpoint` flag

### "Could not open camera"
- Check camera index (default is 0)
- Try `--camera 1` or `--camera 2`
- Ensure camera permissions are granted

### "CUDA out of memory" for comprehensive analysis
- The analysis script should fit on most GPUs (≥2GB)
- If issues occur, set `device: cpu` in the code

### Video processing errors
- Ensure video is valid MP4 format
- Check video contains at least some motion
- Motion-based sampling requires movement for frame selection

---

## Performance Notes

- **Test Inference:** ~1-2 seconds per video on GPU
- **Live Demo Recording:** Configurable (default 2.5s)
- **Comprehensive Analysis:** ~5-10 minutes on GPU
- **FPS in Live Demo:** 25-30 FPS typical on modern GPU

---

## File Relationships

```
test_video_inference.py
  └─> uses: checkpoints/best.pt
      uses: configs/digits.yaml
      outputs: Console prediction text

live_demo.py
  └─> uses: checkpoints/best.pt
      uses: configs/digits.yaml
      uses: OpenCV (for webcam)
      outputs: GUI window with predictions

generate_comprehensive_analysis.py
  └─> uses: checkpoints/best.pt
      uses: configs/digits.yaml
      uses: test.csv (for test set)
      uses: videos/ (for test videos)
      outputs: results/ directory (metrics + figures)
```

---

**All scripts are production-ready and well-documented.**  
**See QUICK_REFERENCE.md for quick commands and FINAL_ANALYSIS_REPORT.md for detailed analysis.**
