# AirLetters Paper Notes for Implementation

Paper: `AirLetters: An Open Video Dataset of Characters Drawn in the Air`

## Dataset

- Total videos: 161,652
- Classes: 38
- Train split: 128,745 videos from 958 workers
- Validation split: 16,480 videos from 412 workers
- Test split: 16,427 videos from 411 workers
- Labels: 26 letters, 10 digits, `Doing Nothing`, and `Doing Other Things`

The official split is worker-disjoint. Do not edit `train.csv`, `val.csv`, or
`test.csv`.

## Paper Preprocessing

- Videos are resampled to 30 FPS.
- Videos are resized with aspect ratio preserved.
- Training uses random resized crop with scale `(0.7, 1.0)`.
- Evaluation/testing uses center crop.
- The paper studies 1, 8, 16, 24, 32, and 48 frames.
- Accuracy improves strongly as frame count increases, with 48 frames performing
  best among the tested frame counts.

## CNN + LSTM Results

From Table 3 and Appendix Table 5:

| Model | Frames | Frame Size | Optimizer | LR | Batch Size | Params | Top-1 |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| ResNet-101 + LSTM | 48 | 224x224 | Adam | 1e-4 | 8 | 43.72M | 58.45 |
| ResNet-50 + LSTM | 48 | 224x224 | Adam | 1e-3 | 32 | 24.73M | 63.24 |

Both use FP32, ImageNet-1k initialization with RA1 recipe, random resized crop
for training, center crop for evaluation, label smoothing `0.1`, and no gradient
clipping.

## Initial Project Choice

The first runnable baseline uses `configs/cnn_bilstm_subset.yaml`:

- ResNet-50 + BiLSTM
- 16 frames
- 112x112 crops
- balanced subset sampling per class
- Adam, LR `1e-3`, label smoothing `0.1`
- batch size `2`, `num_workers: 0` for Windows reliability

This is intentionally smaller than the paper setting so training can begin on a
6 GB laptop GPU. After the subset run works, scale toward the paper setting by
raising `num_frames` to `32` or `48`, `image_size` to `224`, and increasing the
subset size.

