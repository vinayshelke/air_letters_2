"""MediaPipe hand-landmark extractor for AirLetters videos (Tasks API).

Compatible with mediapipe >= 0.10 which uses the new Tasks API
(``mediapipe.tasks``) instead of the removed ``mp.solutions.hands`` API.

Extracts 21 hand landmarks (x, y, z) from each frame of a decoded video.
Only the dominant (most-confident) hand is kept per frame; frames where no
hand is detected receive a zero vector so the sequence length stays fixed.

Output shape: ``(num_frames, 63)``  — 21 landmarks × 3 coordinates.

The hand landmarker model file (``hand_landmarker.task``) is downloaded
automatically on the first import if it is not already present in
``TASK_MODEL_PATH``.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "mediapipe >= 0.10 is required for landmark extraction. "
        "Install it with: pip install mediapipe"
    ) from exc

# ---------------------------------------------------------------------------
# Model file — downloaded once and cached locally
# ---------------------------------------------------------------------------
TASK_MODEL_URL: str = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/"
    "float16/1/hand_landmarker.task"
)
TASK_MODEL_PATH: Path = Path(__file__).parent / "hand_landmarker.task"

LANDMARK_DIM: int = 63  # 21 landmarks × (x, y, z)


def _ensure_model() -> str:
    """Download the hand landmarker task model if it is not already present."""
    if not TASK_MODEL_PATH.exists():
        print(
            f"[landmarks] Downloading hand landmarker model to {TASK_MODEL_PATH} …",
            flush=True,
        )
        urllib.request.urlretrieve(TASK_MODEL_URL, TASK_MODEL_PATH)
        print("[landmarks] Download complete.", flush=True)
    return str(TASK_MODEL_PATH)


def extract_landmarks_from_frames(
    frames_rgb: np.ndarray,
    min_hand_detection_confidence: float = 0.5,
    min_hand_presence_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> np.ndarray:
    """Extract hand landmarks from a sequence of RGB frames.

    Uses the MediaPipe Tasks API (mediapipe >= 0.10).

    Args:
        frames_rgb: ``uint8`` array of shape ``(T, H, W, 3)`` in RGB order.
        min_hand_detection_confidence: Minimum confidence for hand detection.
        min_hand_presence_confidence: Minimum confidence for hand presence.
        min_tracking_confidence: Minimum confidence for landmark tracking.

    Returns:
        Float32 array of shape ``(T, 63)``. Frames with no detected hand
        contain all zeros.
    """
    model_path = _ensure_model()
    num_frames = frames_rgb.shape[0]
    result_array = np.zeros((num_frames, LANDMARK_DIM), dtype=np.float32)

    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,  # frame-by-frame (no timestamps)
        num_hands=1,
        min_hand_detection_confidence=min_hand_detection_confidence,
        min_hand_presence_confidence=min_hand_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    with mp_vision.HandLandmarker.create_from_options(options) as detector:
        for t, frame in enumerate(frames_rgb):
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            detection_result = detector.detect(mp_image)

            if not detection_result.hand_landmarks:
                continue  # leave zero vector for this frame

            # When multiple hands detected (num_hands > 1), pick the most
            # confident one using the handedness score.
            if (
                len(detection_result.hand_landmarks) > 1
                and detection_result.handedness
            ):
                scores = [
                    detection_result.handedness[i][0].score
                    for i in range(len(detection_result.hand_landmarks))
                ]
                best_idx = int(np.argmax(scores))
            else:
                best_idx = 0

            landmarks = detection_result.hand_landmarks[best_idx]
            for lm_idx, lm in enumerate(landmarks):
                base = lm_idx * 3
                result_array[t, base] = lm.x
                result_array[t, base + 1] = lm.y
                result_array[t, base + 2] = lm.z

    return result_array
