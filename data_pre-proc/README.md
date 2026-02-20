# anomaly-detection-dashcam
Real-Time Road Anomaly Detection from Dashcam Footage on Raspberry Pi

# Dataset Preparation Summary – RDD2022

## Final Dataset Choice
**RDD2022** selected as the primary dataset due to:
- Dashcam-style vehicle-mounted images
- Explicit pothole and road damage annotations
- Prior usage with YOLO-based detectors
- Compatibility with edge deployment constraints

---

## Label Strategy
- Original RDD2022 classes:
  - longitudinal crack
  - transverse crack
  - alligator crack
  - pothole
  - other corruption

- Final strategy:
  - ❌ Remove all crack classes
  - ✅ Keep pothole + other corruption
  - ✅ Merge into single class: `road_anomaly` (class 0)

**Rationale:**
Thin cracks are FP-prone in video and unreliable under INT8 quantization.
The task is anomaly detection, not damage taxonomy.

---

## Preprocessing Steps Applied

### 1. Label Cleaning
- Removed crack annotations
- Remapped pothole and corruption to class 0

### 2. Small Box Filtering
- Removed boxes with area < 1% of image
- Reduces false positives and jitter

### 3. Image Resizing
- All images resized to **416×416**
- YOLO-safe due to normalized labels

---

## Data Augmentation (Train Only)
Applied light, realistic augmentations:
- Brightness / contrast jitter
- Motion blur (dashcam simulation)

No geometric or heavy color transformations used.

---

## Dataset Profiling Insights
Profiling performed to understand:
- Anomaly frequency per image
- Typical bounding box size
- Aspect ratio distribution

These insights guide:
- Confidence threshold selection
- Temporal persistence logic
- Precision-focused inference tuning

Total images: 32627
Images with anomalies: 5881
Anomaly ratio: 0.18
Avg box area: 0.0868
Median box area: 0.0505
Avg aspect ratio: 1.82

---

## What the Model Is Expected to Learn
- Detect large, unsafe road surface anomalies
- Ignore thin cracks, shadows, and lane markings
- Produce stable detections in continuous video

---

## Ready for Training
Dataset is YOLO-ready and optimized for:
- Precision over recall
- Real-time edge inference
- Robust deployment on Raspberry Pi
