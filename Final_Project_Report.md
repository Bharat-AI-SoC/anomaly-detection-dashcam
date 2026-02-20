<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;500&display=swap');

body {
    font-family: 'Georgia', 'Times New Roman', serif;
    font-size: 12px;
    line-height: 1.6;
    color: #1a1a1a;
    text-align: justify;
}

h1 {
    font-family: 'Calibri', 'Segoe UI', sans-serif;
    font-weight: bold;
    font-size: 18px;
    text-align: center;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}

h2 {
    font-family: 'Calibri', 'Segoe UI', sans-serif;
    font-weight: bold;
    font-size: 16px;
    text-align: center;
    margin-top: 20px;
    margin-bottom: 10px;
}

h3 {
    font-family: 'Calibri', 'Segoe UI', sans-serif;
    font-weight: 600;
    font-size: 14px;
    text-align: left;
    margin-top: 16px;
    margin-bottom: 8px;
}

h4 {
    font-family: 'Calibri', 'Segoe UI', sans-serif;
    font-weight: 600;
    font-size: 13px;
    text-align: left;
    margin-top: 12px;
    margin-bottom: 6px;
}

p {
    font-family: 'Georgia', 'Times New Roman', serif;
    font-size: 12px;
    line-height: 1.6;
    margin-bottom: 8px;
    text-align: justify;
}

li {
    font-family: 'Georgia', 'Times New Roman', serif;
    font-size: 12px;
    line-height: 1.5;
    text-align: left;
}

table {
    margin-left: auto;
    margin-right: auto;
    font-size: 11px;
}

td, th {
    font-family: 'Georgia', 'Times New Roman', serif;
    font-size: 11px;
    text-align: center;
    padding: 4px 8px;
}

code, pre {
    font-family: 'Consolas', 'Roboto Mono', monospace;
    font-size: 10px;
    text-align: left;
}

blockquote {
    font-family: 'Georgia', 'Times New Roman', serif;
    font-style: italic;
    text-align: justify;
    margin: 10px 30px;
}
</style>

<!-- ===================== COVER PAGE ===================== -->

<div style="text-align: center; padding-top: 60px; break-after: always;">

<h1>Project Report</h1>

<h2>on</h2>

<p style="font-family: Calibri, sans-serif; font-size: 15px; font-style: italic; margin-top: 18px; text-align: center;">
Real-Time Road Anomaly Detection from Dashcam Footage on Raspberry Pi
</p>

<h3 style="text-align: center; margin-top: 30px;">Submitted by</h3>

<p style="line-height: 1.9; text-align: center;">
Rajnish Ranjan – 21f3001109<br>
Soumya Oruganti – 21f2000969<br>
Vignesh Reddy – 24f2007195
</p>

<h3 style="text-align: center; margin-top: 28px;">Mentor</h3>

<p style="text-align: center;">
Anamika Chhabra<br>
anamika@study.iitm.ac.in
</p>

<h3 style="text-align: center; margin-top: 28px;">February, 2026</h3>

<p style="font-size: 11px; margin-top: 18px; text-align: center;">
Framework: Ultralytics YOLOv5 + ONNX Runtime<br>
Target Platform: Raspberry Pi (Edge AI)<br>
License: AGPL-3.0
</p>

<hr style="margin-top: 25px; width: 60%;">

<p style="font-family: Calibri, sans-serif; font-weight: bold; font-size: 12px; letter-spacing: 0.5px; margin-top: 18px; text-align: center;">
DEPARTMENT OF DATA SCIENCE AND APPLICATION<br>
INDIAN INSTITUTE OF TECHNOLOGY MADRAS<br>
BS DEGREE PROGRAMME<br>
CHENNAI, TAMIL NADU, INDIA
</p>

</div>
<!-- ===================== TABLE OF CONTENTS ===================== -->

<div style="page-break-after: always;">

## Table of Contents

1. [Abstract](#1-abstract)
2. [Introduction](#2-introduction)
3. [Problem Statement](#3-problem-statement)
4. [Methodology](#4-methodology)
   - 4.1 [System Architecture Overview](#41-system-architecture-overview)
   - 4.2 [Model Selection — YOLOv5n](#42-model-selection--yolov5n)
   - 4.3 [Dataset and Preprocessing](#43-dataset-and-preprocessing)
   - 4.4 [Training Pipeline](#44-training-pipeline)
   - 4.5 [Loss Functions](#45-loss-functions)
   - 4.6 [Data Augmentation](#46-data-augmentation)
   - 4.7 [OpenCV Video Pipeline Integration](#47-opencv-video-pipeline-integration)
   - 4.8 [Anomaly Logging and Alert System](#48-anomaly-logging-and-alert-system)
5. [Hardware Utilization](#5-hardware-utilization)
   - 5.1 [Raspberry Pi — Edge Deployment Platform](#51-raspberry-pi--edge-deployment-platform)
   - 5.2 [GPU-Accelerated Training (Host Machine)](#52-gpu-accelerated-training-host-machine)
   - 5.3 [Mixed Precision Training (AMP)](#53-mixed-precision-training-amp)
   - 5.4 [Memory and Compute Constraints on Raspberry Pi](#54-memory-and-compute-constraints-on-raspberry-pi)
6. [Optimization Techniques](#6-optimization-techniques)
   - 6.1 [INT8 Quantization (ONNX)](#61-int8-quantization-onnx)
   - 6.2 [Edge-Optimized Model Export](#62-edge-optimized-model-export)
   - 6.3 [ONNX Runtime Inference on ARM](#63-onnx-runtime-inference-on-arm)
   - 6.4 [Input Resolution and Batch Optimization](#64-input-resolution-and-batch-optimization)
   - 6.5 [AutoAnchor Optimization](#65-autoanchor-optimization)
   - 6.6 [Learning Rate Scheduling](#66-learning-rate-scheduling)
   - 6.7 [Exponential Moving Average (EMA)](#67-exponential-moving-average-ema)
   - 6.8 [Layer Freezing and Transfer Learning](#68-layer-freezing-and-transfer-learning)
7. [Results](#7-results)
   - 7.1 [Evaluation Metrics](#71-evaluation-metrics)
   - 7.2 [Model Size and Quantization Benchmark](#72-model-size-and-quantization-benchmark)
   - 7.3 [Raspberry Pi Inference Performance](#73-raspberry-pi-inference-performance)
   - 7.4 [Detection Accuracy on Road Anomalies](#74-detection-accuracy-on-road-anomalies)
8. [Conclusion](#8-conclusion)
9. [Future Work](#9-future-work)
10. [References](#10-references)

</div>

<!-- ===================== ABSTRACT ===================== -->

<div style="page-break-after: always;">

<a id="1-abstract"></a>

## Abstract

The idea behind this project was pretty simple — we wanted to see if a cheap, small computer like the Raspberry Pi could actually spot road hazards from a dashcam feed in real time. Potholes, random debris on the road, fallen objects — these things cause real damage to vehicles and sometimes even accidents, and most of the time no one reports them until it is too late.

So what we did is this: we picked YOLOv5n, which is one of the lighter object detection models out there, and trained it specifically on images of potholes and road obstacles. After training, we did not just leave it as a big floating-point model. We converted it into an INT8-quantized ONNX format, which basically shrinks the model down to about 4 MB from the original 14 MB, and that makes a huge difference when you are running things on a Raspberry Pi with only 4 GB of RAM and no GPU to speak of.

On the software side, we built a pipeline using OpenCV that grabs frames from the dashcam, feeds them through the ONNX model, and then checks if any road anomaly was found. If something is detected — say a pothole with confidence above our threshold — the system logs the event with a timestamp, draws the bounding box on the frame, and saves a short video clip of that moment. The whole thing runs at about 5 to 6 frames per second on the Pi at 320 by 320 resolution, which honestly is good enough for typical city driving speeds where you would encounter these hazards.

The key things we focused on throughout were making sure the model stays accurate even after aggressive quantization, keeping the memory footprint low enough for the Pi, and building a logging system that actually gives useful data — timestamps, coordinates, saved clips — so that someone could review or even map out problem spots on a road later.

We found that the quantization only costs about 1 to 2 percent in detection accuracy, which felt like a very fair tradeoff for getting nearly 4 times the speed and a much smaller model. Overall, it worked out better than we initially expected.

</div>

<!-- ===================== INTRODUCTION ===================== -->

<div style="page-break-after: always;">

<a id="2-introduction"></a>

## 2. Introduction

Anyone who has driven on Indian roads — or really any developing country's roads — knows how common potholes and random road debris can be. These are not just minor inconveniences. They damage suspensions, blow out tyres, and in the worst case cause serious accidents, especially at night or in poor weather when visibility is low. Municipal bodies generally rely on manual surveys or complaint-based systems to track and fix these issues, which is slow and unreliable.

The thought process behind this project was: what if we could attach a small, cheap AI system to a dashcam that automatically spots these hazards and logs them? The Raspberry Pi seemed like the obvious hardware choice — it costs about 35 to 75 dollars, runs on 5 watts of power from a car USB port, and has enough computing capability to run lightweight neural networks if we optimise things properly.

For the detection model, we went with YOLOv5n by Ultralytics. We considered a few other options like MobileNet-SSD and YOLOv5n, but YOLOv5n gave us the best tradeoff between detection quality and size. The nano version was faster but missed too many detections, and MobileNet-SSD's accuracy was quite low for our use case. YOLOv5n, once quantized to INT8, shrinks down to about 4 MB which fits nicely within the Pi's constraints.

The whole project can be thought of in two phases. First, we train the model on a separate machine with a proper GPU — that is the computationally expensive part. Second, we take the trained and compressed model, load it onto the Raspberry Pi, and run real-time inference using ONNX Runtime and OpenCV. The Pi captures frames from a connected dashcam, runs each frame through the model, and if it detects a pothole or obstacle, it logs the event and saves a clip.

### Project Objectives

- Build an edge AI application on the Raspberry Pi that processes dashcam footage in real time.
- Train a YOLOv5n model to detect road anomalies (potholes, obstacles) from dashcam images.
- Convert the trained model using INT8 quantization through ONNX Runtime for edge deployment.
- Hook up the model with an OpenCV video pipeline for continuous frame capture and processing.
- Implement a logging system with timestamps and automatic video clip saving when anomalies are detected.
- Optimise everything so it actually runs usably on the Pi's limited hardware — 4 GB RAM, ARM CPU, no GPU.

</div>

<!-- ===================== PROBLEM STATEMENT ===================== -->

<div style="page-break-after: always;">

<a id="3-problem-statement"></a>

## 3. Problem Statement

> Real-Time Road Anomaly Detection from Dashcam Footage on Raspberry Pi — Build an edge AI application on Raspberry Pi that processes dashcam footage in real-time to detect and log road anomalies such as potholes and unexpected obstacles.

The core task is to pick a lightweight object detector, get it onto an edge device in a compressed format, and integrate it with a video pipeline that can capture, process, and act on each frame. When something is detected, the system needs to produce useful output — timestamped logs, saved video clips — not just flash a bounding box on screen.

### Key Deliverables

| # | Deliverable | Status |
|---|---|---|
| 1 | Lightweight detector selection and training (YOLOv5n) | Completed |
| 2 | Edge-optimized model conversion (INT8 ONNX) | Completed |
| 3 | OpenCV video pipeline integration | Completed |
| 4 | Timestamped anomaly logging and clip saving | Completed |
| 5 | Deployment and testing on Raspberry Pi | Completed |

</div>

<!-- ===================== METHODOLOGY ===================== -->

<a id="4-methodology"></a>

## 4. Methodology

<a id="41-system-architecture-overview"></a>

### 4.1 System Architecture Overview

The system works in two distinct phases. We will walk through both.

In the first phase, training happens on a host machine that has a proper GPU. We feed our annotated road anomaly images into YOLOv5n, train until the metrics look good, export the best weights, and then quantize the model to INT8 ONNX format.

Phase 1 — Training (Host Machine with GPU)

```
Dataset (Road Anomaly Images)
    ↓
YOLOv5n Training (GPU, AMP, Transfer Learning)
    ↓
Best Weights (best.pt)
    ↓
ONNX Export → INT8 Post-Training Quantization
    ↓
best_int8.onnx (Edge-optimized model)
```

In the second phase, everything runs on the Raspberry Pi. The dashcam feeds frames into OpenCV, each frame gets preprocessed and run through the quantized ONNX model, and if the model detects something, the system logs it and saves a short clip.

Phase 2 — Inference (Raspberry Pi)

```
Dashcam / Pi Camera (Video Stream)
    ↓
OpenCV VideoCapture (frame-by-frame)
    ↓
Preprocessing (Letterbox resize, normalize)
    ↓
ONNX Runtime Inference (INT8 model)
    ↓
Post-processing (NMS, confidence filter)
    ↓
┌──────────────────────────┐
│ Anomaly Detected?        │
│  YES → Log timestamp     │
│       → Save video clip  │
│       → Draw bounding box│
│  NO  → Continue          │
└──────────────────────────┘
    ↓
Annotated display / headless logging
```

---

<a id="42-model-selection--yolov5n"></a>

### 4.2 Model Selection — YOLOv5n

We spent a fair bit of time evaluating which model to go with. The table below captures why we chose YOLOv5n over the alternatives:

| Criterion | YOLOv5n | MobileNet-SSD | YOLOv5n |
|---|---|---|---|
| Model Size (FP32) | ~14 MB | ~23 MB | ~3.9 MB |
| mAP@0.5 (COCO) | 56.8% | ~21% | 45.7% |
| Speed (CPU) | Moderate | Fast | Fast |
| Edge Quantization | Excellent (ONNX/TFLite) | Good (TFLite) | Excellent |
| Accuracy-Size Tradeoff | Best | Low accuracy | Lower accuracy |

MobileNet-SSD was fast but its detection accuracy was really not up to the mark for our needs. YOLOv5n was tempting because of its tiny size, but it missed too many potholes in our initial experiments. YOLOv5n ended up being the sweet spot — after INT8 quantization it comes down to about 4 MB anyway, so the original size difference does not matter much.

The model architecture, defined in models/yolov5n.yaml, uses these parameters:

| Parameter | Value |
|---|---|
| Number of Classes (nc) | 2 (pothole, obstacle) |
| Depth Multiple | 0.33 |
| Width Multiple | 0.50 |
| Input Resolution | 416 × 416 (training) / 320 × 320 (edge inference) |

#### Backbone (CSPDarknet53-based)

The backbone is made up of Conv layers, C3 blocks (Cross Stage Partial with 3 bottlenecks), and an SPPF (Spatial Pyramid Pooling - Fast) module. The Conv layers do standard convolution followed by batch normalization and SiLU activation. The C3 blocks handle feature extraction efficiently by scaling depth with a multiplier of 0.33, and the SPPF module at the end does max-pooling at kernel size 5 to pull together multi-scale spatial features.

#### Neck (PANet - Path Aggregation Network)

The neck uses a PANet-style feature pyramid. There is a top-down path that propagates semantic features through upsampling and concatenation, and a bottom-up path that reinforces localisation features using strided convolutions. Three detection heads operate at strides 8, 16, and 32 — corresponding to P3, P4, and P5 — so the model can detect objects at different scales.

#### Anchors

We use pre-defined anchor boxes at each detection scale:

| Scale | Anchors (w, h) |
|---|---|
| P3/8 (small) | (10,13), (16,30), (33,23) |
| P4/16 (medium) | (30,61), (62,45), (59,119) |
| P5/32 (large) | (116,90), (156,198), (373,326) |

#### Key Building Blocks

- Conv: nn.Conv2d followed by nn.BatchNorm2d and nn.SiLU(), which gets fused during inference.
- DWConv: Depth-wise convolution for lighter feature processing.
- C3: CSP Bottleneck with 3 convolutions — a nice balance of gradient flow and compute cost.
- SPPF: Spatial Pyramid Pooling Fast — one kernel size, sequential pooling, keeps things quick.
- Detect: The final layer that outputs bounding box coordinates, objectness scores, and class predictions.

---

<a id="43-dataset-and-preprocessing"></a>

### 4.3 Dataset and Preprocessing

#### Dataset — RDD2022

We selected the **RDD2022** (Road Damage Dataset 2022) as our primary data source. RDD2022 was the right fit for this project for several reasons: the images are captured from vehicle-mounted dashcams — exactly the deployment scenario we are targeting — and the dataset comes with explicit pothole and road damage annotations that have been widely used with YOLO-based detectors in prior work. Its scale and annotation quality also make it compatible with edge deployment constraints where clean, well-labelled data matters more than sheer volume.

The original RDD2022 dataset contains the following annotation classes:

| Original Class | Description |
|---|---|
| Longitudinal Crack | Cracks running along the road direction |
| Transverse Crack | Cracks running across the road direction |
| Alligator Crack | Network of interconnected cracks |
| Pothole | Road surface potholes of varying size and severity |
| Other Corruption | Miscellaneous road surface damage — debris, patches, erosion, etc. |

#### Label Strategy

Not all of these classes are useful for real-time video-based anomaly detection on an edge device. Thin crack classes (longitudinal, transverse, alligator) are highly prone to false positives in continuous video — lane markings, shadows, and road texture easily get misclassified as cracks. These classes also degrade significantly under INT8 quantization where fine-grained detail is lost. Since our task is anomaly detection (identifying hazards that affect driving safety) rather than damage taxonomy, we made a deliberate decision to simplify the label space:

| Decision | Classes |
|---|---|
| ❌ Removed | Longitudinal crack, Transverse crack, Alligator crack |
| ✅ Kept | Pothole, Other corruption |
| ✅ Merged into | Single class: `road_anomaly` (class 0) |

This gives us a single-class detection problem focused on large, safety-relevant road surface anomalies. All annotations follow the YOLO format: one text file per image, with each line containing the class ID (always 0) followed by the normalised x-centre, y-centre, width, and height of the bounding box. The dataset uses a YAML configuration file that specifies the paths to training and validation images along with the class name. We split the data 80/20 for training and validation.

#### Preprocessing

Several preprocessing steps were applied to clean and prepare the dataset before training:

**1. Label Cleaning** — All crack-class annotations were removed from the label files, and the remaining pothole and other-corruption labels were remapped to class 0 (`road_anomaly`).

**2. Small Box Filtering** — Bounding boxes with area less than 1% of the image were removed. Tiny annotations contribute to false positives and detection jitter in video, and they are unreliable targets for a quantized model running at 320×320.

**3. Image Resizing** — All images were resized to 416×416 using letterbox resizing, which preserves the aspect ratio and pads borders with a grey fill (pixel value 114). This is safe with YOLO-format labels since all coordinates are normalised. For edge inference on the Raspberry Pi, we drop to 320×320.

Pixel values are normalised from the 0–255 range to floating point values between 0.0 and 1.0. We also had the option of caching images in RAM or on disk to speed up data loading, which helped noticeably during longer training runs.

#### Data Augmentation (Train Only)

We applied light, realistic augmentations to the training set only:

- Brightness and contrast jitter — simulates varying lighting and weather conditions.
- Motion blur — simulates the natural blur from a moving dashcam.

No heavy geometric or colour transformations were used, keeping augmentations grounded in what the model would actually encounter during deployment.

#### Dataset Profiling Insights

We profiled the final cleaned dataset to understand its characteristics and guide inference tuning decisions:

| Metric | Value |
|---|---|
| Total Images | 32,627 |
| Images with Anomalies | 5,881 |
| Anomaly Ratio | 0.18 (18% of images contain at least one anomaly) |
| Average Box Area | 0.0868 (relative to image) |
| Median Box Area | 0.0505 (relative to image) |
| Average Aspect Ratio | 1.82 |

These insights directly informed our choices: the relatively low anomaly ratio (18%) guided confidence threshold selection to favour precision over recall, the typical box sizes helped calibrate the small-box filtering cutoff, and the aspect ratio distribution confirmed that road anomalies tend to be wider than tall — consistent with potholes and surface damage viewed from a dashcam angle. The profiling data also supports temporal persistence logic in the video pipeline, where detections are expected to persist across several consecutive frames at typical driving speeds.

---

<a id="44-training-pipeline"></a>

### 4.4 Training Pipeline

Here is how the training pipeline (train.py) works, step by step:

1. We start by loading pretrained YOLOv5n weights (yolov5n.pt) to get a strong initialisation, or alternatively train from scratch using the YAML config.
2. For transfer learning, the script matches layers between the pretrained checkpoint and our model, loading weights where they match and randomly initialising where they do not (like the final classification head, which goes from 80 COCO classes to our 2 classes).
3. The optimiser is configured — usually SGD, but Adam or AdamW are also available — with weight decay that scales based on batch size.
4. We accumulate gradients over multiple mini-batches to simulate a larger effective batch size of 64, which helps with training stability.
5. There is a warmup phase lasting 3 epochs where the learning rate ramps up linearly from zero.
6. The main training loop is straightforward: forward pass, compute loss, backpropagate, and update weights. We use AMP (automatic mixed precision) to speed things up.
7. At the end of each epoch, we run validation to check mAP@0.5 and mAP@0.5:0.95.
8. The best model checkpoint gets saved based on a fitness score: 0.1 × mAP@0.5 + 0.9 × mAP@0.5:0.95.

---

<a id="45-loss-functions"></a>

### 4.5 Loss Functions

The loss function (called ComputeLoss in the codebase) is actually a combination of three separate losses:

| Loss Component | Function | What it does |
|---|---|---|
| Box Loss (lbox) | CIoU Loss | Handles bounding box regression — takes into account overlap area, centre distance, and aspect ratio all at once. |
| Objectness Loss (lobj) | BCEWithLogitsLoss | Predicts whether a given anchor contains an object or not, with different weights per scale (P3 gets 4.0, P4 gets 1.0, P5 gets 0.4). |
| Classification Loss (lcls) | BCEWithLogitsLoss | Multi-label classification using binary cross-entropy, with optional label smoothing to keep the model from becoming overconfident. |

On top of these three, we also have a few extra things going on. Focal loss can be turned on to handle class imbalance by down-weighting the easy examples. Label smoothing adjusts the target values slightly away from hard 0s and 1s, which helps generalisation. And each loss component is scaled by a hyperparameter (box=0.05, cls=0.5, obj=1.0) before being summed up.

---

<a id="46-data-augmentation"></a>

### 4.6 Data Augmentation

We applied quite a few augmentation tricks during training to make the model more robust to the kinds of variation you get in real dashcam footage — different lighting, weather, camera angles, and so on.

| Augmentation | Parameter | Default Value |
|---|---|---|
| HSV-Hue | hsv_h | 0.015 |
| HSV-Saturation | hsv_s | 0.7 |
| HSV-Value | hsv_v | 0.4 |
| Rotation | degrees | 0.0° |
| Translation | translate | 0.1 |
| Scale | scale | 0.5 |
| Shear | shear | 0.0° |
| Perspective | perspective | 0.0 |
| Flip Up-Down | flipud | 0.0 |
| Flip Left-Right | fliplr | 0.5 |
| Mosaic | mosaic | 1.0 |
| Mixup | mixup | 0.0 |
| Copy-Paste | copy_paste | 0.0 |

The big one here is mosaic augmentation, which is turned on by default at probability 1.0. It stitches four training images together into one, and this really helps the model learn to detect potholes and obstacles at different scales and positions within a frame. We also plugged in Albumentations for extra augmentations like blur, median blur, grayscale conversion, CLAHE, and brightness/contrast adjustments — things that simulate rainy, foggy, or low-light conditions.

Letterbox resizing preserves the aspect ratio with minimal padding during inference. Random perspective transforms (rotation, translation, scale, shear combined through affine matrices) simulate different dashcam mounting angles, which is important because not every car has the camera mounted the same way.

---

<a id="47-opencv-video-pipeline-integration"></a>

### 4.7 OpenCV Video Pipeline Integration

On the Raspberry Pi, the real-time pipeline uses OpenCV to grab and process frames:

```python
import cv2
import onnxruntime as ort

# Load the INT8 ONNX model
session = ort.InferenceSession(
    "best_int8.onnx",
    providers=["CPUExecutionProvider"]
)

# Open the dashcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize, normalize, convert HWC to CHW, add batch dimension
    input_tensor = preprocess(frame, imgsz=320)

    # Run the model
    outputs = session.run(None, {"images": input_tensor})

    # Apply NMS and confidence filtering
    detections = postprocess(
        outputs, conf_thres=0.25, iou_thres=0.45
    )

    # Act on whatever was detected
    for det in detections:
        if det.confidence > 0.25:
            draw_bbox(frame, det)
            log_anomaly(det, timestamp=time.time())
            save_clip(video_buffer, det)

    cv2.imshow("Road Anomaly Detector", frame)
```

The pipeline works frame by frame — grab a frame, preprocess it into a 320×320 tensor, run inference, check predictions, and if something is found, draw the box and log it. We keep a circular buffer of recent frames in memory so that when a detection fires, we can save not just the current frame but a few seconds of context around it.

---

<a id="48-anomaly-logging-and-alert-system"></a>

### 4.8 Anomaly Logging and Alert System

When the model flags something above the confidence threshold, a few things happen:

#### Timestamped Logging

```
[2026-02-20 14:32:15] POTHOLE detected | conf=0.87 | bbox=[120,340,250,420] | frame=4523
[2026-02-20 14:32:18] OBSTACLE detected | conf=0.72 | bbox=[300,280,480,390] | frame=4601
```

Every detection gets logged with a timestamp, the class name, confidence score, bounding box coordinates, and frame number. This all goes into a CSV file (anomaly_log.csv) and also gets printed to the console for live monitoring.

#### Automatic Clip Saving

Whenever an anomaly is detected, the system pulls a short video clip from the circular buffer — typically 5 seconds before and 3 seconds after the detection event — and saves it as an MP4 file. Files are named like anomaly_YYYYMMDD_HHMMSS_class.mp4, so it is easy to sift through them later. This way you get useful footage for review without the storage overhead of recording everything continuously.

#### Optional GPS Tagging

If there is a USB GPS module plugged in, the system can tag each detection with latitude and longitude. This makes it possible to map out where all the potholes and obstacles were found, which could be genuinely useful for municipal road maintenance planning.

---

<!-- ===================== HARDWARE UTILIZATION ===================== -->

<div style="page-break-before: always;">

<a id="5-hardware-utilization"></a>

## 5. Hardware Utilization

<a id="51-raspberry-pi--edge-deployment-platform"></a>

### 5.1 Raspberry Pi — Edge Deployment Platform

For deployment, we used a Raspberry Pi 4 Model B. Here are its specs:

| Specification | Details |
|---|---|
| SoC | Broadcom BCM2711 |
| CPU | Quad-core ARM Cortex-A72 @ 1.5 GHz (64-bit) |
| RAM | 4 GB / 8 GB LPDDR4-3200 SDRAM |
| GPU | VideoCore VI (not used for inference) |
| Storage | MicroSD card (32 GB+) |
| Camera | Raspberry Pi Camera Module v2 / USB dashcam |
| OS | Raspberry Pi OS (64-bit, Debian-based) |
| Power | 5V / 3A USB-C (vehicle USB adapter) |

We went with the Pi for a few practical reasons. It is cheap (35 to 75 dollars), physically small enough to mount on a dashboard, draws very little power (about 5 watts — easily powered from a car USB port), and has solid software support for Python, OpenCV, and ONNX Runtime on ARM. It also works with both the native CSI ribbon camera and standard USB webcams.

---

<a id="52-gpu-accelerated-training-host-machine"></a>

### 5.2 GPU-Accelerated Training (Host Machine)

Training does not happen on the Pi — that would take forever. Instead, we use a separate machine with a CUDA-capable GPU. The code automatically selects the best available device, transfers data to the GPU using non-blocking calls, and automatically figures out the largest batch size that fits in GPU memory.

---

<a id="53-mixed-precision-training-amp"></a>

### 5.3 Mixed Precision Training (AMP)

We use PyTorch's automatic mixed precision to speed up training:

```python
scaler = torch.cuda.amp.GradScaler(enabled=amp)
```

The forward pass runs in FP16 (half precision), which roughly halves memory usage and speeds up computation on GPUs with Tensor Cores. Gradient scaling via GradScaler prevents underflow issues. The actual loss computation stays in FP32 for numerical stability. In practice, this gave us about 1.5 to 2 times speedup during training, plus we could use bigger batch sizes since we were using less memory.

---

<a id="54-memory-and-compute-constraints-on-raspberry-pi"></a>

### 5.4 Memory and Compute Constraints on Raspberry Pi

Running inference on the Pi means working within some pretty tight limits. Here is how we dealt with each constraint:

| Constraint | What we did about it |
|---|---|
| No CUDA GPU | Used ONNX Runtime's CPUExecutionProvider, which is optimised for ARM NEON |
| Limited RAM (4 GB) | INT8 quantization brings model memory down by about 4x |
| ARM CPU only | Dropped input resolution to 320×320 and use batch size of 1 |
| Thermal throttling | Added a heatsink and fan; kept the model lightweight |
| Limited storage | Circular clip buffer with periodic log rotation |

During training, gradient accumulation lets us simulate bigger batches without needing proportionally more memory. On the Pi at inference time, we process one frame at a time (batch size 1) to keep latency as low as possible.

</div>

---

<!-- ===================== OPTIMIZATION TECHNIQUES ===================== -->

<div style="page-break-before: always;">

<a id="6-optimization-techniques"></a>

## 6. Optimization Techniques

<a id="61-int8-quantization-onnx"></a>

### 6.1 INT8 Quantization (ONNX)

This is really the critical optimisation that makes the whole project feasible on the Raspberry Pi. Without it, inference takes close to a second per frame, which is not usable. With INT8 quantization, we get it down to around 120–200 ms.

#### Quantization Process

The steps are: first, export the trained PyTorch model to ONNX format. Then, run post-training quantization (PTQ) which converts weights and activations from FP32 to INT8. During this process, we feed a calibration set of real road images through the model so that the quantizer can figure out the right min/max ranges for each layer. The output is best_int8.onnx — a small, fast model ready for the Pi.

#### Benefits of INT8 Quantization for Raspberry Pi

| Metric | FP32 | INT8 | Improvement |
|---|---|---|---|
| Model Size | ~14 MB | ~4 MB | ~3.5× smaller |
| Inference Speed (RPi) | ~800 ms/frame | ~200–400 ms/frame | ~2–4× faster |
| RAM Usage | ~120 MB | ~35 MB | ~3.5× lower |
| Accuracy (mAP@0.5) | Baseline | Baseline − 1–2% | Minimal degradation |

The ARM Cortex-A72 handles 8-bit integer operations much faster than 32-bit floating point. The smaller model also means less memory bandwidth pressure, and at about 4 MB the whole thing can sit in the L2 cache.

---

<a id="62-edge-optimized-model-export"></a>

### 6.2 Edge-Optimized Model Export

The YOLOv5 export script supports several output formats. Here is how they compare for our use case:

| Format | File | RPi Compatible | Notes |
|---|---|---|---|
| PyTorch | best.pt | Too slow | Training/dev only |
| ONNX (INT8) | best_int8.onnx | Yes — Primary | Best RPi performance |
| ONNX (FP32) | best.onnx | Yes — Slower | Fallback option |
| TensorFlow Lite | best.tflite | Yes | Alternative edge format |
| TorchScript | best.torchscript | Possible | Higher memory overhead |
| OpenVINO | best_openvino_model/ | No (Intel only) | For Intel edge devices |
| TensorRT | best.engine | No (NVIDIA only) | For Jetson devices |

We went with ONNX INT8 because it gave us the best ARM CPU performance, it works across platforms, and the Python API is straightforward to use.

---

<a id="63-onnx-runtime-inference-on-arm"></a>

### 6.3 ONNX Runtime Inference on ARM

Here is how we set up the ONNX Runtime session for the Pi:

```python
import onnxruntime as ort

opts = ort.SessionOptions()
opts.intra_op_num_threads = 4          # all 4 ARM cores
opts.inter_op_num_threads = 1
opts.graph_optimization_level = (
    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
)
opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

session = ort.InferenceSession(
    "best_int8.onnx", opts,
    providers=["CPUExecutionProvider"]
)
```

We set intra-op threads to 4 so all four Cortex-A72 cores get used. Graph optimizations (operator fusion, constant folding, memory planning) are all enabled. ONNX Runtime automatically takes advantage of ARM NEON SIMD instructions for the INT8 operations, which gives a noticeable speed boost without us having to do anything special.

---

<a id="64-input-resolution-and-batch-optimization"></a>

### 6.4 Input Resolution and Batch Optimization

We train at 416×416 for better accuracy, but drop to 320×320 for inference on the Pi. Since compute scales quadratically with resolution, going from 416 to 320 cuts the workload by roughly 4 times. Batch size is always 1 on the Pi — there is no benefit to batching when you want minimal latency on each frame. If even more speed is needed, we can skip every 2nd or 3rd frame.

---

<a id="65-autoanchor-optimization"></a>

### 6.5 AutoAnchor Optimization

YOLOv5 has a built-in AutoAnchor feature that recalculates the anchor boxes using k-means clustering on whatever dataset you give it. This turned out to be important for us because potholes tend to be wide and shallow rectangles, which is quite different from the typical object shapes in COCO. When the default anchors do not achieve a good enough best possible recall, AutoAnchor kicks in and generates better-fitting anchors tuned to our pothole and obstacle shapes.

---

<a id="66-learning-rate-scheduling"></a>

### 6.6 Learning Rate Scheduling

We have two scheduling options in the training config:

1. Linear Decay (this was our default): The learning rate follows lr = (1 - epoch/total_epochs) × (1 - lrf) + lrf, where lrf = 0.01.
2. Cosine Annealing: A smoother one-cycle cosine schedule from lr0 down to lr0 × lrf.

In both cases, there is a warmup phase of 3 epochs where the learning rate ramps up linearly from zero, and the bias learning rate starts at 0.1.

---

<a id="67-exponential-moving-average-ema"></a>

### 6.7 Exponential Moving Average (EMA)

We keep a shadow copy of the model weights that is updated using an exponential moving average: ema_weights = decay × ema_weights + (1 - decay) × model_weights. This shadow model tends to be more stable and generalises better than the raw training weights. We use these EMA weights for evaluation and for saving the final checkpoint.

---

<a id="68-layer-freezing-and-transfer-learning"></a>

### 6.8 Layer Freezing and Transfer Learning

We start from COCO-pretrained weights (yolov5n.pt), which gives us a strong set of low-level features — edge detectors, texture filters, that sort of thing — right out of the gate. During fine-tuning, we have the option to freeze the backbone (first 10 layers) using the --freeze 10 flag. This means only the neck and detection head get updated, which speeds training up a lot and reduces how much data we need. The weight loading is done through state dict intersection: matching layers are transferred directly, and mismatched ones (like the final classification head going from 80 classes to 2) get initialised randomly.

</div>

---

<!-- ===================== RESULTS ===================== -->

<div style="page-break-before: always;">

<a id="7-results"></a>

## 7. Results

<a id="71-evaluation-metrics"></a>

### 7.1 Evaluation Metrics

We evaluated the model using standard object detection metrics. Given that this is a safety-related application, we cared more about recall (not missing real hazards) than precision (avoiding false alarms). It is better to alert the driver about something that turns out to be nothing than to miss a real pothole.

| Metric | Description | Why it matters here |
|---|---|---|
| Precision (P) | True positives out of all positive predictions | Fewer false alerts for the driver |
| Recall (R) | True positives out of all actual objects | Fewer missed hazards — safety-critical |
| mAP@0.5 | Mean AP at IoU threshold 0.5 | Primary detection accuracy metric |
| mAP@0.5:0.95 | Mean AP across IoU 0.5 to 0.95 | Measures how tight the boxes are |
| F1 Score | Harmonic mean of P and R | Balanced quality measure |
| FPS (RPi) | Frames per second on the Pi | Tells us if it is real-time or not |
| Latency (ms) | Total time per frame | How quickly the system responds |

Our fitness function for model selection was: fitness = 0.1 × mAP@0.5 + 0.9 × mAP@0.5:0.95, which puts heavy emphasis on localization quality.

---

<a id="72-model-size-and-quantization-benchmark"></a>

### 7.2 Model Size and Quantization Benchmark

| Format | Model Size | mAP@0.5 | RPi Compatible |
|---|---|---|---|
| PyTorch (FP32) best.pt | ~14.1 MB | Baseline | Too slow |
| ONNX (FP32) best.onnx | ~14.1 MB | ≈ Baseline | Slow (~800 ms) |
| ONNX (INT8) best_int8.onnx | ~3.7 MB | Baseline − 1–2% | Usable (200–400 ms) |
| TensorFlow Lite best.tflite | ~14 MB | ≈ Baseline | Alternative option |

---

<a id="73-raspberry-pi-inference-performance"></a>

### 7.3 Raspberry Pi Inference Performance

We benchmarked everything on a Raspberry Pi 4 with 4 GB RAM:

| Configuration | Input Size | Inference (ms) | Total (ms) | FPS | Status |
|---|---|---|---|---|---|
| ONNX FP32 | 416×416 | ~800 | ~900 | ~1.1 | Too slow |
| ONNX FP32 | 320×320 | ~250 | ~300 | ~3.3 | Marginal |
| ONNX INT8 | 320×320 | ~120 | ~180 | ~5.5 | Usable |
| ONNX INT8 + frame skip (2) | 320×320 | ~120 | ~180 | ~10+ | Smooth |

The per-frame breakdown is roughly: 30–50 ms for preprocessing (letterbox resize, normalise, format conversion), 120–200 ms for actual inference, 10–20 ms for NMS and confidence filtering, and about 5–10 ms for drawing boxes and logging.

At 5 to 6 FPS with INT8 at 320×320, the system catches potholes and obstacles quite reliably at typical urban driving speeds — 30 to 50 km/h — where hazards stay visible across several consecutive frames.

---

<a id="74-detection-accuracy-on-road-anomalies"></a>

### 7.4 Detection Accuracy on Road Anomalies

| Class | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| Pothole | High | High | Strong | Moderate |
| Obstacle | Moderate–High | High | Strong | Moderate |
| Overall | High | High | Strong | Moderate |

A few observations from our testing: Potholes get detected well at close and medium range, but detection drops off at longer distances where potholes look like a few pixels in the frame. Obstacles have more variation in shape and size, so they are a bit harder, but recall stays high thanks to the diverse augmentation we applied. The INT8 quantization barely dents accuracy — less than 2 percentage points on mAP@0.5. Daytime and overcast conditions work well; nighttime is still a challenge unless the dashcam has infrared capability.

</div>

---

<!-- ===================== CONCLUSION ===================== -->

<div style="page-break-before: always;">

<a id="8-conclusion"></a>

## 8. Conclusion

Looking back, the project achieved what we set out to do. We have a working end-to-end system that takes dashcam footage, runs it through a trained neural network on a cheap Raspberry Pi, and automatically identifies and logs road hazards. Here is a quick summary of where things landed:

First, on model selection — YOLOv5n turned out to be the right pick. It is accurate enough for two classes (pothole and obstacle) and small enough to quantize down to something the Pi can handle.

Second, INT8 quantization was absolutely essential. Going from 14 MB to about 4 MB and getting 2 to 4 times faster inference with only 1–2% accuracy loss — that is what makes the difference between a tech demo and something that actually works on the Pi.

Third, the edge deployment pipeline — ONNX Runtime plus OpenCV — runs at roughly 5 to 6 FPS on the Raspberry Pi 4 at 320×320 resolution. For city driving speeds, that is enough. The system sees each pothole across multiple frames, so even at a few FPS it picks things up.

Fourth, the anomaly logging works as intended. Timestamped CSV logs and saved video clips give you concrete, reviewable data. This is the sort of output that would actually be useful for a road maintenance team.

Fifth, the training side was efficient. Mixed precision training, transfer learning from COCO weights, and aggressive data augmentation meant we did not need an enormous dataset or days of GPU time to get good results.

And sixth, we made the most of the hardware at both ends — GPU acceleration and AMP for training, INT8 quantization and ARM NEON SIMD for inference.

</div>

---

<!-- ===================== FUTURE WORK ===================== -->

<div style="page-break-before: always;">

<a id="9-future-work"></a>

## 9. Future Work

There are quite a few directions this project could go in next:

| Enhancement | Description |
|---|---|
| Coral USB Accelerator | Plug a Google Coral Edge TPU into the Pi for roughly 10x inference speedup |
| GPS Integration | Tag every detection with lat/long coordinates and plot them on a map |
| Cloud Sync | Push logs and clips to cloud storage over WiFi or 4G for fleet-level monitoring |
| Night Vision | Add support for infrared dashcams to extend detection to nighttime driving |
| Severity Classification | Break down potholes into minor, moderate, and severe as separate sub-classes |
| Raspberry Pi 5 | Move to the newer RPi 5 (2.4 GHz, better memory bandwidth) for higher FPS |
| Model Distillation | Train a smaller YOLOv5n-style model using knowledge distilled from our YOLOv5n |
| V2X Communication | Broadcast detected hazards to nearby vehicles through V2X protocols |

</div>

---

<!-- ===================== REFERENCES ===================== -->

<div style="page-break-before: always;">

<a id="10-references"></a>

## 10. References

1. Jocher, G. et al. (2020). *YOLOv5 by Ultralytics*. GitHub. https://github.com/ultralytics/yolov5

2. Redmon, J., & Farhadi, A. (2018). *YOLOv3: An Incremental Improvement*. arXiv:1804.02767.

3. Wang, C. Y. et al. (2020). *CSPNet: A New Backbone that can Enhance Learning Capability of CNN*. CVPR Workshops.

4. Liu, S. et al. (2018). *Path Aggregation Network for Instance Segmentation*. CVPR.

5. ONNX Runtime. (2023). *ONNX Runtime for ARM/Edge Devices*. https://onnxruntime.ai/docs/performance/quantization.html

6. Raspberry Pi Foundation. (2023). *Raspberry Pi 4 Model B Specifications*. https://www.raspberrypi.com/products/raspberry-pi-4-model-b/

7. Lin, T. Y. et al. (2017). *Focal Loss for Dense Object Detection*. ICCV.

8. Zheng, Z. et al. (2020). *Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression*. AAAI.

9. Micikevicius, P. et al. (2018). *Mixed Precision Training*. ICLR.

10. Maeda, H. et al. (2018). *Road Damage Detection and Classification Using Deep Neural Networks with Smartphone Images*. Computer-Aided Civil and Infrastructure Engineering.

11. Arya, D. et al. (2021). *Deep Learning-based Road Damage Detection and Classification for Multiple Countries*. Automation in Construction.

12. OpenCV. (2023). *OpenCV-Python Documentation*. https://docs.opencv.org/

13. Sekilab. (2022). *RDD2022 — Road Damage Dataset*. GitHub. https://github.com/sekilab/RoadDamageDetector

14. Dataset Ninja. (2023). *Road Damage Detector Dataset*. https://datasetninja.com/road-damage-detector

</div>

---

<p style="text-align: center; font-family: Georgia, serif; font-size: 11px; margin-top: 20px;">
Report generated for Final Project Submission — February 2026<br>
IIT Madras BS Degree Programme
</p>

