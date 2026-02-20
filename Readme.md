<div align="center">

# Dashcam-Based Road Anomaly Detection — Edge Deployment on Raspberry Pi

**A compact AI system that spots potholes and road hazards in real time using a YOLOv5s detector quantized to INT8 and running entirely on a Raspberry Pi 4.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![YOLOv5](https://img.shields.io/badge/YOLOv5-v7.0-brightgreen?logo=yolo)](https://github.com/ultralytics/yolov5)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-INT8-orange?logo=onnx)](https://onnxruntime.ai/)
[![Raspberry Pi](https://img.shields.io/badge/Raspberry_Pi_4-ARM-c51a4a?logo=raspberrypi)](https://www.raspberrypi.com/)
[![License](https://img.shields.io/badge/License-AGPL--3.0-red)](LICENSE)

</div>

---

## What Is This Project About?

Bad roads are a serious safety concern — potholes cause tire blowouts, suspension damage, and accidents, especially when drivers can't react in time. This project tries to tackle that problem using a small, affordable AI system built around a Raspberry Pi and a regular dashcam.

The idea is straightforward: mount a camera on the dashboard, feed the video into a lightweight object detection model (YOLOv5s), and have it flag road surface anomalies like potholes and unexpected obstacles — all in real time, right on the device, without needing cloud connectivity or a powerful GPU.

To make this feasible on a $35 single-board computer, the trained model goes through INT8 quantization, shrinking it from ~14 MB down to about 4 MB while keeping accuracy within an acceptable range. The result is a system that processes roughly 5–6 frames per second on four ARM cores — fast enough for typical city driving speeds.

When an anomaly is spotted, the system saves a short video clip and logs the event with a timestamp so you can review incidents later.

---

## Highlights

- Detects potholes and road obstacles from live dashcam footage
- Runs on a Raspberry Pi 4 with no GPU — just four ARM CPU cores
- INT8-quantized ONNX model: roughly 3.5x smaller and 2–4x faster than the FP32 version
- OpenCV handles the entire video pipeline — capture, resize, annotate, display
- Every detection is timestamped and logged to CSV (class, confidence, bounding box coordinates, frame index)
- Automatically saves short video clips whenever an anomaly is detected
- Trained using transfer learning from COCO-pretrained YOLOv5s weights
- Mixed-precision training (AMP) keeps GPU training efficient
- Augmentation pipeline includes mosaic composition, HSV colour jitter, motion blur, and brightness/contrast shifts
- Can export to ONNX, TorchScript, TFLite, OpenVINO, TensorRT, CoreML, and other formats

---

## How It Works

The project breaks down into two distinct phases:

### Phase 1 — Training on a GPU Machine

You start with a labelled road anomaly dataset (we used RDD2022). The images and their YOLO-format annotations go through a data preparation pipeline — label merging, small-box filtering, resizing, and light augmentations. Then a YOLOv5s model is trained on a GPU using transfer learning from COCO pretrained weights.

Once training is done, the best checkpoint (`best.pt`) gets exported to ONNX format and then quantized to INT8 using ONNX Runtime's post-training quantization. The output is a ~4 MB file called `best_int8.onnx`.

```
Labelled Dataset (RDD2022, cleaned and filtered)
        |
YOLOv5s Training (GPU, mixed precision, transfer learning)
        |
Best Checkpoint — best.pt
        |
ONNX Export + INT8 Quantization
        |
best_int8.onnx (~4 MB, edge-ready)
```

### Phase 2 — Live Inference on Raspberry Pi

On the Pi, a Python script opens the dashcam feed using OpenCV and the Pi Camera 2 library. Each frame is resized to 416x416 pixels, rearranged from HWC to CHW layout, and fed into the ONNX Runtime session configured with four threads. After inference, a confidence threshold filters out weak predictions. If a detection passes the threshold, the system draws a bounding box, appends the event to a log, and triggers a background thread that saves a rolling video buffer as an MP4 clip.

```
Pi Camera / Dashcam
        |
OpenCV Frame Capture (640x480)
        |
Resize to 416x416 + Normalize + Transpose
        |
ONNX Runtime Inference (INT8, 4 threads)
        |
Confidence Check
        |
    Detected? ── Yes ──> Log to CSV + Save video clip + Draw box
        |
    No ──> Next frame
```

### About the Model

We use YOLOv5-Small, which sits in a sweet spot between accuracy and speed for edge devices.

| Part | What It Does |
|---|---|
| Backbone | CSPDarknet53 with C3 bottleneck blocks and an SPPF pooling module |
| Neck | PANet — merges features from multiple scales (top-down and bottom-up) |
| Detection Head | Three heads at strides 8, 16, and 32 covering small, medium, and large objects |
| Parameters | Around 7.2 million |
| Depth / Width Multipliers | 0.33 / 0.50 (keeps the model compact) |

---

## Repository Layout

The repo is organised into three main areas — data preparation, model building (training + export), and inference:

```
.
├── data_pre-proc/               # Scripts for cleaning and preparing the dataset
│   ├── merge_labels.py          # Drops crack classes, remaps pothole + corruption to class 0
│   ├── filter_small_boxes.py    # Removes bounding boxes smaller than 1% of image area
│   ├── resize_images.py         # Resizes all images to 416x416
│   ├── augment_images.py        # Brightness/contrast jitter and motion blur for training set
│   ├── dataset_profile.py       # Prints stats — anomaly frequency, box sizes, aspect ratios
│   ├── requirements.txt         # numpy, opencv-python
│   └── README.md                # Notes on the data pipeline and RDD2022 choices
│
├── model_building/              # YOLOv5-based training, export, and evaluation
│   ├── train.py                 # Train YOLOv5 on a custom dataset
│   ├── detect.py                # Run detection on images, videos, webcam, or streams
│   ├── val.py                   # Evaluate mAP, precision, recall on a validation split
│   ├── export.py                # Convert trained weights to ONNX, TFLite, TorchScript, etc.
│   ├── benchmarks.py            # Compare inference speed and accuracy across export formats
│   ├── best_int8.onnx           # Ready-to-deploy INT8 quantized model (~4 MB)
│   ├── data.yaml                # Dataset config — paths and class names (road_anomaly)
│   ├── requirements.txt         # Full dependency list (PyTorch, OpenCV, ultralytics, etc.)
│   ├── hubconf.py               # PyTorch Hub integration
│   ├── tutorial.ipynb           # Interactive walkthrough notebook
│   ├── models/                  # Architecture definitions (.yaml configs + Python modules)
│   │   ├── common.py            # Core blocks — Conv, C3, SPPF, Detect, and more
│   │   ├── yolo.py              # Model construction logic
│   │   ├── yolov5s.yaml         # Primary config (Small variant)
│   │   ├── hub/                 # Alternate architectures (P6, BiFPN, ghost, transformer)
│   │   └── segment/             # Segmentation model definitions
│   ├── data/                    # Dataset YAML configs and download helpers
│   │   ├── hyps/                # Hyperparameter presets (low, med, high, no-aug, VOC, Objects365)
│   │   ├── images/              # Sample test images
│   │   └── scripts/             # Shell scripts to fetch COCO, ImageNet, etc.
│   ├── utils/                   # Training and inference utilities
│   │   ├── general.py           # NMS, file I/O, logging helpers
│   │   ├── dataloaders.py       # Data loading with caching and augmentation
│   │   ├── augmentations.py     # Mosaic, HSV, perspective, and Albumentations transforms
│   │   ├── loss.py              # CIoU box loss, BCE objectness and classification losses
│   │   ├── metrics.py           # mAP, precision, recall computation
│   │   ├── torch_utils.py       # EMA, device selection, model profiling
│   │   ├── autoanchor.py        # Automatic anchor fitting via k-means
│   │   └── loggers/             # TensorBoard, Weights & Biases, ClearML, Comet
│   ├── classify/                # Image classification scripts (train, predict, val)
│   └── segment/                 # Instance segmentation scripts (train, predict, val)
│
├── inference/
│   └── inference.py             # Raspberry Pi deployment script (Pi Camera 2 + ONNX Runtime)
│
└── Readme.md                    # You are here
```

---

## Getting Started

### What You Need

**Software:**

| Dependency | Minimum Version |
|---|---|
| Python | 3.8 |
| PyTorch | 1.8.0 |
| CUDA (training only) | 11.0 |
| ONNX Runtime | 1.10.0 |
| OpenCV | 4.1.1 |

**Hardware:**

- **For training:** A machine with an NVIDIA GPU. CPU training works but is very slow.
- **For inference:** A Raspberry Pi 4 Model B with at least 4 GB RAM, or really any computer that can run Python and ONNX Runtime.

### Setup

**1. Grab the code**

```bash
git clone https://github.com/Bharat-AI-SoC/anomaly-detection-dashcam.git
cd anomaly-detection-dashcam
```

**2. Set up a virtual environment (recommended)**

```bash
python -m venv venv

# On Linux or macOS
source venv/bin/activate

# On Windows
venv\Scripts\activate
```

**3. Install the core dependencies**

```bash
cd model_building
pip install -r requirements.txt
```

**4. If you plan to export to additional formats**

```bash
pip install onnx onnx-simplifier onnxruntime   # ONNX support
pip install coremltools                          # CoreML
pip install tensorflow                           # TensorFlow / TFLite
pip install openvino-dev                         # OpenVINO
```

---

## Data Preparation

We built our dataset from **RDD2022** (Road Damage Detection 2022), a publicly available collection of dashcam images with annotated road damage. Here is why we picked it and what we changed:

### Why RDD2022?

- The images look like what a real dashcam would capture — shot from vehicle-mounted cameras
- It comes with bounding box annotations for different types of road damage
- It has been widely used alongside YOLO-based detectors in prior research
- The image sizes and annotation style fit well with edge deployment constraints

### What We Changed

The original dataset includes five classes — longitudinal cracks, transverse cracks, alligator cracks, potholes, and other corruption. For our use case, crack detection on a moving vehicle is unreliable (the boxes are tiny and noisy under INT8 quantization), so we dropped all three crack classes entirely. We kept potholes and "other corruption" and merged them into a single unified class called `road_anomaly` (class 0).

### Cleaning Pipeline

Each script in the `data_pre-proc/` folder handles one step:

1. **merge_labels.py** — Filters out crack annotations (classes 0–2 in RDD2022), keeps classes 3 and 4, and remaps everything to class 0
2. **filter_small_boxes.py** — Drops any bounding box whose area is less than 1% of the image — these are too small to detect reliably and tend to cause false positives
3. **resize_images.py** — Resizes all images to 416x416 pixels (safe to do since YOLO annotations use normalised coordinates)
4. **augment_images.py** — Applies light augmentations to the training split only: random brightness/contrast shifts and occasional motion blur to simulate dashcam conditions
5. **dataset_profile.py** — Prints summary statistics to help choose thresholds and understand the data

### Dataset Statistics (After Cleaning)

| Metric | Value |
|---|---|
| Total images | 32,627 |
| Images containing anomalies | 5,881 |
| Anomaly presence ratio | 18% |
| Average bounding box area | 8.7% of image |
| Median bounding box area | 5.1% of image |
| Average aspect ratio | 1.82 |

The relatively low anomaly ratio (only 18% of images contain a labelled anomaly) means the model needs to learn to stay quiet on clean road surfaces — favouring precision over recall was a deliberate choice to avoid constant false alerts.

---

## Usage

### Training the Model

Train the YOLOv5s detector on your prepared dataset:

```bash
cd model_building

# Using COCO-pretrained weights as a starting point (recommended)
python train.py --data data.yaml --weights yolov5s.pt --img 416 --epochs 100

# Training from a blank slate
python train.py --data data.yaml --weights '' --cfg models/yolov5s.yaml --img 416

# Fine-tune with specific hyperparameters and a custom batch size
python train.py --data data.yaml --weights yolov5s.pt --img 416 \
    --batch-size 16 --epochs 200 --hyp data/hyps/hyp.scratch-med.yaml

# Distribute across multiple GPUs
python -m torch.distributed.run --nproc_per_node 4 train.py \
    --data data.yaml --weights yolov5s.pt --img 416 --device 0,1,2,3

# Freeze the backbone to only train the head — handy when data is limited
python train.py --data data.yaml --weights yolov5s.pt --img 416 --freeze 10
```

**Commonly used flags:**

| Flag | Default | What It Does |
|---|---|---|
| `--data` | `data/coco128.yaml` | Points to your dataset config file |
| `--weights` | `yolov5s.pt` | Starting weights — leave empty for training from scratch |
| `--cfg` | `''` | Model architecture definition (YAML) |
| `--img` | `416` | Training image resolution |
| `--batch-size` | `16` | Batch size across all GPUs |
| `--epochs` | `100` | How many epochs to train |
| `--hyp` | `hyp.scratch-low.yaml` | Hyperparameter preset |
| `--device` | `''` | GPU index or `cpu` |
| `--freeze` | `[0]` | Freeze this many layers from the top |

After training finishes, the results land in `runs/train/exp/`. You will find `weights/best.pt` (the best checkpoint based on fitness score), training curves, a confusion matrix, and PR curves.

### Exporting to ONNX and Quantizing to INT8

Once you have a trained `best.pt`, export it:

```bash
# Standard ONNX export
python export.py --weights runs/train/exp/weights/best.pt --include onnx --img 320

# Export to several formats at once
python export.py --weights best.pt --include onnx torchscript tflite

# Dynamic batch size for flexible deployment
python export.py --weights best.pt --include onnx --dynamic
```

Then apply INT8 quantization to compress the model for edge use:

```python
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    "best.onnx",
    "best_int8.onnx",
    weight_type=QuantType.QInt8
)
```

The repository already includes a pre-quantized `best_int8.onnx` so you can skip this step if you just want to run inference.

**Export formats this codebase supports:**

| Format | Flag | Output |
|---|---|---|
| PyTorch | — | `best.pt` |
| TorchScript | `torchscript` | `best.torchscript` |
| ONNX | `onnx` | `best.onnx` |
| OpenVINO | `openvino` | `best_openvino_model/` |
| TensorRT | `engine` | `best.engine` |
| CoreML | `coreml` | `best.mlpackage` |
| TF SavedModel | `saved_model` | `best_saved_model/` |
| TF Lite | `tflite` | `best.tflite` |
| TF Edge TPU | `edgetpu` | `best_edgetpu.tflite` |
| PaddlePaddle | `paddle` | `best_paddle_model/` |

### Running Detection

You can point the detector at almost any source — single images, video files, webcams, RTSP streams, or even YouTube URLs:

```bash
# Live webcam feed
python detect.py --weights best_int8.onnx --source 0

# Single image
python detect.py --weights best_int8.onnx --source path/to/image.jpg

# Dashcam recording
python detect.py --weights best_int8.onnx --source path/to/dashcam_video.mp4

# A whole folder of images
python detect.py --weights best_int8.onnx --source path/to/images/

# RTSP network stream
python detect.py --weights best_int8.onnx --source 'rtsp://example.com/media.mp4'

# Tweak confidence and NMS thresholds
python detect.py --weights best_int8.onnx --source 0 --conf-thres 0.3 --iou-thres 0.5

# Save detections to text files and CSV
python detect.py --weights best_int8.onnx --source path/to/video.mp4 --save-txt --save-csv
```

Results go to `runs/detect/exp/`.

### Validating Performance

```bash
# With the quantized model
python val.py --weights best_int8.onnx --data data.yaml --img 320

# With the full PyTorch checkpoint
python val.py --weights runs/train/exp/weights/best.pt --data data.yaml --img 416

# Detailed per-class breakdown
python val.py --weights best.pt --data data.yaml --verbose
```

### Deploying on a Raspberry Pi

**Step 1: Copy the model over**

```bash
scp model_building/best_int8.onnx pi@<your-pi-ip>:~/anomaly-detector/
```

**Step 2: Install dependencies on the Pi**

```bash
pip install opencv-python-headless numpy onnxruntime picamera2
```

**Step 3: Run the live detection script**

The `inference/inference.py` script is purpose-built for the Pi. It opens a Pi Camera 2 feed at 640x480, runs each frame through the INT8 model, and when something is detected above the confidence threshold (0.5 by default), a background thread saves the last 30 frames as an MP4 clip. A small live preview window shows an FPS counter and a red dot when an anomaly is spotted.

```bash
cd ~/anomaly-detector
python inference.py
```

Press **q** to quit the live view.

You can also use the general-purpose detection script if you prefer:

```bash
python detect.py --weights best_int8.onnx --source 0 --img 320 --device cpu --view-img
```

---

## Training Configuration and Loss Details

### What the Model Learns to Detect

| Class ID | Label | Description |
|---|---|---|
| 0 | road_anomaly | Potholes, large surface damage, and unexpected road obstructions |

We intentionally collapsed everything into one class. The goal here is anomaly alerting, not a fine-grained damage taxonomy.

### Loss Functions

| Loss Component | Implementation | Role |
|---|---|---|
| Box regression | CIoU (Complete IoU) | Accounts for overlap, centre distance, and aspect ratio simultaneously |
| Objectness | BCE with logits | Predicts whether each anchor contains an object, balanced across detection scales |
| Classification | BCE with logits | Multi-label class prediction (though we only have one class, the architecture supports more) |

### Key Hyperparameters

| Setting | Value |
|---|---|
| Optimiser | SGD with momentum 0.937 and weight decay 5e-4 |
| Starting learning rate | 0.01 |
| LR schedule | Cosine decay down to 1% of initial |
| Warmup | 3 epochs |
| Training resolution | 416 x 416 |
| Mosaic augmentation | On, probability 1.0 |
| Horizontal flip | Probability 0.5 |
| HSV jitter | Hue ±0.015, Saturation ±0.7, Value ±0.4 |
| Best-model selection | Weighted fitness: 10% mAP@0.5 + 90% mAP@0.5:0.95 |

---

## Performance Numbers

### Model Size After Quantization

| Variant | Size on Disk | Accuracy Impact | Works on RPi? |
|---|---|---|---|
| PyTorch FP32 (`best.pt`) | ~14 MB | Baseline | Far too slow |
| ONNX FP32 (`best.onnx`) | ~14 MB | Same as baseline | Barely (~1 FPS) |
| **ONNX INT8 (`best_int8.onnx`)** | **~3.7 MB** | **1–2% below baseline** | **Yes, ~5.5 FPS** |

### Inference Speed on Raspberry Pi 4

| Setup | Resolution | Raw Inference | End-to-End | Frames/sec |
|---|---|---|---|---|
| ONNX FP32 | 416x416 | ~800 ms | ~900 ms | ~1.1 |
| ONNX FP32 | 320x320 | ~250 ms | ~300 ms | ~3.3 |
| **ONNX INT8** | **320x320** | **~120 ms** | **~180 ms** | **~5.5** |
| ONNX INT8 with frame skipping | 320x320 | ~120 ms | ~180 ms | 10+ effective |

**Where the time goes (INT8, 320x320 input):**

- Pre-processing (resize, normalise, transpose): 30–50 ms
- Neural network inference: 120–200 ms
- Non-maximum suppression: 10–20 ms
- Drawing boxes and writing logs: 5–10 ms

At around 5–6 FPS, the system picks up potholes and obstacles comfortably at city driving speeds of 30–50 km/h.

---

## Optimisation Strategies We Used

| What | When | Why |
|---|---|---|
| INT8 post-training quantization | After export | Cuts model size by ~3.5x and speeds up inference 2–4x |
| Automatic mixed precision (AMP) | During training | Nearly halves GPU memory use, speeds up training ~1.5–2x |
| Transfer learning from COCO | Training start | The backbone already understands edges, textures, shapes — converges much faster |
| Backbone freezing | Fine-tuning | Prevents the pretrained features from being overwritten when data is limited |
| Automatic anchor tuning | Before training | Fits anchor boxes to the specific shapes of potholes and road damage |
| Exponential moving average (EMA) | Training | Produces smoother, more stable final weights |
| Lower inference resolution | Deployment | 320x320 instead of 416x416 cuts computation by ~4x |
| Multi-threaded ONNX Runtime | Deployment | Spreads work across all four Cortex-A72 cores |
| ONNX graph optimisation | Deployment | Fuses operations and folds constants for fewer total computations |

---

## Where This Could Go Next

There are several natural extensions we see for this project:

- **Google Coral USB Accelerator** — plugging in an Edge TPU could push inference speed up by roughly 10x
- **GPS tagging** — pair detections with GPS coordinates to build a heatmap of road damage on something like OpenStreetMap
- **Cloud upload** — sync logs and saved clips over WiFi or 4G for centralised fleet monitoring
- **Night mode** — integrate an infrared-capable camera for better detection after dark
- **Severity grading** — split anomalies into sub-categories like minor, moderate, and severe
- **Raspberry Pi 5** — the newer hardware with its faster 2.4 GHz cores should improve FPS noticeably
- **Model distillation** — compress YOLOv5s further into a YOLOv5n-sized student model
- **Vehicle-to-vehicle alerts** — broadcast detected hazards to nearby cars through V2X communication protocols

---

## References and Acknowledgements

This project builds on top of open-source work by many researchers and engineers. A few key ones:

1. Jocher, G. et al. — *YOLOv5*, Ultralytics (2020). [GitHub](https://github.com/ultralytics/yolov5)
2. Wang, C.-Y. et al. — *CSPNet: A New Backbone that can Enhance Learning Capability of CNN*, CVPR Workshops (2020)
3. Liu, S. et al. — *Path Aggregation Network for Instance Segmentation*, CVPR (2018)
4. Maeda, H. et al. — *Road Damage Detection Using Deep Neural Networks with Smartphone Images*, CACAIE (2018)
5. Arya, D. et al. — *Deep Learning-based Road Damage Detection and Classification*, Automation in Construction (2021)
6. Zheng, Z. et al. — *Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression*, AAAI (2020)
7. Micikevicius, P. et al. — *Mixed Precision Training*, ICLR (2018)
8. ONNX Runtime documentation — [Post-Training Quantization](https://onnxruntime.ai/docs/performance/quantization.html)
9. Raspberry Pi Foundation — [Pi 4 Model B Specifications](https://www.raspberrypi.com/products/raspberry-pi-4-model-b/)

---

## License

Released under the **GNU Affero General Public License v3.0 (AGPL-3.0)**. See the [LICENSE](LICENSE) file for the full text.

Built on top of [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5) by Glenn Jocher and contributors.

---

<div align="center">
<i>Developed for edge-deployed road safety — Bharat AI SoC, 2026</i>
</div>
