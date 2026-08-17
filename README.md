# Railway Crack Detection

YOLOv8-based detection of rail surface defects (cracks and gaps) from photographs of railway track, with a CLI tool for single-image inference, a training pipeline, and **TRACKSCAN** — a Flask web app for scanning images through the browser.

> Dataset sourced from Roboflow Universe: **[railway-crack-detection-ezd8g-hojvc](https://universe.roboflow.com/23f3001829s-workspace/railway-crack-detection-ezd8g-hojvc)**

<!-- Optional: drop a screenshot or short GIF of TRACKSCAN here, e.g. docs/trackscan-preview.png -->

---

## Features

- **YOLOv8 detection** — single-class rail defect detector (`railway-gap`), trained with [Ultralytics](https://github.com/ultralytics/ultralytics)
- **CLI inference** (`detect.py`) — run detection on a single image from the command line, with full threshold/size/device control
- **Training pipeline** (`trainer.py`) — argparse-driven YOLOv8 training with configurable augmentation, precision, and run tracking
- **TRACKSCAN web app** (`app.py` + `templates/` + `static/`) — drag-and-drop or camera-capture image upload, adjustable scan settings, and live annotated results, usable on both desktop and mobile

## Tech stack

| Layer | Tools |
|---|---|
| Model | Ultralytics YOLOv8, PyTorch |
| Image I/O | OpenCV, Pillow |
| Backend | Flask |
| Frontend | Vanilla HTML / CSS / JS (no framework) |

---
## STEPS

->befoure starting the traning put the data in dataset folder in the yolov8 format with matching labels of of image in two seprate folder for each 
->create a runs folder here all the weights and results will be stored 
->use "trainer.py --help" to get to know about the args
->use the detect to detect the gap locally or use the app.py file for the online use by enabling it on your local machine u could access the website on the dedicated link 

---

## Project structure

```
Railway-Crack-Detection/
├── app.py                   # Flask backend — serves the web app and /api/predict
├── detect.py                # CLI single-image detection
├── trainer.py                # YOLOv8 training script
├── dataset_analyzer.py       # dataset validation (referenced by trainer.py --check)
├── requirements.txt
├── pretrained weights/
│   └── best.pt                # trained model weights
├── dataset/
│   └── data.yaml              # class list + train/val/test paths
├── runs/                     # raw Ultralytics training run outputs (per --project)
├── results/                  # promoted metrics & sample predictions — see below
├── templates/
│   └── index.html
└── static/
    ├── css/style.css
    ├── js/script.js
    ├── uploads/                # runtime: images uploaded through the web app
    └── results/                # runtime: annotated images from web app scans
```

> `results/` (training showcase artifacts) and `static/results/` (runtime scan outputs written by `app.py`) are deliberately separate folders, even though the names are similar — this keeps every web scan from cluttering the training metrics folder shown below.

---

## Setup

```bash
git clone <this-repo>
cd Railway-Crack-Detection
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Place your trained weights at the path `detect.py` expects (or pass `--weights` explicitly):

```
pretrained weights/best.pt
```

---

## Usage

### Web app (TRACKSCAN)

```bash
python app.py
```

Then open **http://127.0.0.1:5000/** in a browser. Upload a rail image (drag-and-drop, tap to select, or capture directly on mobile), optionally adjust confidence/IoU/image size/device in the settings panel, and hit **Scan for defects**. Results — annotated image, defect count, per-detection confidence and bounding box — render in place.

> Load the app through this URL, not through a separate static-file server (e.g. VS Code Live Server). A different server on a different port can't reach `/api/predict` and will return `405 Method Not Allowed`.

### CLI detection

```bash
python detect.py --input path/to/image.jpg
```

| Flag | Default | Description |
|---|---|---|
| `--input` | *(required)* | Path to the input image |
| `--weights` | `pretrained weights/best.pt` | Path to trained `.pt` weights |
| `--output` | `results/<name>_detected.<ext>` | Output image path |
| `--conf` | `0.10` | Confidence threshold |
| `--iou` | `0.45` | IoU threshold for NMS |
| `--imgsz` | `960` | Inference image size |
| `--max-det` | `100` | Max detections per image |
| `--device` | auto | `0` for GPU, `cpu`, or omit |
| `--show` | off | Display the annotated result in an OpenCV window |

### Training

```bash
python trainer.py --data-yaml dataset/data.yaml --model yolov8s.pt --epochs 100 --batch-size 4
```

Key flags: `--image-size`, `--epochs`, `--patience`, `--batch-size`, `--device`, `--workers` (defaults to `0` for Windows `DataLoader` stability), `--seed`, `--rect`, `--half`, plus augmentation controls (`--degrees`, `--translate`, `--scale`, `--hsv-h/s/v`, `--fliplr`, `--flipud`, `--mosaic`, `--mixup`). Run `python trainer.py --help` for the full list.

Before training starts, `trainer.py` runs a preflight check against `--data-yaml` and points you to `dataset_analyzer.py --check` for deeper dataset validation if that check fails. After training, it validates and (unless `--skip-test`) tests the best checkpoint, then writes `experiment_summary.json` alongside the run.

---

## API reference

```
POST /api/predict
Content-Type: multipart/form-data
```

| Field | Required | Default | Notes |
|---|---|---|---|
| `file` | yes | — | `.jpg` / `.jpeg` / `.png` / `.webp` |
| `conf` | no | `0.10` | 0–1 |
| `iou` | no | `0.45` | 0–1 |
| `imgsz` | no | `960` | |
| `max_det` | no | `100` | |
| `device` | no | auto | `cpu` or `0` |

**Response — 200:**

```json
{
  "success": true,
  "image_filename": "rail1_20260815_142210_123.jpg",
  "result_image": "rail1_20260815_142210_456.jpg",
  "result_image_url": "/results/rail1_20260815_142210_456.jpg",
  "total_detections": 1,
  "detections": [
    {
      "class_id": 0,
      "class_name": "railway-gap",
      "confidence": 0.62,
      "bbox": { "x1": 1814, "y1": 964, "x2": 2466, "y2": 1660 }
    }
  ],
  "parameters": { "confidence": 0.10, "iou": 0.45, "imgsz": 960, "max_det": 100 },
  "timestamp": "2026-08-15T14:22:10.456000"
}
```

**Response — error** (400 / 404 / 413 / 500): `{ "error": "..." }`

Other routes: `GET /` (serves the app), `GET /health`, `GET /api/config` (model classes + defaults), `GET /results/<filename>` (annotated images).

---

## Results

Training artifacts live in [`results/`](results/). Exact numbers are in [`plot.csv`](results/plot.csv) and [`plot.png`](results/plot.png) — not restated here to avoid going stale.

**Training curves & confusion matrix**

<table>
<tr>
<td><img src="results/plot.png" width="400"/></td>
<td><img src="results/confusion_matrix_normalized.png" width="400"/></td>
</tr>
<tr>
<td align="center">Loss / precision / recall / mAP over training</td>
<td align="center">Confusion matrix (normalized)</td>
</tr>
</table>

<table>
<tr>
<td><img src="results/BoxPR_curve.png" width="300"/></td>
<td><img src="results/BoxF1_curve.png" width="300"/></td>
<td><img src="results/labels.jpg" width="300"/></td>
</tr>
<tr>
<td align="center">Precision–Recall curve</td>
<td align="center">F1 curve</td>
<td align="center">Label distribution</td>
</tr>
</table>

Also available: [`BoxP_curve.png`](results/BoxP_curve.png), [`BoxR_curve.png`](results/BoxR_curve.png), [`confusion_matrix.png`](results/confusion_matrix.png).

**Sample predictions vs. ground truth**

<table>
<tr>
<td><img src="results/val_batch0_labels.jpg" width="400"/></td>
<td><img src="results/val_batch0_pred.jpg" width="400"/></td>
</tr>
<tr>
<td align="center">Ground truth</td>
<td align="center">Model prediction</td>
</tr>
</table>

More validation batches: [`val_batch1_labels.jpg`](results/val_batch1_labels.jpg) / [`val_batch1_pred.jpg`](results/val_batch1_pred.jpg), [`val_batch2_labels.jpg`](results/val_batch2_labels.jpg) / [`val_batch2_pred.jpg`](results/val_batch2_pred.jpg).

**Training batch mosaics:** [`train_batch0.jpg`](results/train_batch0.jpg), [`train_batch1.jpg`](results/train_batch1.jpg), [`train_batch2.jpg`](results/train_batch2.jpg), [`train_batch29120.jpg`](results/train_batch29120.jpg), [`train_batch29121.jpg`](results/train_batch29121.jpg), [`train_batch29122.jpg`](results/train_batch29122.jpg)

---

## Troubleshooting

- **`OMP: Error #15` / process dies mid-request** — PyTorch and OpenCV can each bundle Intel's OpenMP runtime; loading both in one process crashes at the native level. `detect.py` sets `KMP_DUPLICATE_LIB_OK=TRUE` before importing `ultralytics` to work around this — keep that line if you refactor imports.
- **`405 Method Not Allowed` on `/api/predict`** — the page is being served by something other than `app.py` (commonly a separate static-file server like Live Server on a different port). Load the app from `http://127.0.0.1:5000/`, started via `python app.py`.
- **`imgsz=960` and inference speed** — both the CLI and the web app default to `960` for consistency with each other. If your model was trained at a smaller resolution (check `--image-size` in your training run), a smaller `--imgsz` / settings-panel value will run faster with no accuracy cost.

---

## Acknowledgments

- Dataset: [Railway Crack Detection](https://universe.roboflow.com/23f3001829s-workspace/railway-crack-detection-ezd8g-hojvc) on Roboflow Universe
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)

## Description

Training was performed on a local machine equipped with an **11th-generation Intel Core i7 processor, NVIDIA RTX 3050 GPU, and 32 GB RAM**. The model was trained for **150 epochs**, and the complete training process took approximately **3.5 hours** on this system.
