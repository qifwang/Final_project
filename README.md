# Survue Object Detection — Final Project (CS5330)

This project implements an end-to-end object detection pipeline for the **Survue** bicycle-safety system.

Survue collects real-world road images from a vehicle-mounted camera to detect:

- **human** (pedestrians / cyclists)  
- **vehicle** (cars, trucks, etc.)  
- **trafficsign** (traffic signs and signals)

The hardware is bike-mounted and has **limited compute**, so the goal is to design a model that is:

- **Lightweight** (small file size, low parameter count)  
- **Fast** (near real-time inference)  
- **Accurate** (good mAP on the provided validation set)

We use **YOLOv8n (nano)** as the backbone detector and build a full pipeline from:

1. COCO-style annotations → YOLO format  
2. Training on the Survue dataset  
3. Evaluation and visualization for the final report & presentation  

---

## 📁 Project Structure (analysis-ready files marked)

**Symbols:**

- ⭐ **Final model / key artifact**
- 📊 **Useful for analysis / numeric results**
- 🖼 **Useful for figures in report / slides**

```text
Final_project/
├── data/
│   └── survue.yaml
│       # YOLOv8 dataset config (uses absolute paths to train/val images)
│
├── datasets/
│   ├── annotations/                # Original Survue annotations (COCO format)
│   │   ├── instances.json
│   │   ├── instances_train.json
│   │   └── instances_val.json
│   │
│   ├── images/                     # Raw images
│   │   ├── train/                  # 393 training images (input)
│   │   └── val/                    # 107 validation images (input)
│   │
│   └── labels/                     # YOLO-format labels (generated)
│       ├── train/                  # .txt labels for each train image
│       └── val/                    # .txt labels for each val image
│
├── notebooks/
│   ├── 01_convert_coco_to_yolo.ipynb   # Convert COCO annotations → YOLO labels
│   ├── 02_train_yolov8n.ipynb          # Train YOLOv8n on the Survue dataset
│   └── 03_eval_and_viz.ipynb           # Evaluate model + visualize predictions
│
│   └── runs/
│       └── detect/                     # YOLOv8 auto-generated training runs
│           ├── survue_yolov8n/         # Early experiment (not used)
│           ├── survue_yolov8n2/        # Early experiment (not used)
│           └── survue_yolov8n3/        # ✅ Final training run (50 epochs)
│               ├── weights/
│               │   ├── best.pt     ⭐   # Final selected model checkpoint
│               │   └── last.pt         # Weights after the last epoch
│               ├── results.png     📊🖼 # Loss & mAP curves across epochs
│               ├── results.csv     📊  # Numeric per-epoch logs (mAP, P/R, losses)
│               ├── labels.jpg      📊🖼 # Class distribution visualization
│               └── other logs          # Additional training outputs
│
├── outputs/
│   └── viz_val/
│       └── examples/
│           ├── *.jpg              🖼   # Predicted val images with bounding boxes
│
├── src/
│   ├── convert_coco_to_yolo.py        # Helper for COCO → YOLO conversion
│   └── utils.py                       # (Optional) utility functions
│
└── README.md                          # This file
```

---

## 🔧 Environment & Dependencies

The project was developed and tested in a JupyterLab / Anaconda environment with:

- Python 3.13
- `ultralytics` (YOLOv8)
- `torch` (CPU version on Apple Silicon)
- Standard scientific stack: `numpy`, `matplotlib`, etc.

**Install YOLOv8:**

```bash
pip install ultralytics
```

On the course cluster, follow the instructor's environment setup instructions and then install `ultralytics` inside the active environment.

---

## 🚀 How to Run the Pipeline

### 1️⃣ Data Layout (already prepared)

The dataset is expected under `datasets/` as:

```text
datasets/
├── annotations/
│   ├── instances_train.json
│   └── instances_val.json
├── images/
│   ├── train/
│   └── val/
└── labels/          # Created in step 2
    ├── train/
    └── val/
```

Nothing to run here if you already copied the provided Survue data.

---

### 2️⃣ Notebook 01 — COCO → YOLO Conversion

**File:** `notebooks/01_convert_coco_to_yolo.ipynb`

This notebook:

- Loads COCO-style annotations:
  - `datasets/annotations/instances_train.json`
  - `datasets/annotations/instances_val.json`
- Converts each COCO bounding box `[x, y, w, h]` into YOLO format:
  - `(class_id, x_center, y_center, width, height)` normalized to `[0, 1]`
- Writes `.txt` label files into:
  ```text
  datasets/labels/train/*.txt
  datasets/labels/val/*.txt
  ```

**Key function used:**

```python
from convert_coco_to_yolo import convert_split

convert_split(train_json, DATASET_ROOT, split="train")
convert_split(val_json, DATASET_ROOT, split="val")
```

After running this notebook, the dataset is ready for YOLOv8 training.

---

### 3️⃣ Notebook 02 — Train YOLOv8n

**File:** `notebooks/02_train_yolov8n.ipynb`

This notebook trains the YOLOv8n (nano) model on the Survue dataset.

#### 3.1 Dataset config (`data/survue.yaml`)

`survue.yaml` uses absolute paths to avoid path confusion:

```yaml
train: /Users/your_username/Desktop/5330/Final_project/datasets/images/train
val: /Users/your_username/Desktop/5330/Final_project/datasets/images/val

names:
  0: human
  1: trafficsign
  2: vehicle
```

On the cluster, update the paths accordingly (e.g., `/courses/CS5330...`).

#### 3.2 Training code (core cell)

```python
from ultralytics import YOLO
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
DATA_YAML = os.path.join(PROJECT_ROOT, "data", "survue.yaml")

model = YOLO("yolov8n.pt")  # load YOLOv8n base model

results = model.train(
    data=DATA_YAML,
    epochs=50,
    imgsz=640,
    batch=16,
    device="cpu",              # or '0' for GPU if available
    name="survue_yolov8n3"     # training run name
)
```

This produces:

```text
notebooks/runs/detect/survue_yolov8n3/
├── weights/
│   ├── best.pt     ⭐
│   └── last.pt
├── results.png     📊🖼
├── results.csv     📊
├── labels.jpg      📊🖼
└── ...
```

These files are central for the final analysis.

---

### 4️⃣ Notebook 03 — Evaluation & Visualization

**File:** `notebooks/03_eval_and_viz.ipynb`

This notebook:

- Loads the final model (`best.pt`)
- Runs evaluation on the Survue validation set
- Saves visualizations of predictions on validation images

#### 4.1 Load the trained model

```python
from ultralytics import YOLO
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
DATA_YAML = os.path.join(PROJECT_ROOT, "data", "survue.yaml")

RUN_DIR = os.path.join(PROJECT_ROOT, "notebooks", "runs", "detect", "survue_yolov8n3")
WEIGHTS_BEST = os.path.join(RUN_DIR, "weights", "best.pt")

model = YOLO(WEIGHTS_BEST)
```

#### 4.2 Evaluate on validation set

```python
metrics = model.val(data=DATA_YAML, device="cpu")

print("mAP50-95:", metrics.box.map)
print("mAP50:", metrics.box.map50)
print("mAP75:", metrics.box.map75)

for i, class_name in enumerate(metrics.names.values()):
    print(f"{class_name:12s} | mAP50 = {metrics.box.maps[i]:.3f}")
```

These numbers are used in the **Results** section of the report.

#### 4.3 Visualize predictions

```python
VAL_IMAGES_DIR = os.path.join(PROJECT_ROOT, "datasets", "images", "val")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "viz_val")

results = model.predict(
    source=VAL_IMAGES_DIR,
    save=True,
    project=OUTPUT_DIR,
    name="examples",
    imgsz=640,
    device="cpu",
)
```

Prediction images are saved to:

```text
outputs/viz_val/examples/*.jpg  🖼
```

These are ideal for inclusion in the report and presentation (success & failure cases).

---

## 📊 Final Model Performance (Summary)

From evaluation using `best.pt`:

### Overall (all classes)
- **Precision (P)** ≈ 0.695
- **Recall (R)** ≈ 0.459
- **mAP50** ≈ 0.51
- **mAP50–95** ≈ 0.308

### Per-class mAP50
- `human` ≈ 0.56
- `vehicle` ≈ 0.576
- `trafficsign` ≈ 0.395

### Model characteristics
- **Parameters:** ~ 3.0M
- **FLOPs:** ~ 8.1 GFLOPs
- **Checkpoint size:** ~ 6.2 MB ⭐
- **Inference speed:** ~ 82 ms / image (CPU, Apple M4)

These results satisfy the requirements for a **lightweight**, **reasonably accurate**, and **real-time-capable** object detector for Survue's bike-mounted hardware.

---

## 📝 Report & Presentation

### Final Report Structure

1. **Introduction** — What you did and why
2. **Background** — Object detection methods overview
3. **Methods/Analysis** — Model architecture, training process, evaluation metrics
4. **Results** — Performance metrics, visualizations
5. **Discussion** — Interpretation of results
6. **Limitations** — Challenges encountered
7. **Future Work** — Potential improvements

### Final Presentation

- 10–15 minute recorded video
- All team members must present
- Summarize key points from the report
- Include visualizations from `outputs/viz_val/examples/`

---

## 📚 References

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [COCO Dataset Format](https://cocodataset.org/#format-data)
- Course materials and toy detection code
