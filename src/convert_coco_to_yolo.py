"""
convert_coco_to_yolo.py

Convert COCO-format annotations (instances_train/val.json) into
YOLO-format labels and organize images into train/val folders.

Expected folder layout before running this script:

survue_yolov8/
├── datasets/
│   ├── images/           # contains 500 raw .jpg images
│   └── annotations/
│       ├── instances_train.json
│       └── instances_val.json
└── src/
    ├── convert_coco_to_yolo.py
    └── utils.py

After running this script, we will have:

survue_yolov8/
├── datasets/
│   ├── images/
│   │   ├── train/        # training images
│   │   └── val/          # validation images
│   ├── labels/
│   │   ├── train/        # YOLO txt labels for training
│   │   └── val/          # YOLO txt labels for validation
│   └── annotations/
│       ├── instances_train.json
│       └── instances_val.json
"""

import json
import os
import shutil
from collections import defaultdict
from typing import Dict, Any

from tqdm import tqdm

from utils import ensure_dir, coco_bbox_to_yolo, clamp01


def build_category_mapping(categories):
    """
    Build a mapping from COCO category_id to contiguous 0-based YOLO class index.

    Args:
        categories: List of category dicts from COCO JSON.

    Returns:
        Dictionary mapping {coco_category_id: yolo_class_index}.
    """
    # Sort by category id to keep a stable ordering
    categories_sorted = sorted(categories, key=lambda c: c["id"])
    cat_id_to_idx = {cat["id"]: idx for idx, cat in enumerate(categories_sorted)}
    return cat_id_to_idx


def group_annotations_by_image(annotations):
    """
    Group annotations by image_id for faster lookup.

    Args:
        annotations: List of annotation dicts from COCO JSON.

    Returns:
        Dict[image_id, List[annotation_dict]]
    """
    grouped = defaultdict(list)
    for ann in annotations:
        image_id = ann["image_id"]
        grouped[image_id].append(ann)
    return grouped


def convert_split(
    json_path: str,
    dataset_root: str,
    split: str = "train",
    raw_images_subdir: str = "images",
) -> None:
    """
    Convert one split (train or val) from COCO to YOLO format.

    This function:
        - Reads the COCO JSON file.
        - Copies the corresponding images into datasets/images/{split}/
        - Writes YOLO labels into datasets/labels/{split}/

    Args:
        json_path: Path to the instances_train.json or instances_val.json file.
        dataset_root: Path to the 'datasets' directory.
        split: 'train' or 'val'.
        raw_images_subdir: Subdirectory under dataset_root where the original
                           flat list of images is stored, default 'images'.
    """
    print(f"[INFO] Converting split='{split}' from COCO to YOLO...")
    print(f"[INFO] JSON: {json_path}")
    print(f"[INFO] dataset_root: {dataset_root}")

    # Directories
    raw_images_dir = os.path.join(dataset_root, raw_images_subdir)
    split_images_dir = os.path.join(dataset_root, "images", split)
    split_labels_dir = os.path.join(dataset_root, "labels", split)

    ensure_dir(split_images_dir)
    ensure_dir(split_labels_dir)

    # Load COCO JSON
    with open(json_path, "r") as f:
        coco: Dict[str, Any] = json.load(f)

    images = coco["images"]
    annotations = coco["annotations"]
    categories = coco["categories"]

    # Build helper mappings
    cat_id_to_idx = build_category_mapping(categories)
    anns_by_image = group_annotations_by_image(annotations)
    img_id_to_info = {img["id"]: img for img in images}

    for img_id, img_info in tqdm(img_id_to_info.items(), desc=f"Processing {split} images"):
        file_name = img_info["file_name"]
        width = img_info["width"]
        height = img_info["height"]

        # Source & destination image paths
        src_img_path = os.path.join(raw_images_dir, file_name)
        dst_img_path = os.path.join(split_images_dir, file_name)

        if not os.path.exists(src_img_path):
            raise FileNotFoundError(f"Image not found: {src_img_path}")

        # Copy image only if not already present
        if not os.path.exists(dst_img_path):
            shutil.copy2(src_img_path, dst_img_path)

        # Prepare label file
        label_lines = []
        for ann in anns_by_image.get(img_id, []):
            bbox = ann["bbox"]  # COCO format [x, y, w, h]
            cat_id = ann["category_id"]

            # Skip degenerate boxes
            if bbox[2] <= 0 or bbox[3] <= 0:
                continue

            x_center, y_center, w, h = coco_bbox_to_yolo(bbox, width, height)

            # Clamp to [0, 1] just in case
            x_center = clamp01(x_center)
            y_center = clamp01(y_center)
            w = clamp01(w)
            h = clamp01(h)

            if w <= 0 or h <= 0:
                continue

            class_idx = cat_id_to_idx[cat_id]
            label_lines.append(
                f"{class_idx} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
            )

        # Write YOLO label file
        label_filename = os.path.splitext(file_name)[0] + ".txt"
        label_path = os.path.join(split_labels_dir, label_filename)

        with open(label_path, "w") as f:
            f.write("\n".join(label_lines))

    print(f"[INFO] Done converting split='{split}'.")
    print(f"[INFO] Images dir: {split_images_dir}")
    print(f"[INFO] Labels dir: {split_labels_dir}")


if __name__ == "__main__":
    """
    Example usage (run this from the 'survue_yolov8/' root directory):

        python -m src.convert_coco_to_yolo

    Make sure that:
        datasets/annotations/instances_train.json
        datasets/annotations/instances_val.json
        datasets/images/  (contains 500 raw images)

    already exist.
    """
    # Adjust this path if necessary. Here we assume the script is run
    # from the 'survue_yolov8/' root directory.
    DATASET_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets")
    ANN_DIR = os.path.join(DATASET_ROOT, "annotations")

    train_json = os.path.join(ANN_DIR, "instances_train.json")
    val_json = os.path.join(ANN_DIR, "instances_val.json")

    convert_split(train_json, DATASET_ROOT, split="train")
    convert_split(val_json, DATASET_ROOT, split="val")

    print("[INFO] COCO -> YOLO conversion finished for both train and val splits.")
