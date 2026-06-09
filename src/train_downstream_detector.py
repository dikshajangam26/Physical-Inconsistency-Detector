import os
import csv
import random

import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from intervention_engine import generate_counterfactual_variation

# =========================================================
# Paths
# =========================================================
DATA_ROOT = "data/multiview/raw"
CORRUPTED_LIST = "data/multiview/corrupted_list.txt"
SPATIAL_RESULTS_CSV = "results/multiview/spatial_evaluation_results.csv"

RESULTS_DIR = "results"
REPORT_CSV = os.path.join(RESULTS_DIR, "downstream_ablation_results.csv")
VALUE_OF_LIFT_CHART = os.path.join(RESULTS_DIR, "downstream_value_of_lift.png")

# =========================================================
# Settings
# =========================================================
SEED = 42
IMG_SIZE = 64
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-3
VAL_FRACTION = 0.20
TOP_K_PERCENT = 0.20

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# =========================================================
# Utilities
# =========================================================
def ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def load_corrupted_videos(path):
    if not os.path.exists(path):
        return set()

    with open(path, "r") as f:
        return set(line.strip() for line in f if line.strip())


def load_uncertainty_scores(csv_path):
    scores = {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scores[row["video_name"]] = float(row["prob_failure"])
    return scores


def list_all_videos(root):
    return sorted([v for v in os.listdir(root) if os.path.isdir(os.path.join(root, v))])


def read_rgb(path):
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def load_overhead_last_frame(video_name):
    path = os.path.join(DATA_ROOT, video_name, "overhead", "frame_9.png")
    return read_rgb(path)


def find_box_bbox(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    ys, xs = np.where(gray > 10)

    if len(xs) == 0 or len(ys) == 0:
        return [0.0, 0.0, 1.0, 1.0]

    x1, x2 = float(xs.min()), float(xs.max())
    y1, y2 = float(ys.min()), float(ys.max())

    return [x1, y1, x2, y2]


def bbox_to_normalized_xywh(bbox, img_size=64):
    x1, y1, x2, y2 = bbox

    cx = ((x1 + x2) / 2.0) / img_size
    cy = ((y1 + y2) / 2.0) / img_size
    w = max(1.0, x2 - x1) / img_size
    h = max(1.0, y2 - y1) / img_size

    return np.array([cx, cy, w, h], dtype=np.float32)


def normalized_xywh_to_bbox(vec, img_size=64):
    cx, cy, w, h = vec

    cx *= img_size
    cy *= img_size
    w *= img_size
    h *= img_size

    x1 = max(0.0, cx - w / 2.0)
    y1 = max(0.0, cy - h / 2.0)
    x2 = min(float(img_size - 1), cx + w / 2.0)
    y2 = min(float(img_size - 1), cy + h / 2.0)

    return [x1, y1, x2, y2]


def bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    x_left = max(ax1, bx1)
    y_top = max(ay1, by1)
    x_right = min(ax2, bx2)
    y_bottom = min(ay2, by2)

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    inter = (x_right - x_left) * (y_bottom - y_top)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter
    if union <= 0:
        return 0.0

    return float(inter / union)


# =========================================================
# Hard validation augmentations
# =========================================================
def apply_heavy_shadow(image_rgb):
    out = image_rgb.copy().astype(np.float32)
    h, w = out.shape[:2]
    out[:, : w // 2] *= 0.35
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_motion_blur(image_rgb, ksize=11):
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    kernel[ksize // 2, :] = 1.0 / ksize
    return cv2.filter2D(image_rgb, -1, kernel)


def apply_extreme_angle(image_rgb):
    h, w = image_rgb.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, 18, 1.0)
    return cv2.warpAffine(
        image_rgb,
        M,
        (w, h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def build_hard_validation_image(image_rgb, idx):
    mode = idx % 3

    if mode == 0:
        return apply_heavy_shadow(image_rgb)
    elif mode == 1:
        return apply_motion_blur(image_rgb)
    else:
        return apply_extreme_angle(image_rgb)


# =========================================================
# Data split logic
# =========================================================
def build_base_clean_split():
    corrupted = load_corrupted_videos(CORRUPTED_LIST)
    all_videos = list_all_videos(DATA_ROOT)

    clean_videos = [v for v in all_videos if v not in corrupted]

    n_val = max(1, int(len(clean_videos) * VAL_FRACTION))
    val_videos = clean_videos[-n_val:]
    train_clean_videos = clean_videos[:-n_val]

    return train_clean_videos, val_videos, corrupted


def make_record(video_name, image_rgb, split_name, source_name):
    bbox = find_box_bbox(image_rgb)
    target = bbox_to_normalized_xywh(bbox)

    return {
        "video_name": video_name,
        "image": image_rgb,
        "target": target,
        "split": split_name,
        "source": source_name,
    }


def build_training_config_records(config_name):
    train_clean_videos, val_videos, corrupted = build_base_clean_split()
    uncertainty = load_uncertainty_scores(SPATIAL_RESULTS_CSV)

    # training pool used for Lift@k ranking
    train_pool = train_clean_videos + sorted(list(corrupted))
    train_pool = sorted(set(train_pool))

    ranked = sorted(train_pool, key=lambda v: uncertainty.get(v, 0.0), reverse=True)
    top_k_count = max(1, int(len(ranked) * TOP_K_PERCENT))
    top_k_videos = set(ranked[:top_k_count])

    # -------------------------------
    # Base clean data
    # -------------------------------
    train_records = []
    for video_name in train_clean_videos:
        img = load_overhead_last_frame(video_name)
        train_records.append(make_record(video_name, img, "train", "clean_base"))

    # -------------------------------
    # Four configurations
    # -------------------------------
    if config_name == "clean_base":
        pass

    elif config_name == "noisy_baseline":
        # add raw corrupted/unfiltered tracks directly
        for video_name in sorted(list(corrupted)):
            img = load_overhead_last_frame(video_name)
            train_records.append(make_record(video_name, img, "train", "raw_corrupted"))

    elif config_name == "naive_filtering":
        # drop top 20% highest uncertainty
        train_records = [r for r in train_records if r["video_name"] not in top_k_videos]

    elif config_name == "counterfactual_synthetics":
        # add corrected synthetic versions of the exact same top 20% uncertain videos
        for video_name in sorted(top_k_videos):
            img = load_overhead_last_frame(video_name)

            # choose simple failure mode for intervention
            if video_name in corrupted:
                failure_mode = "spatial_residual_anomaly"
            else:
                failure_mode = "temporal_sync_drift"

            repaired = generate_counterfactual_variation(img, failure_mode)
            train_records.append(
                make_record(video_name, repaired, "train", "counterfactual_synthetic")
            )

    else:
        raise ValueError(f"Unknown config: {config_name}")

    # -------------------------------
    # Hard validation set
    # -------------------------------
    val_records = []
    for idx, video_name in enumerate(val_videos):
        img = load_overhead_last_frame(video_name)
        hard_img = build_hard_validation_image(img, idx)
        val_records.append(make_record(video_name, hard_img, "val", "hard_validation"))

    return train_records, val_records, top_k_videos


# =========================================================
# Dataset
# =========================================================
class BoxDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        img = r["image"].astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        target = r["target"]

        return (
            torch.tensor(img, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32),
        )


# =========================================================
# Tiny downstream box regressor
# =========================================================
class TinyBoxRegressor(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        return x


def train_model(train_records):
    dataset = BoxDataset(train_records)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = TinyBoxRegressor()
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for images, targets in loader:
            optimizer.zero_grad()

            preds = model(images)
            loss = criterion(preds, targets)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / max(1, len(loader))
        print(f"Epoch {epoch + 1}/{EPOCHS} - train loss: {avg_loss:.4f}")

    return model


def evaluate_model(model, val_records):
    dataset = BoxDataset(val_records)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model.eval()

    ious = []
    acc50 = []

    with torch.no_grad():
        for images, targets in loader:
            pred = model(images).squeeze(0).cpu().numpy()
            target = targets.squeeze(0).cpu().numpy()

            pred_bbox = normalized_xywh_to_bbox(pred, IMG_SIZE)
            true_bbox = normalized_xywh_to_bbox(target, IMG_SIZE)

            iou = bbox_iou(pred_bbox, true_bbox)

            ious.append(iou)
            acc50.append(1 if iou >= 0.50 else 0)

    mean_iou = float(np.mean(ious)) if ious else 0.0
    bbox_acc50 = float(np.mean(acc50)) if acc50 else 0.0

    return {
        "mean_iou": mean_iou,
        "bbox_acc50": bbox_acc50,
    }


# =========================================================
# Plotting / reporting
# =========================================================
def save_report(results_dict):
    ensure_results_dir()

    with open(REPORT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["config", "bbox_acc50", "mean_iou"])

        for config_name, metrics in results_dict.items():
            writer.writerow([config_name, metrics["bbox_acc50"], metrics["mean_iou"]])

    print(f"Saved report to {REPORT_CSV} ✅")


def plot_value_of_lift(results_dict):
    labels = [
        "Clean Base Data Only",
        "Base + Corrupted/Unfiltered",
        "Base + Naive Filtering",
        "Base + Counterfactual Synthetics",
    ]

    keys = [
        "clean_base",
        "noisy_baseline",
        "naive_filtering",
        "counterfactual_synthetics",
    ]

    values = [results_dict[k]["bbox_acc50"] for k in keys]

    plt.figure(figsize=(10, 5))
    plt.bar(labels, values)
    plt.ylabel("BBox Accuracy @ IoU >= 0.50")
    plt.title("Value of Lift: Downstream Detector Performance")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(VALUE_OF_LIFT_CHART)

    print(f"Saved chart to {VALUE_OF_LIFT_CHART} ✅")


# =========================================================
# Main
# =========================================================
def main():
    ensure_results_dir()

    configs = [
        "clean_base",
        "noisy_baseline",
        "naive_filtering",
        "counterfactual_synthetics",
    ]

    all_results = {}

    for config_name in configs:
        print("=" * 70)
        print(f"Running config: {config_name}")

        train_records, val_records, top_k_videos = build_training_config_records(config_name)

        print(f"Train records: {len(train_records)}")
        print(f"Validation records: {len(val_records)}")
        print(
            f"Top {int(TOP_K_PERCENT * 100)}% uncertainty videos in Lift@k policy: "
            f"{len(top_k_videos)}"
        )

        model = train_model(train_records)
        metrics = evaluate_model(model, val_records)

        all_results[config_name] = metrics

        print(
            f"{config_name} -> "
            f"bbox_acc50={metrics['bbox_acc50']:.4f}, "
            f"mean_iou={metrics['mean_iou']:.4f}"
        )

    save_report(all_results)
    plot_value_of_lift(all_results)

    print("\nFinal comparison:")
    for config_name, metrics in all_results.items():
        print(
            f"{config_name}: "
            f"bbox_acc50={metrics['bbox_acc50']:.4f}, "
            f"mean_iou={metrics['mean_iou']:.4f}"
        )


if __name__ == "__main__":
    main()
