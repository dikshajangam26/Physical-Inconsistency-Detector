import os
import csv
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

# --------------------------------------------------
# Multi-view DINOv2 feature extraction
# --------------------------------------------------
# Reads synchronized pairs from:
# data/multiview/raw/video_x/overhead/frame_0.png
# data/multiview/raw/video_x/overhead/frame_9.png
# data/multiview/raw/video_x/ego/frame_0.png
# data/multiview/raw/video_x/ego/frame_9.png
#
# Saves:
#  before_overhead
#  before_ego
#  after_overhead
#  after_ego
#  before_spatial = [before_overhead ; before_ego]
#  after_spatial  = [after_overhead  ; after_ego]

DATA_ROOT = "data/multiview/raw"
OUTPUT_ROOT = "data/multiview/processed_features"
ACTIONS_CSV = "data/multiview/actions.csv"
MODEL_NAME = "facebook/dinov2-base"


def load_action_map(csv_path):
    action_map = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            action_map[row["video_name"]] = int(row["action_id"])
    return action_map


def extract_feature(image_path, processor, model):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    cls_embedding = outputs.last_hidden_state[:, 0, :]
    return cls_embedding.squeeze(0).cpu().numpy().astype(np.float32)


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    print("Loading DINOv2 model...")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    action_map = load_action_map(ACTIONS_CSV)
    videos = sorted([v for v in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, v))])

    print(f"Processing {len(videos)} videos...")

    for video_name in videos:
        overhead_dir = os.path.join(DATA_ROOT, video_name, "overhead")
        ego_dir = os.path.join(DATA_ROOT, video_name, "ego")

        before_overhead_path = os.path.join(overhead_dir, "frame_0.png")
        after_overhead_path = os.path.join(overhead_dir, "frame_9.png")
        before_ego_path = os.path.join(ego_dir, "frame_0.png")
        after_ego_path = os.path.join(ego_dir, "frame_9.png")

        before_overhead = extract_feature(before_overhead_path, processor, model)
        after_overhead = extract_feature(after_overhead_path, processor, model)
        before_ego = extract_feature(before_ego_path, processor, model)
        after_ego = extract_feature(after_ego_path, processor, model)

        # multi-view state descriptor
        before_spatial = np.concatenate([before_overhead, before_ego]).astype(np.float32)
        after_spatial = np.concatenate([after_overhead, after_ego]).astype(np.float32)

        save_path = os.path.join(OUTPUT_ROOT, f"{video_name}.npz")
        np.savez_compressed(
            save_path,
            action_id=np.int64(action_map[video_name]),
            before_overhead=before_overhead,
            after_overhead=after_overhead,
            before_ego=before_ego,
            after_ego=after_ego,
            before_spatial=before_spatial,
            after_spatial=after_spatial,
        )

    print("Multi-view feature extraction complete ✅")


if __name__ == "__main__":
    main()