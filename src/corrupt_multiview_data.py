import os
import cv2
import random

# --------------------------------------------------
# Multi-view corruption script
# --------------------------------------------------
# Creates a real spatial occlusion anomaly:
# - Overhead last frame stays correct and visible
# - Ego last frame is occluded by a black box
# This makes the object visible in one view but blocked in the other.

DATA_ROOT = "data/multiview/raw"
CORRUPTED_LIST = "data/multiview/corrupted_list.txt"

CORRUPTION_FRACTION = 0.20
IMG_SIZE = 64
OCCLUDER_SIZE = 24


def load_videos(root):
    return sorted([v for v in os.listdir(root) if os.path.isdir(os.path.join(root, v))])


def add_occluder(img):
    out = img.copy()
    x0 = (IMG_SIZE - OCCLUDER_SIZE) // 2
    y0 = (IMG_SIZE - OCCLUDER_SIZE) // 2
    out[y0:y0 + OCCLUDER_SIZE, x0:x0 + OCCLUDER_SIZE] = 0
    return out


def main():
    videos = load_videos(DATA_ROOT)
    num_corrupt = int(len(videos) * CORRUPTION_FRACTION)
    corrupt_videos = random.sample(videos, num_corrupt)

    os.makedirs(os.path.dirname(CORRUPTED_LIST), exist_ok=True)
    with open(CORRUPTED_LIST, "w") as f:
        for vid in corrupt_videos:
            f.write(vid + "\n")

    for vid in corrupt_videos:
        ego_last = os.path.join(DATA_ROOT, vid, "ego", "frame_9.png")
        img = cv2.imread(ego_last)
        if img is None:
            continue

        occluded = add_occluder(img)
        cv2.imwrite(ego_last, occluded)

    print(f"Corrupted {num_corrupt} videos with ego-view occlusion ✅")
    print(f"Saved corrupted list to {CORRUPTED_LIST} ✅")


if __name__ == "__main__":
    main()