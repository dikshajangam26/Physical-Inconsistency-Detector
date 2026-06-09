import os
import csv
import cv2
import shutil
import numpy as np

# ---------------------------------
# Multi-view synthetic dataset maker
# ---------------------------------
# Creates two synchronized camera views for each video:
#   1) overhead
#   2) ego
#
# Folder structure:
# data/multiview/raw/video_0/overhead/frame_0.png
# data/multiview/raw/video_0/ego/frame_0.png
#
# Action ids:
# 0 = right, 1 = left, 2 = down, 3 = up

OUTPUT_ROOT = "data/multiview/raw"
ACTIONS_CSV = "data/multiview/actions.csv"

NUM_VIDEOS = 100
FRAMES_PER_VIDEO = 10
IMG_SIZE = 64
SQUARE_SIZE = 10
STEP = 4

ACTIONS = {
    0: (STEP, 0),    # right
    1: (-STEP, 0),   # left
    2: (0, STEP),    # down
    3: (0, -STEP),   # up
}


def make_clean_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def make_blank():
    return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)


def draw_square(img, x, y, color=(255, 255, 255)):
    cv2.rectangle(img, (x, y), (x + SQUARE_SIZE, y + SQUARE_SIZE), color, -1)
    return img


def project_to_ego(x, y):
    """
    Fake egocentric projection.
    This is not real 3D geometry, but it gives a second synchronized camera view.
    """
    ego_x = int(8 + 0.75 * x)
    ego_y = int(46 - 0.45 * y)

    ego_x = max(0, min(ego_x, IMG_SIZE - SQUARE_SIZE - 1))
    ego_y = max(0, min(ego_y, IMG_SIZE - SQUARE_SIZE - 1))
    return ego_x, ego_y


def choose_safe_start(action_id):
    if action_id == 0:   # right
        x = np.random.randint(0, 15)
        y = np.random.randint(10, 45)
    elif action_id == 1: # left
        x = np.random.randint(40, 54)
        y = np.random.randint(10, 45)
    elif action_id == 2: # down
        x = np.random.randint(10, 45)
        y = np.random.randint(0, 15)
    else:                # up
        x = np.random.randint(10, 45)
        y = np.random.randint(40, 54)
    return x, y


def main():
    make_clean_dir(OUTPUT_ROOT)
    os.makedirs(os.path.dirname(ACTIONS_CSV), exist_ok=True)

    with open(ACTIONS_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video_name", "action_id"])

        for vid in range(NUM_VIDEOS):
            video_name = f"video_{vid}"
            action_id = vid % 4
            dx, dy = ACTIONS[action_id]
            x, y = choose_safe_start(action_id)

            overhead_dir = os.path.join(OUTPUT_ROOT, video_name, "overhead")
            ego_dir = os.path.join(OUTPUT_ROOT, video_name, "ego")
            os.makedirs(overhead_dir, exist_ok=True)
            os.makedirs(ego_dir, exist_ok=True)

            writer.writerow([video_name, action_id])

            for frame_idx in range(FRAMES_PER_VIDEO):
                overhead = make_blank()
                ego = make_blank()

                overhead = draw_square(overhead, x, y)

                ego_x, ego_y = project_to_ego(x, y)
                ego = draw_square(ego, ego_x, ego_y)

                cv2.imwrite(os.path.join(overhead_dir, f"frame_{frame_idx}.png"), overhead)
                cv2.imwrite(os.path.join(ego_dir, f"frame_{frame_idx}.png"), ego)

                x += dx
                y += dy
                x = max(0, min(x, IMG_SIZE - SQUARE_SIZE - 1))
                y = max(0, min(y, IMG_SIZE - SQUARE_SIZE - 1))

    print("Multi-view dataset generated ✅")
    print(f"Saved action labels to {ACTIONS_CSV} ✅")


if __name__ == "__main__":
    main()