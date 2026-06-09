import os
import csv
import cv2
import numpy as np
import shutil

# clean old data first
if os.path.exists("data/raw"):
    shutil.rmtree("data/raw")

os.makedirs("data/raw", exist_ok=True)

num_videos = 100
frames_per_video = 10
img_size = 64
square_size = 10
step = 4

# 4 simple actions
# 0 = right, 1 = left, 2 = down, 3 = up
actions = {
    0: (step, 0),
    1: (-step, 0),
    2: (0, step),
    3: (0, -step),
}

metadata_path = "data/actions.csv"

with open(metadata_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["video_name", "action_id"])

    for vid in range(num_videos):
        video_name = f"video_{vid}"
        video_folder = os.path.join("data/raw", video_name)
        os.makedirs(video_folder, exist_ok=True)

        action_id = vid % 4
        dx, dy = actions[action_id]

        # choose start position safely so square stays in frame
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

        writer.writerow([video_name, action_id])

        for frame in range(frames_per_video):
            img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
            cv2.rectangle(img, (x, y), (x + square_size, y + square_size), (255, 255, 255), -1)

            frame_path = os.path.join(video_folder, f"frame_{frame}.png")
            cv2.imwrite(frame_path, img)

            x += dx
            y += dy

print("New dataset generated ✅")
print("Saved action labels in data/actions.csv ✅")