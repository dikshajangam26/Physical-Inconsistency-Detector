import os
import csv
import random
import cv2
import numpy as np

data_dir = "data/raw"

# load action labels
action_map = {}
with open("data/actions.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        action_map[row["video_name"]] = int(row["action_id"])

videos = sorted(os.listdir(data_dir))
num_corrupt = int(0.2 * len(videos))
corrupt_videos = random.sample(videos, num_corrupt)

with open("data/corrupted_list.txt", "w") as f:
    for vid in corrupt_videos:
        f.write(vid + "\n")

for vid in corrupt_videos:
    video_path = os.path.join(data_dir, vid)
    last_frame_path = os.path.join(video_path, "frame_9.png")

    # make strong corruption: random noise image
    noise = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    cv2.imwrite(last_frame_path, noise)

print(f"Corrupted {num_corrupt} videos ✅")
print("Saved corrupted list to data/corrupted_list.txt ✅")
