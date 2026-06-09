import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

# --------------------------------------------------
# Evaluate multi-view probe
# --------------------------------------------------
# True multi-view embedding residual:
#   residual = || z_after_spatial - z_hat_after_spatial ||_2
#
# Spatial inconsistency spike:
#   spatial_residual_anomaly = | ego_error - overhead_error |
#
# Idea:
# If ego view is occluded but overhead view is still correct,
# ego error becomes much larger than overhead error.

FEATURE_DIR = "data/multiview/processed_features"
CORRUPTED_LIST = "data/multiview/corrupted_list.txt"
MODEL_PATH = "models/multiview/probe_model_multiview.pth"

RESULTS_DIR = "results"
RESULTS_CSV = os.path.join(RESULTS_DIR, "multiview", "spatial_evaluation_results.csv")

NUM_ACTIONS = 4


def load_corrupted_set(path):
    if not os.path.exists(path):
        return set()

    with open(path, "r") as f:
        return set(line.strip() for line in f.readlines() if line.strip())


class TransitionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_size),
        )

    def forward(self, x):
        return self.net(x)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    corrupted = load_corrupted_set(CORRUPTED_LIST)
    feature_files = sorted([f for f in os.listdir(FEATURE_DIR) if f.endswith(".npz")])

    sample = np.load(os.path.join(FEATURE_DIR, feature_files[0]))
    spatial_dim = sample["before_spatial"].shape[0]

    model = TransitionModel(input_size=spatial_dim + NUM_ACTIONS, output_size=spatial_dim)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    results = []
    residuals_for_calibration = []
    labels_for_calibration = []

    for filename in feature_files:
        video_name = filename.replace(".npz", "")
        data = np.load(os.path.join(FEATURE_DIR, filename))

        before_spatial = data["before_spatial"].astype(np.float32)
        after_spatial = data["after_spatial"].astype(np.float32)

        after_overhead = data["after_overhead"].astype(np.float32)
        after_ego = data["after_ego"].astype(np.float32)

        action_id = int(data["action_id"])

        action = np.zeros(NUM_ACTIONS, dtype=np.float32)
        action[action_id] = 1.0

        input_vector = np.concatenate([before_spatial, action]).astype(np.float32)
        input_tensor = torch.tensor(input_vector, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            pred_spatial = model(input_tensor).squeeze(0).numpy()

        # split prediction into two views
        half = pred_spatial.shape[0] // 2
        pred_overhead = pred_spatial[:half]
        pred_ego = pred_spatial[half:]

        # true residual in combined multi-view embedding space
        residual = float(np.linalg.norm(after_spatial - pred_spatial))

        # per-view residuals
        overhead_error = float(np.linalg.norm(after_overhead - pred_overhead))
        ego_error = float(np.linalg.norm(after_ego - pred_ego))

        # visibility mismatch / spatial inconsistency
        spatial_residual_anomaly = float(abs(ego_error - overhead_error))

        label = 1 if video_name in corrupted else 0

        results.append({
            "video_name": video_name,
            "residual": residual,
            "spatial_residual_anomaly": spatial_residual_anomaly,
            "overhead_error": overhead_error,
            "ego_error": ego_error,
            "label": label,
        })

        residuals_for_calibration.append(residual)
        labels_for_calibration.append(label)

    # Platt scaling on the multi-view residuals
    residuals = np.array(residuals_for_calibration, dtype=np.float32).reshape(-1, 1)
    labels = np.array(labels_for_calibration, dtype=np.int64)

    calibrator = LogisticRegression()
    calibrator.fit(residuals, labels)

    probabilities = calibrator.predict_proba(residuals)[:, 1]

    for i, p in enumerate(probabilities):
        results[i]["prob_failure"] = float(p)

    # save CSV for your Rerun visualization
    with open(RESULTS_CSV, "w") as f:
        f.write("video_name,residual,prob_failure,label,spatial_residual_anomaly,overhead_error,ego_error\n")
        for row in results:
            f.write(
                f"{row['video_name']},{row['residual']},{row['prob_failure']},{row['label']},"
                f"{row['spatial_residual_anomaly']},{row['overhead_error']},{row['ego_error']}\n"
            )

    clean = [r["residual"] for r in results if r["label"] == 0]
    corrupt = [r["residual"] for r in results if r["label"] == 1]

    spatial_clean = [r["spatial_residual_anomaly"] for r in results if r["label"] == 0]
    spatial_corrupt = [r["spatial_residual_anomaly"] for r in results if r["label"] == 1]

    print("Multi-view evaluation complete ✅")
    print(f"Mean clean residual: {np.mean(clean):.4f}")
    print(f"Mean corrupt residual: {np.mean(corrupt):.4f}")
    print(f"Mean clean spatial anomaly: {np.mean(spatial_clean):.4f}")
    print(f"Mean corrupt spatial anomaly: {np.mean(spatial_corrupt):.4f}")

    auroc = roc_auc_score(labels, probabilities)
    auprc = average_precision_score(labels, probabilities)
    brier = brier_score_loss(labels, probabilities)

    print(f"AUROC: {auroc:.6f}")
    print(f"AUPRC: {auprc:.6f}")
    print(f"Brier Score: {brier:.6f}")
    print(f"Saved spatial results to {RESULTS_CSV} ✅")


if __name__ == "__main__":
    main()