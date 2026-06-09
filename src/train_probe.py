import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# --------------------------------------------------
# Train multi-view Transition Probe
# --------------------------------------------------
# Input  = before_spatial + one-hot(action)
# Output = after_spatial
#
# Trained only on clean videos.

FEATURE_DIR = "data/multiview/processed_features"
CORRUPTED_LIST = "data/multiview/corrupted_list.txt"
MODEL_DIR = "models/multiview"
MODEL_PATH = os.path.join(MODEL_DIR, "probe_model_multiview.pth")

NUM_ACTIONS = 4
EPOCHS = 20
LR = 1e-3
SEED = 42


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
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    corrupted = load_corrupted_set(CORRUPTED_LIST)
    feature_files = sorted([f for f in os.listdir(FEATURE_DIR) if f.endswith(".npz")])

    X = []
    Y = []

    for filename in feature_files:
        video_name = filename.replace(".npz", "")

        # train only on clean data
        if video_name in corrupted:
            continue

        data = np.load(os.path.join(FEATURE_DIR, filename))

        before_spatial = data["before_spatial"].astype(np.float32)
        after_spatial = data["after_spatial"].astype(np.float32)
        action_id = int(data["action_id"])

        action = np.zeros(NUM_ACTIONS, dtype=np.float32)
        action[action_id] = 1.0

        input_vector = np.concatenate([before_spatial, action]).astype(np.float32)

        X.append(input_vector)
        Y.append(after_spatial)

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)

    print("Training samples (clean only):", len(X))
    print("Input shape:", X.shape)
    print("Output shape:", Y.shape)

    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)

    model = TransitionModel(input_size=X.shape[1], output_size=Y.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()

        pred = model(X)
        loss = criterion(pred, Y)

        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {loss.item():.6f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)

    print(f"Saved multi-view probe to {MODEL_PATH} ✅")


if __name__ == "__main__":
    main()