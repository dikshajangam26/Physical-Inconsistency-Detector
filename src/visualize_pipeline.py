import os
import csv
import time
import re
from collections import deque

import cv2
import numpy as np
import torch
import rerun as rr

from intervention_engine import generate_counterfactual_variation, get_intervention_summary
from semantic_engine import (
    parse_action_command,
    tracking_data_translator,
    validate_scene_graph,
)

# =========================================================
# Settings
# =========================================================
APP_ID = "miraxis_transition_inspector"
RERUN_ENDPOINT = "rerun+http://127.0.0.1:9876/proxy"

SPATIAL_RESULTS_CSV = "results/multiview/spatial_evaluation_results.csv"
MULTIVIEW_RAW_DATA_DIR = "data/multiview/raw"

TIMELINE_NAME = "sample_step"

# Existing routing thresholds
HUMAN_ROUTING_THRESHOLD = 0.50
SPATIAL_ANOMALY_THRESHOLD = 8.0

# Temporal sync thresholds
TEMPORAL_SYNC_THRESHOLD = 1.4
FRAME_DROP_VISUAL_VELOCITY_THRESHOLD = 0.15

# Semantic mismatch settings
SEMANTIC_ANOMALY_START = 84
SEMANTIC_ANOMALY_END = 90
SEMANTIC_WINDOW = 5
SEMANTIC_MISMATCH_ENTITY = "evidence/spatial_semantic_mismatch"

PLAYBACK_DELAY_SEC = 0.75

OVERHEAD_CAM_ENTITY = "world/overhead_cam"
ROBOT_EGO_CAM_ENTITY = "world/robot_ego_cam"
INTERVENTION_ENTITY = "intervention/proposed_synthetic_variation"

IMG_WIDTH = 64
IMG_HEIGHT = 64

# ---------------------------------------------------------
# Mock camera intrinsics
# ---------------------------------------------------------
OVERHEAD_K = np.array([
    [50.0, 0.0, 32.0],
    [0.0, 50.0, 32.0],
    [0.0, 0.0, 1.0],
], dtype=np.float32)

ROBOT_EGO_K = np.array([
    [45.0, 0.0, 32.0],
    [0.0, 45.0, 32.0],
    [0.0, 0.0, 1.0],
], dtype=np.float32)

OVERHEAD_CAM_POSITION = np.array([0.0, 0.0, 2.0], dtype=np.float32)
ROBOT_EGO_CAM_POSITION = np.array([0.0, -0.6, 0.4], dtype=np.float32)

# ---------------------------------------------------------
# Timing anomaly slices
# ---------------------------------------------------------
TIMING_DRIFT_START = 20
TIMING_DRIFT_END = 40
TIMING_DRIFT_SHIFT = 5

DROPPED_FRAME_START = 60
DROPPED_FRAME_END = 72

ROLLING_WINDOW = 5


# =========================================================
# Helpers
# =========================================================
def safe_video_sort_key(video_name):
    match = re.search(r"(\d+)$", video_name)
    if match:
        return int(match.group(1))
    return video_name


def load_rows(csv_path):
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "video_name": row["video_name"],
                "residual": float(row["residual"]),
                "prob_failure": float(row["prob_failure"]),
                "label": int(row["label"]),
                "spatial_residual_anomaly": float(row["spatial_residual_anomaly"]),
                "overhead_error": float(row["overhead_error"]),
                "ego_error": float(row["ego_error"]),
            })
    rows = sorted(rows, key=lambda x: safe_video_sort_key(x["video_name"]))
    return rows


def read_rgb(path):
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def load_multiview_images(video_name):
    overhead_path = os.path.join(MULTIVIEW_RAW_DATA_DIR, video_name, "overhead", "frame_9.png")
    ego_path = os.path.join(MULTIVIEW_RAW_DATA_DIR, video_name, "ego", "frame_9.png")

    overhead_rgb = read_rgb(overhead_path)
    ego_rgb = read_rgb(ego_path)
    return overhead_rgb, ego_rgb


def log_status_text(message, level="INFO"):
    """
    Version-tolerant TextLog helper.
    """
    try:
        rr.log("routing/status", rr.TextLog(message, level=level))
        return
    except Exception:
        pass

    try:
        level_map = {
            "INFO": rr.TextLogLevel.INFO,
            "WARN": rr.TextLogLevel.WARN,
            "ERROR": rr.TextLogLevel.ERROR,
        }
        rr.log("routing/status", rr.TextLog(message, level=level_map[level]))
        return
    except Exception:
        pass

    rr.log("routing/status", rr.TextLog(message))


def log_camera_geometry():
    try:
        rr.log(OVERHEAD_CAM_ENTITY, rr.Transform3D(translation=OVERHEAD_CAM_POSITION))
    except Exception:
        pass

    try:
        rr.log(ROBOT_EGO_CAM_ENTITY, rr.Transform3D(translation=ROBOT_EGO_CAM_POSITION))
    except Exception:
        pass

    try:
        rr.log(
            OVERHEAD_CAM_ENTITY,
            rr.Pinhole(
                image_from_camera=OVERHEAD_K,
                width=IMG_WIDTH,
                height=IMG_HEIGHT,
            ),
        )
    except Exception:
        try:
            rr.log(
                OVERHEAD_CAM_ENTITY,
                rr.Pinhole(
                    focal_length=[OVERHEAD_K[0, 0], OVERHEAD_K[1, 1]],
                    principal_point=[OVERHEAD_K[0, 2], OVERHEAD_K[1, 2]],
                    width=IMG_WIDTH,
                    height=IMG_HEIGHT,
                ),
            )
        except Exception:
            pass

    try:
        rr.log(
            ROBOT_EGO_CAM_ENTITY,
            rr.Pinhole(
                image_from_camera=ROBOT_EGO_K,
                width=IMG_WIDTH,
                height=IMG_HEIGHT,
            ),
        )
    except Exception:
        try:
            rr.log(
                ROBOT_EGO_CAM_ENTITY,
                rr.Pinhole(
                    focal_length=[ROBOT_EGO_K[0, 0], ROBOT_EGO_K[1, 1]],
                    principal_point=[ROBOT_EGO_K[0, 2], ROBOT_EGO_K[1, 2]],
                    width=IMG_WIDTH,
                    height=IMG_HEIGHT,
                ),
            )
        except Exception:
            pass


# =========================================================
# Hardware + temporal validation helpers
# =========================================================
def make_mock_hardware_velocity_array(num_steps):
    """
    Create a mock hardware joint-velocity signal aligned with the timeline.
    """
    t = np.arange(num_steps, dtype=np.float32)

    hardware = (
        1.8
        + 0.9 * np.sin(0.25 * t)
        + 0.35 * np.cos(0.08 * t)
        + 0.15 * np.sin(0.9 * t)
    )

    return hardware.astype(np.float32)


def inject_timing_drift(hardware_values, start_idx, end_idx, shift_steps):
    """
    Shift hardware values in a slice to simulate lag/lead drift.
    """
    drifted = hardware_values.copy()
    n = len(drifted)

    for i in range(start_idx, min(end_idx, n)):
        shifted_idx = min(max(i + shift_steps, 0), n - 1)
        drifted[i] = hardware_values[shifted_idx]

    return drifted


def center_of_mass_x(rgb_image):
    """
    Estimate a simple x-position from the bright object.
    """
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mask = gray > 10

    if not np.any(mask):
        return 0.0

    xs = np.where(mask)[1].astype(np.float32)
    return float(xs.mean())


def compute_visual_velocity(current_overhead_rgb, previous_overhead_rgb):
    """
    Estimate visual motion from object x-shift.
    """
    if previous_overhead_rgb is None:
        return 0.0

    x_prev = center_of_mass_x(previous_overhead_rgb)
    x_now = center_of_mass_x(current_overhead_rgb)

    return abs(x_now - x_prev)


def rolling_absolute_mismatch(visual_hist, hardware_hist):
    """
    Windowed temporal validation.
    Compare normalized visual-velocity vs hardware-velocity windows.
    """
    if len(visual_hist) == 0 or len(hardware_hist) == 0:
        return 0.0

    v = np.array(visual_hist, dtype=np.float32)
    h = np.array(hardware_hist, dtype=np.float32)

    v = (v - v.mean()) / (v.std() + 1e-6)
    h = (h - h.mean()) / (h.std() + 1e-6)

    return float(np.mean(np.abs(v - h)))


# =========================================================
# Semantic helper
# =========================================================
def choose_action_claim(step_idx, video_name):
    """
    Inject a semantic anomaly slice:
    claim = 'Box placed in bin'
    but observed box still stays on the table in the image.
    """
    if SEMANTIC_ANOMALY_START <= step_idx < SEMANTIC_ANOMALY_END:
        return "Box placed in bin"

    return "Box placed on table"


# =========================================================
# Main
# =========================================================
def main():
    rr.init(APP_ID)
    rr.connect_grpc(RERUN_ENDPOINT)

    print("Connected to Rerun viewer ✅")
    print(f"Loading spatial results from: {SPATIAL_RESULTS_CSV}")

    rows = load_rows(SPATIAL_RESULTS_CSV)
    print(f"Loaded {len(rows)} verification samples ✅")
    print(f"Torch version: {torch.__version__}")

    log_camera_geometry()
    print("Logged camera geometry + intrinsics ✅")

    # -----------------------------------------------------
    # Hardware signal setup
    # -----------------------------------------------------
    base_hardware_velocity = make_mock_hardware_velocity_array(len(rows))
    hardware_velocity = inject_timing_drift(
        base_hardware_velocity,
        TIMING_DRIFT_START,
        TIMING_DRIFT_END,
        TIMING_DRIFT_SHIFT,
    )

    print("Created mock hardware joint velocity array ✅")
    print(
        f"Injected timing drift anomaly from step {TIMING_DRIFT_START} "
        f"to {TIMING_DRIFT_END} with shift {TIMING_DRIFT_SHIFT} ✅"
    )
    print(
        f"Injected dropped-frame visual freeze from step {DROPPED_FRAME_START} "
        f"to {DROPPED_FRAME_END} ✅"
    )

    previous_visual_frame = None
    frozen_overhead_frame = None
    frozen_ego_frame = None

    visual_velocity_history = deque(maxlen=ROLLING_WINDOW)
    hardware_velocity_history = deque(maxlen=ROLLING_WINDOW)
    semantic_discrepancy_history = deque(maxlen=SEMANTIC_WINDOW)

    for step_idx, row in enumerate(rows):
        video_name = row["video_name"]
        residual = row["residual"]
        prob_failure = row["prob_failure"]
        label = row["label"]
        spatial_residual = row["spatial_residual_anomaly"]

        rr.set_time(TIMELINE_NAME, sequence=step_idx)

        # -------------------------------------------------
        # Load visual streams
        # -------------------------------------------------
        overhead_rgb, ego_rgb = load_multiview_images(video_name)

        # Dropped-frame anomaly: visual freeze while hardware continues
        if DROPPED_FRAME_START <= step_idx < DROPPED_FRAME_END:
            if frozen_overhead_frame is None:
                frozen_overhead_frame = overhead_rgb.copy()
                frozen_ego_frame = ego_rgb.copy()

            overhead_rgb = frozen_overhead_frame.copy()
            ego_rgb = frozen_ego_frame.copy()

        rr.log(f"{OVERHEAD_CAM_ENTITY}/image", rr.Image(overhead_rgb))
        rr.log(f"{ROBOT_EGO_CAM_ENTITY}/image", rr.Image(ego_rgb))

        # -------------------------------------------------
        # Existing evidence
        # -------------------------------------------------
        rr.log("evidence/latent_residual", rr.Scalars(residual))
        rr.log("evidence/calibrated_uncertainty", rr.Scalars(prob_failure))
        rr.log("evidence/spatial_residual_anomaly", rr.Scalars(spatial_residual))

        rr.log("meta/video_name", rr.TextLog(f"Video: {video_name}"))
        rr.log("meta/ground_truth_label", rr.Scalars(label))

        # -------------------------------------------------
        # Hardware telemetry
        # -------------------------------------------------
        hardware_joint_velocity = float(hardware_velocity[step_idx])
        rr.log("hardware/joint_velocity", rr.Scalars(hardware_joint_velocity))

        # -------------------------------------------------
        # Temporal validation
        # -------------------------------------------------
        visual_velocity = compute_visual_velocity(overhead_rgb, previous_visual_frame)

        visual_velocity_history.append(float(visual_velocity))
        hardware_velocity_history.append(float(hardware_joint_velocity))

        temporal_sync_drift = rolling_absolute_mismatch(
            visual_velocity_history,
            hardware_velocity_history,
        )

        rr.log("evidence/temporal_sync_drift", rr.Scalars(temporal_sync_drift))

        previous_visual_frame = overhead_rgb.copy()

        # -------------------------------------------------
        # Semantic scene-graph reasoning
        # -------------------------------------------------
        action_claim = choose_action_claim(step_idx, video_name)

        expected_scene_graph, expected_matrix = parse_action_command(action_claim)
        observed_scene_graph, observed_matrix, tracking_info = tracking_data_translator(overhead_rgb)

        mismatch_vector, semantic_mismatch_flag, mismatch_details = validate_scene_graph(
            observed_scene_graph,
            expected_scene_graph,
        )

        semantic_discrepancy_history.append(semantic_mismatch_flag)
        rolling_semantic_mismatch = float(np.mean(semantic_discrepancy_history))

        rr.log(SEMANTIC_MISMATCH_ENTITY, rr.Scalars(rolling_semantic_mismatch))
        rr.log("meta/action_claim", rr.TextLog(f"Action claim: {action_claim}"))

        # -------------------------------------------------
        # Fault logic
        # -------------------------------------------------
        dropped_frame_detected = (
            DROPPED_FRAME_START <= step_idx < DROPPED_FRAME_END
            and visual_velocity <= FRAME_DROP_VISUAL_VELOCITY_THRESHOLD
            and abs(hardware_joint_velocity) > 0.25
        )

        sync_fault_detected = temporal_sync_drift >= TEMPORAL_SYNC_THRESHOLD
        semantic_override = (
            semantic_mismatch_flag == 1
            and action_claim.strip().lower() == "box placed in bin"
        )      
          
        send_to_human = (
            prob_failure >= HUMAN_ROUTING_THRESHOLD
            or spatial_residual >= SPATIAL_ANOMALY_THRESHOLD
        )

        # -------------------------------------------------
        # Global routing logic
        # -------------------------------------------------
        if semantic_override:
            intervention_img = generate_counterfactual_variation(
                ego_rgb,
                "spatial_residual_anomaly"
            )
            rr.log(INTERVENTION_ENTITY, rr.Image(intervention_img))

            semantic_msg = (
                f"[{video_name}] 🚨 SEMANTIC MISMATCH: "
                f"Action claim contradicts physical scene graph state edges. | "
                f"claim='{action_claim}' | "
                f"rolling_semantic_mismatch={rolling_semantic_mismatch:.4f}"
            )

            log_status_text(semantic_msg, level="ERROR")
            print(semantic_msg)

        elif sync_fault_detected or dropped_frame_detected:
            if spatial_residual >= SPATIAL_ANOMALY_THRESHOLD:
                failure_mode = "combined"
            else:
                failure_mode = "temporal_sync_drift"

            intervention_img = generate_counterfactual_variation(ego_rgb, failure_mode)
            rr.log(INTERVENTION_ENTITY, rr.Image(intervention_img))

            intervention_summary = get_intervention_summary(video_name, failure_mode)

            fault_msg = (
                f"🚨 TIMING FAULT: Hardware/Visual Sync Drift Detected | "
                f"step={step_idx} | video={video_name} | "
                f"sync_drift={temporal_sync_drift:.4f} | "
                f"visual_velocity={visual_velocity:.4f} | "
                f"hardware_joint_velocity={hardware_joint_velocity:.4f} | "
                f"{intervention_summary}"
            )

            log_status_text(fault_msg, level="ERROR")
            print(fault_msg)

        elif send_to_human:
            if spatial_residual >= SPATIAL_ANOMALY_THRESHOLD:
                failure_mode = "spatial_residual_anomaly"
            else:
                failure_mode = "temporal_sync_drift"

            intervention_img = generate_counterfactual_variation(ego_rgb, failure_mode)
            rr.log(INTERVENTION_ENTITY, rr.Image(intervention_img))

            intervention_summary = get_intervention_summary(video_name, failure_mode)

            msg = (
                f"[HUMAN REVIEW REQUIRED] {video_name} | "
                f"prob_failure={prob_failure:.4f}, "
                f"latent_residual={residual:.4f}, "
                f"spatial_residual={spatial_residual:.4f}, "
                f"sync_drift={temporal_sync_drift:.4f}, "
                f"semantic_mismatch={rolling_semantic_mismatch:.4f} | "
                f"{intervention_summary}"
            )

            log_status_text(msg, level="WARN")
            print(msg)

        else:
            msg = (
                f"[AUTO ACCEPTED] {video_name} | "
                f"prob_failure={prob_failure:.4f}, "
                f"latent_residual={residual:.4f}, "
                f"spatial_residual={spatial_residual:.4f}, "
                f"sync_drift={temporal_sync_drift:.4f}, "
                f"semantic_mismatch={rolling_semantic_mismatch:.4f}"
            )

            log_status_text(msg, level="INFO")
            print(msg)

        time.sleep(PLAYBACK_DELAY_SEC)

    print("Multi-view + hardware + semantic validation playback complete ✅")


if __name__ == "__main__":
    main()