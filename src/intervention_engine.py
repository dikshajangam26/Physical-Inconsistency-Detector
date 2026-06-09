import cv2
import numpy as np


def apply_gaussian_repair(image_rgb, kernel_size=9, sigma=2.0):
    """
    Apply a soft blur to simulate correction for localized spatial distortion
    or occlusion noise.
    """
    return cv2.GaussianBlur(image_rgb, (kernel_size, kernel_size), sigma)


def apply_localized_gamma_contrast(image_rgb, gamma=0.85, alpha=1.20, beta=8):
    """
    Apply a localized gamma + contrast shift in the center region
    to simulate illumination leveling for timing/temporal faults.
    """
    out = image_rgb.copy().astype(np.float32) / 255.0

    h, w = out.shape[:2]
    x0 = int(w * 0.20)
    x1 = int(w * 0.80)
    y0 = int(h * 0.20)
    y1 = int(h * 0.80)

    roi = out[y0:y1, x0:x1]

    # gamma correction
    roi = np.power(np.clip(roi, 0.0, 1.0), gamma)

    # contrast + brightness
    roi = np.clip(alpha * roi + (beta / 255.0), 0.0, 1.0)

    out[y0:y1, x0:x1] = roi
    out = (out * 255.0).astype(np.uint8)
    return out


def generate_counterfactual_variation(image_rgb, failure_mode):
    """
    Generate a proposed synthetic intervention image based on the detected failure mode.

    failure_mode:
      - 'spatial_residual_anomaly'
      - 'temporal_sync_drift'
      - 'combined'
      - anything else -> return original
    """
    if image_rgb is None:
        raise ValueError("image_rgb is None")

    if failure_mode == "spatial_residual_anomaly":
        return apply_gaussian_repair(image_rgb)

    if failure_mode == "temporal_sync_drift":
        return apply_localized_gamma_contrast(image_rgb)

    if failure_mode == "combined":
        temp = apply_gaussian_repair(image_rgb)
        temp = apply_localized_gamma_contrast(temp)
        return temp

    return image_rgb.copy()


def get_intervention_summary(sample_name, failure_mode):
    """
    Return a readable summary string for logs.
    """
    if failure_mode == "spatial_residual_anomaly":
        return (
            f"[{sample_name}] 🛠️ INTERVENTION GENERATED: "
            f"Counterfactual variation rendered to correct localized anomaly channel."
        )

    if failure_mode == "temporal_sync_drift":
        return (
            f"[{sample_name}] 🛠️ INTERVENTION GENERATED: "
            f"Counterfactual variation rendered to stabilize temporal illumination mismatch."
        )

    if failure_mode == "combined":
        return (
            f"[{sample_name}] 🛠️ INTERVENTION GENERATED: "
            f"Counterfactual variation rendered to correct both spatial and temporal anomaly channels."
        )

    return (
        f"[{sample_name}] 🛠️ INTERVENTION GENERATED: "
        f"Counterfactual variation rendered."
    )