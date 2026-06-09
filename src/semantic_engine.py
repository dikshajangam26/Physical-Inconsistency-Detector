import cv2
import numpy as np

# =========================================================
# Semantic ontology / categories
# =========================================================
OBJECT_TYPES = {
    "box": "movable_object",
    "table": "support_surface",
    "bin": "container",
}

RELATION_TYPES = {
    "ON_TOP_OF",
    "INSIDE",
}

# ---------------------------------------------------------
# IMPORTANT:
# For this synthetic dataset, we treat almost the whole scene
# as "table", unless the box clearly falls inside the bin region.
# This makes normal samples stable and avoids false positives.
# ---------------------------------------------------------
TABLE_BBOX = (0, 0, 63, 63)
BIN_BBOX = (46, 28, 62, 48)


# =========================================================
# Helpers
# =========================================================
def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bbox_intersection_area(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    x_left = max(ax1, bx1)
    y_top = max(ay1, by1)
    x_right = min(ax2, bx2)
    y_bottom = min(ay2, by2)

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    return float((x_right - x_left) * (y_bottom - y_top))


def bbox_area(bbox):
    x1, y1, x2, y2 = bbox
    return float(max(0, x2 - x1) * max(0, y2 - y1))


def bbox_iou(a, b):
    inter = bbox_intersection_area(a, b)
    union = bbox_area(a) + bbox_area(b) - inter
    if union <= 0:
        return 0.0
    return inter / union


# =========================================================
# Scene graph representation
# =========================================================
def compile_scene_graph(nodes, edges):
    return {
        "nodes": nodes,
        "edges": edges,
    }


def scene_graph_to_relation_matrix(scene_graph):
    matrix = {}
    for edge in scene_graph["edges"]:
        key = (edge["src"], edge["rel"], edge["dst"])
        matrix[key] = 1
    return matrix


# =========================================================
# Action parser -> expected semantic graph
# =========================================================
def parse_action_command(action_command):
    """
    For this synthetic project, only compare the final box-support relation.
    """
    action_command = action_command.strip().lower()

    nodes = [
        {"id": "box", "type": OBJECT_TYPES["box"]},
        {"id": "table", "type": OBJECT_TYPES["table"]},
        {"id": "bin", "type": OBJECT_TYPES["bin"]},
    ]

    if action_command == "box placed in bin":
        edges = [
            {"src": "box", "rel": "INSIDE", "dst": "bin"},
        ]
    else:
        # normal default
        edges = [
            {"src": "box", "rel": "ON_TOP_OF", "dst": "table"},
        ]

    scene_graph = compile_scene_graph(nodes, edges)
    relation_matrix = scene_graph_to_relation_matrix(scene_graph)

    return scene_graph, relation_matrix


# =========================================================
# Tracking translator
# =========================================================
def infer_box_bbox_from_image(image_rgb):
    """
    Detect the bright square (box) from the image.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    mask = gray > 10

    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return (0, 0, 1, 1)

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())

    return (x1, y1, x2, y2)


def infer_box_relation(box_bbox):
    """
    Stable synthetic semantic logic:
    - if box overlaps bin enough -> INSIDE bin
    - otherwise -> ON_TOP_OF table
    """
    bin_iou = bbox_iou(box_bbox, BIN_BBOX)

    if bin_iou > 0.18:
        return {"src": "box", "rel": "INSIDE", "dst": "bin"}

    return {"src": "box", "rel": "ON_TOP_OF", "dst": "table"}


def tracking_data_translator(image_rgb):
    nodes = [
        {"id": "box", "type": OBJECT_TYPES["box"]},
        {"id": "table", "type": OBJECT_TYPES["table"]},
        {"id": "bin", "type": OBJECT_TYPES["bin"]},
    ]

    box_bbox = infer_box_bbox_from_image(image_rgb)
    relation = infer_box_relation(box_bbox)

    scene_graph = compile_scene_graph(nodes, [relation])
    relation_matrix = scene_graph_to_relation_matrix(scene_graph)

    tracking_info = {
        "box_bbox": box_bbox,
        "table_bbox": TABLE_BBOX,
        "bin_bbox": BIN_BBOX,
    }

    return scene_graph, relation_matrix, tracking_info


# =========================================================
# Semantic validation
# =========================================================
def validate_scene_graph(observed_graph, expected_graph):
    """
    Compare only the single main box relation.

    Returns:
      mismatch_vector = [0 or 1]
      mismatch_flag = 0 or 1
      mismatch_details = list
    """
    observed_edges = observed_graph["edges"]
    expected_edges = expected_graph["edges"]

    observed_rel = observed_edges[0] if len(observed_edges) > 0 else None
    expected_rel = expected_edges[0] if len(expected_edges) > 0 else None

    if observed_rel == expected_rel:
        return [0], 0, []

    mismatch_details = [{
        "expected": expected_rel,
        "observed": observed_rel,
    }]
    return [1], 1, mismatch_details