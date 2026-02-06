"""
template_matching.py
---------------------
Perform template matching on a target image using OpenCV and highlight the best match
(and optionally multiple matches above a similarity threshold with simple non-maximum suppression).

Usage examples:
  python template_matching.py --image path/to/scene.jpg --template path/to/template.png
  python template_matching.py --image scene.jpg --template patch.png --threshold 0.85 --save-out result.png

Notes:
  * Template matching operates on single-channel (grayscale) images.
  * Method default is TM_CCOEFF_NORMED (values in [-1, 1], where 1 is best match).
"""

import argparse
import os
import sys
from typing import List, Tuple

import cv2
import numpy as np

METHODS = {
    'TM_SQDIFF': cv2.TM_SQDIFF,
    'TM_SQDIFF_NORMED': cv2.TM_SQDIFF_NORMED,
    'TM_CCORR': cv2.TM_CCORR,
    'TM_CCORR_NORMED': cv2.TM_CCORR_NORMED,
    'TM_CCOEFF': cv2.TM_CCOEFF,
    'TM_CCOEFF_NORMED': cv2.TM_CCOEFF_NORMED,
}

def parse_args():
    p = argparse.ArgumentParser(description='Template matching demo with OpenCV')
    p.add_argument('--image', required=True, help='Path to target (scene) image')
    p.add_argument('--template', required=True, help='Path to template image')
    p.add_argument('--method', default='TM_CCOEFF_NORMED', choices=METHODS.keys(),
                   help='OpenCV template matching method')
    p.add_argument('--threshold', type=float, default=None,
                   help='If set (e.g., 0.8), also annotate all matches >= threshold (for _NORMED methods)')
    p.add_argument('--nms-iou', type=float, default=0.3,
                   help='IOU threshold for simple non-maximum suppression when threshold is used')
    p.add_argument('--show', action='store_true', help='Display result windows')
    p.add_argument('--save-out', default=None, help='Optional path to save annotated output image')
    return p.parse_args()

def validate_paths(img_path: str, tpl_path: str):
    if not os.path.isfile(img_path):
        sys.exit(f"[ERROR] Image not found: {img_path}")
    if not os.path.isfile(tpl_path):
        sys.exit(f"[ERROR] Template not found: {tpl_path}")

def match_template(frame_gray: np.ndarray, template_gray: np.ndarray, method_name: str) -> np.ndarray:
    method = METHODS[method_name]
    res = cv2.matchTemplate(frame_gray, template_gray, method)
    return res

def min_max_loc_by_method(result: np.ndarray, method_name: str) -> Tuple[Tuple[int, int], float]:
    # For SQDIFF methods, lower is better; for others, higher is better
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    if 'SQDIFF' in method_name:
        return min_loc, float(min_val)
    else:
        return max_loc, float(max_val)

def draw_best_match(frame_bgr: np.ndarray, top_left: Tuple[int, int], tpl_w: int, tpl_h: int, color=(0, 255, 0), thickness=2):
    bottom_right = (top_left[0] + tpl_w, top_left[1] + tpl_h)
    cv2.rectangle(frame_bgr, top_left, bottom_right, color, thickness)

def to_heatmap(result: np.ndarray) -> np.ndarray:
    # Normalize to 0-255 and apply JET colormap for visualization
    res_norm = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
    res_u8 = res_norm.astype(np.uint8)
    heat = cv2.applyColorMap(res_u8, cv2.COLORMAP_JET)
    return heat

def iou(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> float:
    # boxes as (x1, y1, x2, y2)
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    a_area = (ax2 - ax1) * (ay2 - ay1)
    b_area = (bx2 - bx1) * (by2 - by1)
    union = a_area + b_area - inter_area
    return inter_area / union if union > 0 else 0.0

def nms(boxes: List[Tuple[int,int,int,int]], scores: List[float], iou_thresh: float) -> List[int]:
    # Return indices of kept boxes
    idxs = list(range(len(boxes)))
    # sort by score descending
    idxs.sort(key=lambda i: scores[i], reverse=True)
    keep = []
    while idxs:
        i = idxs.pop(0)
        keep.append(i)
        idxs = [j for j in idxs if iou(boxes[i], boxes[j]) < iou_thresh]
    return keep

def multi_matches(result: np.ndarray, method_name: str, tpl_w: int, tpl_h: int, threshold: float, iou_thresh: float):
    # For SQDIFF, lower is better; convert to score accordingly
    if 'SQDIFF' in method_name:
        # Normalize and invert so higher is better
        res_norm = cv2.normalize(result, None, 0.0, 1.0, cv2.NORM_MINMAX)
        score = 1.0 - res_norm
        mask = (score >= threshold)
    else:
        score = result
        mask = (score >= threshold)
    ys, xs = np.where(mask)
    boxes = []
    scores = []
    for (x, y) in zip(xs, ys):
        boxes.append((x, y, x + tpl_w, y + tpl_h))
        scores.append(float(score[y, x]))
    if not boxes:
        return []
    keep_idx = nms(boxes, scores, iou_thresh)
    kept = [boxes[i] for i in keep_idx]
    return kept

def main():
    args = parse_args()
    validate_paths(args.image, args.template)

    frame_bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    template_bgr = cv2.imread(args.template, cv2.IMREAD_COLOR)
    if frame_bgr is None or template_bgr is None:
        sys.exit('[ERROR] Failed to read image or template.')

    frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
    th, tw = template_gray.shape

    result = match_template(frame_gray, template_gray, args.method)
    best_loc, best_value = min_max_loc_by_method(result, args.method)

    print(f"[INFO] Method: {args.method}")
    print(f"[INFO] Best match value: {best_value:.4f} at location (x={best_loc[0]}, y={best_loc[1]})")

    # Annotate best match
    draw_best_match(frame_bgr, best_loc, tw, th, color=(0, 255, 0), thickness=2)

    # Optionally annotate multiple matches above threshold (for _NORMED methods, threshold in [0,1])
    if args.threshold is not None:
        boxes = multi_matches(result, args.method, tw, th, args.threshold, args.nms_iou)
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 0, 255), 1)
        print(f"[INFO] Multi-match: kept {len(boxes)} boxes with threshold >= {args.threshold}")

    heat = to_heatmap(result)

    if args.save_out:
        # Save side-by-side view: annotated frame + heatmap
        heat_resized = cv2.resize(heat, (frame_bgr.shape[1], frame_bgr.shape[0]))
        side = np.hstack([frame_bgr, heat_resized])
        cv2.imwrite(args.save_out, side)
        print(f"[INFO] Saved output to: {args.save_out}")

    if args.show:
        cv2.imshow('Template Matching - Annotated', frame_bgr)
        cv2.imshow('Template Matching - Heatmap', heat)
        print('[INFO] Press any key in the image window to close...')
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
