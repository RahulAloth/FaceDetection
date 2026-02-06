"""
eye_detection_haar.py
---------------------
Detect eyes in an image using OpenCV's Haar Cascade classifier and draw CIRCLES around detections.

Usage examples:
  python eye_detection_haar.py --image path/to/faces.jpg --show
  python eye_detection_haar.py --image faces.jpg --save-out eyes_annotated.png

Notes:
  * Defaults to OpenCV's built-in haarcascade_eye.xml via cv2.data.haarcascades.
  * You can tweak detection parameters with --scale-factor, --min-neighbors, and --min-size.
"""

import argparse
import os
import sys

import cv2
import numpy as np

def parse_size(s: str):
    try:
        w, h = s.split(',')
        return (int(w), int(h))
    except Exception:
        raise argparse.ArgumentTypeError('min-size must be in the form W,H (e.g., 10,10)')

def parse_color(s: str):
    try:
        b, g, r = s.split(',')
        return (int(b), int(g), int(r))
    except Exception:
        raise argparse.ArgumentTypeError('color must be B,G,R (e.g., 255,0,0)')

def parse_args():
    p = argparse.ArgumentParser(description='Eye detection with Haar Cascade (draw circles)')
    p.add_argument('--image', required=True, help='Path to input image (read in color)')
    default_cascade = os.path.join(cv2.data.haarcascades, 'haarcascade_eye.xml')
    p.add_argument('--cascade', default=default_cascade, help='Path to Haar cascade XML for eyes')
    p.add_argument('--scale-factor', type=float, default=1.02, help='Scale factor for detectMultiScale')
    p.add_argument('--min-neighbors', type=int, default=20, help='minNeighbors for detectMultiScale')
    p.add_argument('--min-size', type=parse_size, default=(10,10), help='Minimum size W,H for detection')
    p.add_argument('--color', type=parse_color, default=(255,0,0), help='Circle color in B,G,R')
    p.add_argument('--thickness', type=int, default=2, help='Circle line thickness')
    p.add_argument('--show', action='store_true', help='Display the annotated image')
    p.add_argument('--save-out', default=None, help='Optional path to save the annotated image')
    return p.parse_args()

def main():
    args = parse_args()

    if not os.path.isfile(args.image):
        sys.exit(f"[ERROR] Image not found: {args.image}")
    if not os.path.isfile(args.cascade):
        sys.exit(f"[ERROR] Cascade XML not found: {args.cascade}")

    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        sys.exit('[ERROR] Failed to read image.')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    eye_cascade = cv2.CascadeClassifier(args.cascade)
    if eye_cascade.empty():
        sys.exit('[ERROR] Failed to load Haar cascade classifier.')

    eyes = eye_cascade.detectMultiScale(
        gray,
        scaleFactor=args.scale_factor,
        minNeighbors=args.min_neighbors,
        minSize=args.min_size
    )

    print(f"[INFO] Detected eyes: {len(eyes)}")
    for (x, y, w, h) in eyes:
        # Circle center is rectangle center
        xc = x + w / 2.0
        yc = y + h / 2.0
        radius = w / 2.0  # or use h/2.0
        cv2.circle(img, (int(xc), int(yc)), int(radius), args.color, args.thickness)

    if args.save_out:
        cv2.imwrite(args.save_out, img)
        print(f"[INFO] Saved annotated image to: {args.save_out}")

    if args.show:
        cv2.imshow('Eye Detection (Haar) - Circles', img)
        print('[INFO] Press any key in the image window to close...')
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
