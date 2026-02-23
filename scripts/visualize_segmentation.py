#!/usr/bin/env python3
"""Visualize SegFormer segmentation output from C++ pipeline.

Takes the raw class-index PNG from `trtf segment` and the original image,
produces a color-coded mask and an overlay.

Usage:
    python3 scripts/visualize_segmentation.py \
      --classes /tmp/segformer_classes.png \
      --image tests/e2e/data/test_img.jpeg \
      --mask /tmp/segformer_mask.png \
      --overlay /tmp/segformer_overlay.png
"""
import argparse
import sys

import numpy as np
from PIL import Image


# ADE20K 150-class color palette (standard dataset colors).
# fmt: off
ADE20K_PALETTE = np.array([
    [120, 120, 120], [180, 120, 120], [  6, 230, 230], [  80,  50,  50],
    [  4, 200,   3], [120, 120,  80], [140, 140, 140], [204,   5, 255],
    [230, 230, 230], [  4, 250,   7], [224,   5, 255], [235, 255,   7],
    [150,   5,  61], [120, 120,  70], [  8, 255,  51], [255,   6,  82],
    [143, 255, 140], [204, 255,   4], [255,  51,   7], [204,  70,   3],
    [  0, 102, 200], [  61, 230, 250], [255,   6,  51], [ 11, 102, 255],
    [255,   7,  71], [255,   9, 224], [  9,   7, 230], [220, 220, 220],
    [255,   9,  92], [112,   9, 255], [  8, 255, 214], [  7, 255, 224],
    [255, 184,   6], [ 10, 255,  71], [255,  41,  10], [ 7, 255, 255],
    [224, 255,   8], [102,   8, 255], [255,  61,   6], [255, 194,   7],
    [255, 122,   8], [  0, 255,  20], [255,   8,  41], [255,   5, 153],
    [  6,  51, 255], [235,  12, 255], [160, 150,  20], [  0, 163, 255],
    [ 140, 140, 140], [250,  10,  15], [ 20, 255,   0], [  31, 255,   0],
    [255,  31,   0], [255, 224,   0], [153, 255,   0], [  0,   0, 255],
    [255,  71, 0  ], [  0, 235, 255], [  0, 173, 255], [  31,   0, 255],
    [  11, 200, 200], [255,  82,   0], [  0, 255, 245], [  0,  61, 255],
    [  0, 255, 112], [  0, 255, 133], [255,   0,   0], [255, 163,   0],
    [255, 102,   0], [194, 255,   0], [  0, 143, 255], [ 51, 255,   0],
    [  0,  82, 255], [  0, 255,  41], [  0, 255, 173], [ 10,   0, 255],
    [173, 255, 0  ], [  0, 255, 153], [255,  92,   0], [255,   0, 255],
    [255,   0, 245], [255,   0, 102], [255, 173,   0], [255,   0, 20 ],
    [255, 184, 184], [  0,  31, 255], [  0, 255,  61], [  0,  71, 255],
    [255,   0, 204], [  0, 255,  194], [  0, 255,  82], [  0, 10, 255],
    [  0, 112, 255], [ 51,   0, 255], [  0, 194, 255], [  0, 122, 255],
    [  0, 255, 163], [255, 153,   0], [  0, 255,  10], [255, 112,   0],
    [143, 255,   0], [ 82,   0, 255], [163, 255,   0], [255, 235,   0],
    [  8, 184, 170], [133,   0, 255], [  0, 255,  92], [184,   0, 255],
    [255,   0,  31], [  0, 184, 255], [  0, 214, 255], [255,   0, 112],
    [ 92, 255,   0], [  0, 224, 255], [112, 224, 255], [ 70, 184, 160],
    [163,   0, 255], [153,   0, 255], [ 71, 255,   0], [255,   0,163],
    [255, 204,   0], [255,   0, 143], [  0, 255, 235], [133, 255,   0],
    [255,   0, 235], [245,   0, 255], [255,   0, 122], [255, 245,   0],
    [ 10, 190, 212], [214, 255,   0], [  0, 204, 255], [ 20,   0, 255],
    [255, 255,   0], [  0, 153, 255], [  0,  41, 255], [  0, 255, 204],
    [ 41,   0, 255], [ 41, 255,   0], [173,   0, 255], [  0, 245, 255],
    [ 71,   0, 255], [122,   0, 255], [  0, 255, 184], [  0, 92, 255],
    [184, 255,   0], [  0, 133, 255], [255, 214,   0], [ 25, 194, 194],
    [102, 255,   0], [92,   0, 255],
], dtype=np.uint8)
# fmt: on


def main():
    parser = argparse.ArgumentParser(
        description="Visualize SegFormer segmentation from C++ output")
    parser.add_argument("--classes", required=True,
                        help="Class-index PNG from trtf segment")
    parser.add_argument("--image", required=True,
                        help="Original input image")
    parser.add_argument("--mask", default="/tmp/segformer_mask.png",
                        help="Output: color-coded segmentation mask")
    parser.add_argument("--overlay", default="/tmp/segformer_overlay.png",
                        help="Output: overlay of mask on original image")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Overlay blend alpha (0=original, 1=mask)")
    args = parser.parse_args()

    # Load class-index image (grayscale, each pixel = class ID)
    class_img = np.array(Image.open(args.classes).convert("L"))
    print(f"Class map: {class_img.shape[1]}x{class_img.shape[0]}, "
          f"classes: {np.unique(class_img).tolist()}", file=sys.stderr)

    # Load original image to get target dimensions
    orig = Image.open(args.image).convert("RGB")
    target_w, target_h = orig.size

    # Upsample class map to original image size (nearest-neighbor for sharp edges)
    class_pil = Image.fromarray(class_img)
    class_upsampled = np.array(
        class_pil.resize((target_w, target_h), Image.NEAREST))

    # Apply color palette to create mask at full resolution
    palette = ADE20K_PALETTE
    mask_rgb = palette[class_upsampled]  # [H, W, 3]

    # Save mask
    mask_pil = Image.fromarray(mask_rgb)
    mask_pil.save(args.mask)
    print(f"Mask saved: {args.mask} ({target_w}x{target_h})", file=sys.stderr)

    orig_np = np.array(orig)

    # Alpha-blend: overlay = alpha * mask + (1 - alpha) * original
    alpha = args.alpha
    overlay_np = (alpha * mask_rgb.astype(np.float32) +
                  (1 - alpha) * orig_np.astype(np.float32))
    overlay_np = np.clip(overlay_np, 0, 255).astype(np.uint8)

    overlay_pil = Image.fromarray(overlay_np)
    overlay_pil.save(args.overlay)
    print(f"Overlay saved: {args.overlay}", file=sys.stderr)


if __name__ == "__main__":
    main()
