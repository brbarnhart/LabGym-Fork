"""Generate a tiny synthetic video for testing the annotator without real footage.

Usage:
    python -m utils.make_test_video output.mp4 --frames 300 --fps 30
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def make_test_video(path: str, frames: int = 300, fps: float = 30.0, size=(640, 480)):
    path = Path(path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)

    for f in range(frames):
        img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        # Simple moving bar + text to simulate "behavior"
        x = int((f % 80) * (size[0] / 80))
        cv2.rectangle(img, (x, 100), (x + 80, 200), (0, 180, 255), -1)
        cv2.putText(img, f"frame {f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        writer.write(img)

    writer.release()
    print(f"Created synthetic test video: {path} ({frames} frames @ {fps} fps)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", help="Output .mp4 path")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()
    make_test_video(args.output, args.frames, args.fps)
