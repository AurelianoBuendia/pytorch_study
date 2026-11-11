"""Track objects in a video and write an annotated video."""

import argparse
from pathlib import Path

import cv2

from api import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--weights", type=Path, default=Path("../models/yolo11n.pt"))
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    if args.video.resolve() == args.output.resolve():
        parser.error("output must differ from the input video")

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {args.video}")
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if fps <= 0 or width <= 0 or height <= 0:
        raise ValueError("video has invalid frame rate or dimensions")

    model = YOLO(args.weights)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise ValueError(f"could not open output video: {args.output}")
    results = model.track(args.video, tracker=args.tracker, imgsz=args.imgsz, stream=True)
    try:
        for result in results:
            frame = result.orig_img.copy()
            for box in result.boxes:
                x1, y1, x2, y2 = (int(value) for value in box.xyxy[0])
                track_id = int(box.id[0])
                class_id = int(box.cls[0])
                label = f"{model.names[class_id]} #{track_id} {float(box.conf[0]):.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, max(y1 - 5, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            writer.write(frame)
    finally:
        results.close()
        writer.release()


if __name__ == "__main__":
    main()
