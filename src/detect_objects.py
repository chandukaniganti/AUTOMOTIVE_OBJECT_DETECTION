"""
python src/detect_objects.py --source path/to/video.mp4
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import time
import cv2
from ultralytics import YOLO
from src.utils import draw_text, resize_for_display
from src.lane_detection import detect_lanes


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='0',
                        help='Video source (0 for webcam or path to video)')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                        help='YOLOv8 model weights or name')
    parser.add_argument('--show-lanes', action='store_true', help='Enable lane detection')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--device', type=str, default='cpu', help='cpu or 0 for GPU')
    return parser.parse_args()


def main():
    args = parse_args()

    src = args.source
    if src == '0' or src == 'webcam':
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(src)
        print("🎥 Video open status:", cap.isOpened())

    model = YOLO(args.model)
    print("✅ Model loaded successfully. Starting video:", args.source)
    model.fuse()

    fps_avg = 0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start = time.time()

        # run detection (ultralytics model expects BGR images)
        results = model(frame, conf=args.conf)[0]

        # plot results (returns annotated image in BGR)
        annotated = results.plot()

        if args.show_lanes:
            annotated = detect_lanes(annotated)

        # annotate FPS and info
        frame_count += 1
        end = time.time()
        fps = 1 / (end - start + 1e-6)
        fps_avg = (fps_avg * (frame_count - 1) + fps) / frame_count
        draw_text(annotated, f"FPS: {fps_avg:.1f}", org=(10, 30))

        display = resize_for_display(annotated, width=1280)
        cv2.imshow('Automotive Object Detection', display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
