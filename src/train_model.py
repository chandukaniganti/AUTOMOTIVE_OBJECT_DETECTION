"""
Wrapper to start YOLOv8 training using ultralytics API.
You must prepare a `data.yaml` file with paths to train/val and class names.

Example data.yaml:

train: ../data/processed/images/train
val: ../data/processed/images/val

nc: 3
names: ['car', 'person', 'truck']

Run:
python src/train_model.py --data data/data.yaml --epochs 50 --imgsz 640
"""

import argparse
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path to data.yaml')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='YOLOv8 model path or name')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size for training')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.model)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch)


if __name__ == '__main__':
    main()
