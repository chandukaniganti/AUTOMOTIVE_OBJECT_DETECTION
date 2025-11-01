# Automotive Object Detection — YOLOv8 Demo


## Quick start (demo)


1. Create and activate a Python virtual environment (recommended).
2. Install requirements: `pip install -r requirements.txt`.
3. Put a sample road video in `data/raw/sample_video.mp4` (or use webcam).
4. Run the detection demo: `python src/detect_objects.py --source data/raw/sample_video.mp4`.


## Streamlit dashboard


Run the interactive demo: `streamlit run app/dashboard.py`.


## Fine-tuning on custom dataset


Prepare `data.yaml` for YOLO training and use `src/train_model.py` or run the `yolo` CLI command as configured in the training script.


## Notes
- `ultralytics` will auto-download `yolov8n.pt` if not present.
- For edge deployment, convert the trained weights to ONNX/TensorRT separately.