# model_inference.py

from ultralytics import YOLO
import onnxruntime as ort
from PIL import Image
import numpy as np
import io


# --------------------------
# Load YOLOv8 and ONNX models
# --------------------------
def load_models(pt_path='best.pt', onnx_path='best.onnx'):
    """Load YOLOv8 PyTorch model and ONNX session."""
    pt_model = YOLO(pt_path)                      # Load YOLOv8 model
    onnx_session = ort.InferenceSession(onnx_path)
    return pt_model, onnx_session


# --------------------------
# Run inference with YOLOv8 (PyTorch)
# --------------------------
def run_pytorch_inference(model, image_bytes):
    """
    Run YOLOv8 PyTorch model on an image.

    Returns:
        detections: List of dicts with keys: bbox, confidence, class_id, label
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    results = model(img, verbose=False)[0]

    boxes = results.boxes.xyxy.cpu().numpy()      # Bounding boxes (x1, y1, x2, y2)
    confs = results.boxes.conf.cpu().numpy()      # Confidence scores
    class_ids = results.boxes.cls.cpu().numpy()   # Class IDs
    class_names = model.names                     # ID to name mapping

    detections = []
    for bbox, conf, cls_id in zip(boxes, confs, class_ids):
        detections.append({
            "bbox": bbox.tolist(),
            "confidence": float(conf),
            "class_id": int(cls_id),
            "label": class_names[int(cls_id)].lower()
        })

    return detections


# --------------------------
# Run inference with ONNX model
# --------------------------
def run_onnx_inference(session, image_bytes, input_size=(640, 640)):
    """
    Run inference using ONNX model on an image.

    Returns:
        Raw output (to be parsed separately)
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(input_size)
    img_np = np.array(img).astype(np.float32) / 255.0
    img_np = np.transpose(img_np, (2, 0, 1))        # CHW
    img_np = np.expand_dims(img_np, axis=0)         # NCHW

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    outputs = session.run([output_name], {input_name: img_np})

    return outputs[0]  # Model-specific post-processing needed elsewhere
