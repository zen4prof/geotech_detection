# model_inference.py

from ultralytics import YOLO
import onnxruntime as ort
from PIL import Image
import numpy as np
import io

def load_models(pt_path='best.pt', onnx_path='best.onnx'):
    pt_model = YOLO(pt_path)                      # YOLOv8 PyTorch
    onnx_session = ort.InferenceSession(onnx_path)
    return pt_model, onnx_session

def run_pytorch_inference(model, image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    results = model(img, verbose=False)[0]

    boxes = results.boxes.xyxy.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()
    class_ids = results.boxes.cls.cpu().numpy()
    class_names = model.names

    detections = []
    for bbox, conf, cls_id in zip(boxes, confs, class_ids):
        detections.append({
            "bbox": bbox.tolist(),
            "confidence": float(conf),
            "class_id": int(cls_id),
            "label": class_names[int(cls_id)].lower()
        })

    return detections

def run_onnx_inference(session, image_bytes, input_size=(640, 640)):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(input_size)
    img_np = np.array(img).astype(np.float32) / 255.0
    img_np = np.transpose(img_np, (2, 0, 1))
    img_np = np.expand_dims(img_np, axis=0)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    outputs = session.run([output_name], {input_name: img_np})

    return outputs[0]
