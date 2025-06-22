import io
import numpy as np
import torch
import onnxruntime as ort
from PIL import Image

# Load PyTorch model
def load_models(model_path):
    if model_path.endswith('.pt'):
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False)
        model.eval()
        return model
    elif model_path.endswith('.onnx'):
        session = ort.InferenceSession(model_path)
        return session
    else:
        raise ValueError("Unsupported model format. Use .pt or .onnx")

# Run inference with PyTorch model
def run_pytorch_inference(model, image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    results = model(img)
    # results.xyxy[0] shape: (num_detections, 6) [x1, y1, x2, y2, confidence, class]
    detections = results.xyxy[0].cpu().numpy()
    classes = results.names
    output = []
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        output.append({
            "bbox": [x1, y1, x2, y2],
            "confidence": float(conf),
            "class_id": int(cls),
            "class_name": classes[int(cls)]
        })
    return output

# Preprocess image for ONNX model
def preprocess_onnx(image_bytes, input_size=(640, 640)):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize(input_size)
    img_np = np.array(img).astype('float32') / 255.0  # normalize to 0-1
    img_np = np.transpose(img_np, (2, 0, 1))  # HWC to CHW
    img_np = np.expand_dims(img_np, axis=0)  # add batch dim
    return img_np

# Run inference with ONNX model
def run_onnx_inference(session, image_bytes):
    input_name = session.get_inputs()[0].name
    input_tensor = preprocess_onnx(image_bytes)
    outputs = session.run(None, {input_name: input_tensor})
    
    # Assuming output[0] is detection results in format (num_detections, 6) [x1,y1,x2,y2,confidence,class]
    detections = outputs[0]
    output = []
    # NOTE: You must know your class names for ONNX model - define here:
    classes = ["block loss", "crack on ashpat", "long crack", "opening on the wall", 
               "vegetation on wall", "vegetation on slope", "vertical crack", "wall deformation", 
               "bad foundation", "corrosion", "slope deformation"]
    
    for det in detections:
        if len(det) < 6:
            continue
        x1, y1, x2, y2, conf, cls = det
        output.append({
            "bbox": [x1, y1, x2, y2],
            "confidence": float(conf),
            "class_id": int(cls),
            "class_name": classes[int(cls)] if int(cls) < len(classes) else "unknown"
        })
    return output
