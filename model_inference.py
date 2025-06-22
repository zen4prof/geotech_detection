# model_inference.py

from ultralytics import YOLO
import onnxruntime as ort
from PIL import Image
import io
import numpy as np
import cv2

def load_onnx_model(onnx_path="best.onnx"):
    session = ort.InferenceSession(onnx_path)
    return session

def preprocess_image(image, input_size=(640, 640)):
    # Resize and pad to keep aspect ratio
    h0, w0 = image.shape[:2]
    r = min(input_size[0] / h0, input_size[1] / w0)
    new_unpad = (int(w0 * r), int(h0 * r))
    resized = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    # Create padded image
    padded = np.full((input_size[1], input_size[0], 3), 114, dtype=np.uint8)
    dw, dh = (input_size[0] - new_unpad[0]) // 2, (input_size[1] - new_unpad[1]) // 2
    padded[dh:dh+new_unpad[1], dw:dw+new_unpad[0], :] = resized

    # Convert BGR to RGB
    img = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)

    # Normalize to 0-1
    img = img.astype(np.float32) / 255.0

    # HWC to CHW
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = np.ascontiguousarray(img)

    return img, r, dw, dh

def run_onnx_inference(session, image):
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: image})
    return outputs
