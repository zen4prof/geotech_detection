# model_inference.py
import onnxruntime as ort
import numpy as np
import cv2


def letterbox(image, new_shape=(640, 640), color=(114, 114, 114)):
    """Resize and pad image while meeting stride-multiple constraints."""
    shape = image.shape[:2]  # current shape [height, width]
    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * ratio)), int(round(shape[0] * ratio))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # width, height padding
    dw /= 2  # divide padding into 2 sides
    dh /= 2

    # resize
    resized = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    # pad
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return padded, ratio, top, left


def load_onnx_model(path):
    """Load ONNX model and return session."""
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return session


def run_onnx_inference(session, image):
    """
    Preprocess image, run ONNX inference, return outputs and resize info.
    image: NumPy array (BGR)
    """
    # Resize and pad image
    img_resized, ratio, pad_top, pad_left = letterbox(image, new_shape=(640, 640))

    # Convert to CHW format, normalize, add batch dim
    img_input = img_resized.transpose(2, 0, 1)  # HWC to CHW
    img_input = np.expand_dims(img_input, 0).astype(np.float32) / 255.0  # scale to [0,1]

    # Inference
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img_input})

    return outputs[0], ratio, pad_top, pad_left
