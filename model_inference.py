# model_inference.py

import onnxruntime as ort
import numpy as np

def load_onnx_model(model_path: str):
    """
    Load ONNX model and return an inference session.
    """
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    return session

def run_onnx_inference(session, input_tensor: np.ndarray):
    """
    Run inference on the input tensor with the ONNX session.
    
    Args:
        session: ONNX InferenceSession object
        input_tensor: numpy array with shape (1, 3, H, W), float32 normalized [0,1]
    
    Returns:
        Numpy array of model outputs.
    """
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor})
    
    # Usually outputs[0] contains detection results - adjust if your model differs
    return outputs[0]
