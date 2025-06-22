# parser.py
import numpy as np

def parse_detections(predictions, ratio, pad_top, pad_left, conf_threshold=0.25):
    """
    Parse raw model output to structured detections.
    
    Args:
        predictions: np.ndarray of shape (N, 6) [x1, y1, x2, y2, conf, class]
        ratio: scale ratio used during preprocessing
        pad_top: padding top added during preprocessing
        pad_left: padding left added during preprocessing
        conf_threshold: minimum confidence to keep detection
    
    Returns:
        List of dicts:
        [
            {
                'class_id': int,
                'confidence': float,
                'bbox': [x1, y1, x2, y2]  # scaled to original image
            },
            ...
        ]
    """
    detections = []

    # Filter by confidence threshold
    preds = predictions[predictions[:, 4] > conf_threshold]

    for *box, conf, cls in preds:
        x1, y1, x2, y2 = box
        
        # Undo padding and scaling to get box coords in original image space
        x1 = (x1 - pad_left) / ratio
        y1 = (y1 - pad_top) / ratio
        x2 = (x2 - pad_left) / ratio
        y2 = (y2 - pad_top) / ratio

        # Clamp to positive coordinates (optional)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = max(0, x2)
        y2 = max(0, y2)

        detection = {
            "class_id": int(cls),
            "confidence": float(conf),
            "bbox": [x1, y1, x2, y2]
        }
        detections.append(detection)

    return detections
