import numpy as np

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45):
    """
    Perform Non-Maximum Suppression on inference results.
    
    Args:
        prediction: numpy array, shape (num_boxes, 6) 
                    columns: [x1, y1, x2, y2, confidence, class]
        conf_thres: confidence threshold to filter boxes
        iou_thres: IOU threshold for NMS
    
    Returns:
        List of filtered boxes after NMS (each box as [x1, y1, x2, y2, conf, class])
    """
    # Filter out low confidence detections
    mask = prediction[:, 4] >= conf_thres
    prediction = prediction[mask]

    if not prediction.shape[0]:
        return []

    # Coordinates
    boxes = prediction[:, :4]
    scores = prediction[:, 4]
    classes = prediction[:, 5]

    # Compute areas of boxes
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    # Sort by confidence
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        # Compute IoU of the remaining boxes with the box i
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
