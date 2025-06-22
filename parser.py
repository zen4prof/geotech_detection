# parser.py
import numpy as np

# --------------------------------------------
#   Helper: Non-Maximum Suppression (XYXY)
# --------------------------------------------
def nms(boxes, scores, classes, conf_thres=0.25, iou_thres=0.45):
    """Return indexes of boxes to keep after NMS."""
    keep = []
    idxs = np.where(scores >= conf_thres)[0]
    boxes, scores, classes = boxes[idxs], scores[idxs], classes[idxs]

    while idxs.size:
        i = idxs[0]
        keep.append(i)

        # IoU with the rest
        xx1 = np.maximum(boxes[i, 0], boxes[idxs[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[idxs[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[idxs[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[idxs[1:], 3])

        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_j = (boxes[idxs[1:], 2] - boxes[idxs[1:], 0]) * (boxes[idxs[1:], 3] - boxes[idxs[1:], 1])
        iou = inter / (area_i + area_j - inter + 1e-6)

        idxs = idxs[1:][iou <= iou_thres]

    return keep


# --------------------------------------------
#   Main parser
# --------------------------------------------
def parse_onnx_output(raw_outputs,
                      scale: float,
                      pad_top: int,
                      pad_left: int,
                      orig_shape,
                      conf_thres=0.25,
                      iou_thres=0.45):
    """
    Convert raw ONNX output -> list[dict].
    raw_outputs : outputs[0] from ONNX Runtime, shape (N, 6)
    orig_shape  : (h, w) of original image
    Returns list of {'bbox':[x1,y1,x2,y2],'confidence':c,'class':cls}
    """
    preds = raw_outputs[0]  # first output tensor
    if preds.ndim == 3:      # sometimes (1, N, 6)
        preds = preds[0]

    if preds.size == 0:
        return []

    boxes      = preds[:, :4]
    scores     = preds[:, 4]
    class_ids  = preds[:, 5].astype(int)

    keep_ids = nms(boxes, scores, class_ids, conf_thres, iou_thres)
    boxes, scores, class_ids = boxes[keep_ids], scores[keep_ids], class_ids[keep_ids]

    # de–pad & rescale back to original image coords
    boxes[:, [0, 2]] -= pad_left
    boxes[:, [1, 3]] -= pad_top
    boxes /= scale

    h, w = orig_shape
    boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, w)  # x1,x2
    boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, h)  # y1,y2

    detections = []
    for b, s, c in zip(boxes, scores, class_ids):
        detections.append({
            "bbox": b.tolist(),       # [x1,y1,x2,y2]
            "confidence": float(s),
            "class": int(c)
        })

    return detections


# --------------------------------------------
#   Thin wrapper so app.py can call parse_detections(...)
# --------------------------------------------
def parse_detections(raw_outputs, scale, pad_top, pad_left, orig_shape):
    """Alias for backward-compatibility with app.py"""
    return parse_onnx_output(raw_outputs, scale, pad_top, pad_left, orig_shape)
