from feedback_data import feedback_data

def parse_model_output(detections):
    """
    Parses the model detection output and enriches with feedback data.

    Parameters:
    - detections: list of dicts, each dict with keys: 'label', 'confidence', 'bbox'
      Example:
      [
        {"label": "block loss", "confidence": 0.85, "bbox": [x1, y1, x2, y2]},
        {"label": "crack on ashpat", "confidence": 0.78, "bbox": [x1, y1, x2, y2]},
      ]

    Returns:
    - List of dicts with enriched information including severity, recommendation, priority.
    """
    results = []

    for det in detections:
        label = det.get("label")
        confidence = det.get("confidence", 0)
        bbox = det.get("bbox", [])

        feedback = feedback_data.get(label, None)

        if feedback:
            results.append({
                "label": label,
                "confidence": confidence,
                "bbox": bbox,
                "score": feedback["score"],
                "severity": feedback["severity"],
                "recommendation": feedback["recommendation"],
                "priority": feedback["priority"],
            })
        else:
            # For unknown detections, provide a default fallback
            results.append({
                "label": label,
                "confidence": confidence,
                "bbox": bbox,
                "score": None,
                "severity": "Unknown",
                "recommendation": "No recommendation available.",
                "priority": "Low"
            })

    return results
