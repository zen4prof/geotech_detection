def generate_recommendations(detections, class_names=None, confidence_threshold=0.3):
    """
    Generate recommendations based on detected objects.

    Args:
        detections (list of dict): List of detections with keys ['bbox', 'confidence', 'class']
        class_names (list of str, optional): List mapping class indices to human-readable names
        confidence_threshold (float): Minimum confidence to consider a detection for recommendations

    Returns:
        list of str: Recommendations messages
    """
    recommendations = []

    if not detections:
        return ["No faults or issues detected."]

    for det in detections:
        conf = det['confidence']
        cls = det['class']

        if conf < confidence_threshold:
            continue

        class_label = class_names[cls] if class_names and cls < len(class_names) else f"class_{cls}"

        # Example recommendations — customize based on your classes and domain
        if class_label.lower() in ['crack', 'fault', 'damage']:
            recommendations.append(
                f"Detected potential {class_label} with confidence {conf:.2f}. Recommend further inspection and repair."
            )
        elif class_label.lower() == 'erosion':
            recommendations.append(
                f"Erosion detected with confidence {conf:.2f}. Consider preventative measures to avoid structural issues."
            )
        else:
            recommendations.append(
                f"Detected {class_label} with confidence {conf:.2f}. Review as necessary."
            )

    if not recommendations:
        recommendations.append("No significant detections above confidence threshold.")

    return recommendations
