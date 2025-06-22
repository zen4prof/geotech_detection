# recommendation.py

from feedback_data import get_feedback_info

feedback_info = get_feedback_info()
feedback_dict = feedback_info["data"]

def generate_recommendations(detections, class_names):
    """
    Given a list of detected classes (strings) and all class_names,
    return a list of human-readable recommendations.

    Args:
        detections (list of str): List of detected fault class labels.
        class_names (list of str): List of all known fault class names.

    Returns:
        list of str: Recommendations corresponding to detected faults.
    """
    recommendations = []
    for detected_class in detections:
        if detected_class in class_names and detected_class in feedback_dict:
            rec = feedback_dict[detected_class]["recommendation"]
            severity = feedback_dict[detected_class]["severity"]
            priority = feedback_dict[detected_class]["priority"]
            recommendations.append(
                f"{detected_class.title()} (Severity: {severity}, Priority: {priority}): {rec}"
            )
        else:
            # Fallback message if class not found in feedback dict
            recommendations.append(f"{detected_class.title()}: No recommendation available.")
    return recommendations
