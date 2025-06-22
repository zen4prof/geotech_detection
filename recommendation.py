# recommendation.py

from feedback_data import feedback_data

def generate_recommendations(parsed_detections):
    """
    Takes parsed detections and returns a detailed list of feedback for each unique detection label.

    Parameters:
        parsed_detections (list of str): List of fault labels detected in the image.

    Returns:
        list of dicts: Each dict contains 'fault', 'severity', 'score', 'recommendation', and 'priority'.
    """
    recommendations = []

    seen = set()
    for label in parsed_detections:
        if label not in seen:
            seen.add(label)
            feedback = feedback_data.get(label.lower())
            if feedback:
                recommendations.append({
                    "fault": label,
                    "severity": feedback["severity"],
                    "score": feedback["score"],
                    "recommendation": feedback["recommendation"],
                    "priority": feedback["priority"]
                })
            else:
                recommendations.append({
                    "fault": label,
                    "severity": "Unknown",
                    "score": "N/A",
                    "recommendation": "No recommendation available for this fault.",
                    "priority": "Unknown"
                })

    return recommendations
