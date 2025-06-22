# Fault Detection and Recommendation System 🧠⚠️

This Streamlit app uses a trained YOLOv8 model to detect structural/geotechnical faults from uploaded images and provide actionable recommendations based on predefined severity, priority, and expert feedback.

## 🚀 Features

- Upload and analyze infrastructure images
- Detect faults using YOLOv8 (`best.pt` or `best.onnx`)
- Parse and interpret detections
- View structured severity, priority, and remediation suggestions
- Filter results by severity or priority
- Export results as CSV

## 🗂 Project Structure

```bash
├── app.py                 # Streamlit UI
├── model_inference.py     # YOLOv8 inference (PyTorch + ONNX)
├── feedback_data.py       # Fault metadata (recommendations, severity, etc.)
├── parser.py              # Parses model predictions
├── recommendation.py      # Maps faults to recommendations
├── requirements.txt       # Dependencies
└── README.md              # Project info
