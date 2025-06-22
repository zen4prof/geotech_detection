import streamlit as st
from model_inference import load_onnx_model, run_onnx_inference
from parser import parse_detections
from recommendation import generate_recommendations
from feedback_data import get_feedback_info
import numpy as np
from PIL import Image
import io

# Cache model loading for performance
@st.cache_resource
def load_models():
    onnx_session = load_onnx_model('best.onnx')
    return onnx_session

def main():
    st.title("Geotechnical Fault Detection")
    st.write("Upload images to detect and get recommendations on geotechnical faults.")

    # Load ONNX model once
    onnx_session = load_models()

    uploaded_file = st.file_uploader("Choose an image", type=['png', 'jpg', 'jpeg'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Convert PIL image to numpy array for model input
        img_array = np.array(image)

        # Run inference
        preds = run_onnx_inference(onnx_session, img_array)

        # Parse detections
        detections = parse_detections(preds)

        # Show raw detections (optional)
        st.subheader("Raw Detections")
        st.write(detections)

        # Generate recommendations
        class_names = get_feedback_info().get('class_names', None)  # Adjust based on your feedback_data
        recommendations = generate_recommendations(detections, class_names)

        st.subheader("Recommendations")
        for rec in recommendations:
            st.write("- " + rec)

        # Optionally: Export recommendations as text
        if st.button("Download Recommendations"):
            recommendation_text = "\n".join(recommendations)
            st.download_button(
                label="Download as TXT",
                data=recommendation_text,
                file_name="recommendations.txt",
                mime="text/plain"
            )

if __name__ == "__main__":
    main()
