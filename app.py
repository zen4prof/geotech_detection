import streamlit as st
from model_inference import load_onnx_model, run_onnx_inference
from parser import parse_detections
from recommendation import generate_recommendations
from feedback_data import get_feedback_info
from PIL import Image
import numpy as np

@st.cache_resource
def load_models():
    onnx_session = load_onnx_model('best.onnx')
    return onnx_session

def main():
    st.title("Geotechnical Fault Detection")
    st.write("Upload images to detect and get recommendations on geotechnical faults.")

    onnx_session = load_models()

    uploaded_file = st.file_uploader("Choose an image", type=['png', 'jpg', 'jpeg'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        # Show uploaded image
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Convert to numpy array
        img_np = np.array(image)

        # Preprocess image to meet model input size, e.g., 640x640
        # (you can adjust this depending on your model)
        input_size = 640
        h, w = img_np.shape[:2]
        scale = min(input_size / w, input_size / h)

        # Resize while keeping aspect ratio
        new_w, new_h = int(w * scale), int(h * scale)
        resized_img = np.array(image.resize((new_w, new_h)))

        # Pad to (input_size, input_size)
        pad_w = input_size - new_w
        pad_h = input_size - new_h

        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        # Create padded image with zeros (black)
        img_padded = np.zeros((input_size, input_size, 3), dtype=np.uint8)
        img_padded[pad_top:pad_top+new_h, pad_left:pad_left+new_w, :] = resized_img

        # Run ONNX inference
        preds = run_onnx_inference(onnx_session, img_padded)

        # Parse detections with correct args
        conf_thres = 0.25
        detections = parse_detections(preds, scale, pad_top, pad_left, conf_threshold=conf_thres)

        # Show raw detections (optional)
        st.subheader("Raw Detections")
        st.write(detections)

        # Generate recommendations
        feedback_info = get_feedback_info()
        class_names = feedback_info.get('class_names', [])
        recommendations = generate_recommendations(detections, class_names)

        st.subheader("Recommendations")
        for rec in recommendations:
            st.write("- " + rec)

        # Optional download button for recommendations text
        if st.button("Download Recommendations"):
            rec_text = "\n".join(recommendations)
            st.download_button(
                label="Download as TXT",
                data=rec_text,
                file_name="recommendations.txt",
                mime="text/plain"
            )

if __name__ == "__main__":
    main()
