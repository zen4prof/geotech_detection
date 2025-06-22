import streamlit as st
import numpy as np
from PIL import Image

from model_inference import load_onnx_model, run_onnx_inference
from parser import parse_onnx_output               # <- use the new parser name
from recommendation import generate_recommendations
from feedback_data import get_feedback_info

# -------------------- Model cache --------------------
@st.cache_resource
def load_session():
    return load_onnx_model("best.onnx")

onnx_session = load_session()

# -------------------- Streamlit UI --------------------
st.title("Geotechnical Fault Detection (ONNX)")

uploaded_file = st.file_uploader(
    "Choose an image", type=["png", "jpg", "jpeg"]
)

if uploaded_file:
    image_pil = Image.open(uploaded_file).convert("RGB")
    st.image(image_pil, caption="Uploaded Image", use_column_width=True)

    image_np = np.array(image_pil)

    # ---------- Inference ----------
    outputs, scale, pad_top, pad_left = run_onnx_inference(
        onnx_session, image_np
    )

    # ---------- Parse detections ----------
    detections = parse_onnx_output(
        outputs, scale, pad_top, pad_left, image_np.shape[:2]
    )

    st.subheader("Parsed Detections")
    st.write(detections)

    # ---------- Recommendations ----------
    fb_info = get_feedback_info()
    class_names = fb_info["class_names"]

    recs = generate_recommendations(detections, class_names)

    st.subheader("Recommendations")
    for r in recs:
        st.write(f"- {r}")

    # ---------- Download ----------
    if st.button("Download Recommendations"):
        txt = "\n".join(recs)
        st.download_button(
            "Download as TXT",
            data=txt,
            file_name="recommendations.txt",
            mime="text/plain",
        )
else:
    st.info("Upload an image to start.")
