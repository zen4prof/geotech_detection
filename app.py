# app.py
import streamlit as st
import numpy as np
from PIL import Image

from model_inference import load_onnx_model, run_onnx_inference
from parser import parse_detections                     # wrapper to parse_onnx_output
from recommendation import generate_recommendations
from feedback_data import get_feedback_info             # returns class list + dict


# ------------------------------------------------------------------
# Cache the ONNX session so we load the model only once per session
# ------------------------------------------------------------------
@st.cache_resource
def load_session():
    return load_onnx_model("best.onnx")                 # make sure best.onnx is present


onnx_session = load_session()


# ------------------------------------------------------------------
# Streamlit user interface
# ------------------------------------------------------------------
st.title("Geotechnical Fault Detection 📐")
st.write(
    """
    Upload a site photograph; the app will detect faults using an ONNX model
    and provide severity‐based maintenance recommendations.
    """
)

uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

# Optional confidence filter
conf_thres = st.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.05)

if uploaded_file:
    # ► Display original image
    img_pil = Image.open(uploaded_file).convert("RGB")
    st.image(img_pil, caption="Uploaded image", use_column_width=True)

    # ► Convert to NumPy array (BGR for OpenCV-style preprocessing)
    img_np = np.array(img_pil)[:, :, ::-1]  # RGB → BGR

    # ► Inference
    raw_outputs, scale, pad_top, pad_left = run_onnx_inference(
        onnx_session, img_np
    )

    # ► Parse model output → list[dict]
    detections = parse_detections(
        raw_outputs, scale, pad_top, pad_left, img_np.shape[:2],
        conf_thres=conf_thres
    )

    # ► Display detections JSON for debugging
    st.subheader("Detections")
    st.json(detections)

    # ► Build recommendations
    fb_info      = get_feedback_info()
    class_names  = fb_info["class_names"]
    feedback_dict = fb_info["data"]

    recs = generate_recommendations(detections, class_names)

    st.subheader("Recommendations")
    if recs:
        for r in recs:
            st.markdown(f"• {r}")
    else:
        st.write("No detections above confidence threshold.")

    # ► Download recommendations as TXT
    if recs and st.button("Download recommendations"):
        txt = "\n".join(recs)
        st.download_button(
            label="Download TXT",
            data=txt,
            file_name="recommendations.txt",
            mime="text/plain",
        )
else:
    st.info("Please upload an image to begin.")
