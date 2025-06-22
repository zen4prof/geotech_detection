import streamlit as st
from model_inference import load_models, run_pytorch_inference, run_onnx_inference
from parser import parse_model_output
from recommendation import generate_recommendations
import pandas as pd
import io

st.set_page_config(page_title="Fault Detection & Recommendations", layout="wide")

# Load models once
@st.cache_resource
def load_all_models():
    pt_model = load_models('best.pt')
    onnx_session = load_models('best.onnx')
    return pt_model, onnx_session

pt_model, onnx_session = load_all_models()

st.title("Fault Detection & Maintenance Recommendations")
st.write("Upload images for fault detection. Choose model, filter results, and get detailed recommendations.")

uploaded_files = st.file_uploader("Upload one or multiple images", accept_multiple_files=True, type=["jpg","jpeg","png"])

model_choice = st.radio("Select Model for Inference", ["PyTorch (.pt)", "ONNX (.onnx)"])

score_filter = st.slider("Minimum Confidence Score", 0.0, 1.0, 0.3, 0.05)
severity_filter = st.multiselect("Filter by Severity", ["Low", "Moderate", "High", "Critical"], default=["Moderate", "High", "Critical"])
priority_filter = st.multiselect("Filter by Priority", ["Low", "Medium", "High", "Urgent"], default=["Medium", "High", "Urgent"])

if uploaded_files:
    all_results = []
    for uploaded_file in uploaded_files:
        image_bytes = uploaded_file.read()
        
        # Run inference
        if model_choice == "PyTorch (.pt)":
            detections = run_pytorch_inference(pt_model, image_bytes)
        else:
            detections = run_onnx_inference(onnx_session, image_bytes)
        
        # Parse detections
        parsed = parse_model_output(detections)
        
        # Generate recommendations
        recs = generate_recommendations(parsed)
        
        # Add image info for summary table
        for r in recs:
            r['image'] = uploaded_file.name
        
        all_results.extend(recs)
    
    # Convert to DataFrame for easy filtering and display
    df = pd.DataFrame(all_results)
    
    # Normalize filter inputs (case insensitive)
    severity_filter = [s.lower() for s in severity_filter]
    priority_filter = [p.lower() for p in priority_filter]
    
    # Filter DataFrame by score, severity, priority
    filtered_df = df[
        (df['confidence'] >= score_filter) & 
        (df['severity'].str.lower().isin(severity_filter)) & 
        (df['priority'].str.lower().isin(priority_filter))
    ]
    
    st.subheader("Detection & Recommendation Summary")
    st.write(f"Showing {len(filtered_df)} faults detected after applying filters.")
    
    st.dataframe(filtered_df[['image','fault','confidence','severity','priority','recommendation']].reset_index(drop=True))
    
    # Export to CSV
    csv_buffer = io.StringIO()
    filtered_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="Download Report as CSV",
        data=csv_buffer.getvalue(),
        file_name="fault_detection_report.csv",
        mime="text/csv"
    )
    
    # Summary dashboard: count faults by severity and priority
    st.subheader("Summary Dashboard")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Faults by Severity")
        severity_counts = filtered_df['severity'].value_counts()
        st.bar_chart(severity_counts)
    with col2:
        st.write("Faults by Priority")
        priority_counts = filtered_df['priority'].value_counts()
        st.bar_chart(priority_counts)

else:
    st.info("Upload images to begin fault detection.")
