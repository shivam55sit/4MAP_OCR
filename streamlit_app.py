import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
import os
import io
from PIL import Image

# Import custom modules
import image_crop
import improved_ocr_extractor

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Pentacam/Oculyzer Data Extractor",
    page_icon="👁️",
    layout="wide"
)

# Custom Styling
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        color: #1f77b4;
        padding-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3rem;
        background-color: #1f77b4;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<h1 class="main-title">👁️ Pentacam/Oculyzer Data Extractor</h1>', unsafe_allow_html=True)

# ============================================================================
# SIDEBAR / UPLOAD
# ============================================================================
with st.sidebar:
    st.header("📂 Upload Scan")
    uploaded_file = st.file_uploader("Choose a 4-Map scan image", type=['jpg', 'jpeg', 'png', 'bmp'])
    
    if uploaded_file:
        st.success("File uploaded successfully!")
        if st.button("🧹 Clear All"):
            if 'ocr_data' in st.session_state:
                st.session_state['ocr_data'] = None
            st.rerun()

# ============================================================================
# MAIN CONTENT
# ============================================================================
if uploaded_file is not None:
    # Save uploaded file to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        tmp.write(uploaded_file.getvalue())
        temp_path = tmp.name

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📷 Original Image")
        img_original = Image.open(uploaded_file)
        st.image(img_original, use_container_width=True, caption="Uploaded Scan")

    with col2:
        st.subheader("📐 Detected Crops")
        # Perform crop for preview
        regions = image_crop.crop_left_panel(temp_path)
        
        if regions is not None:
            # Display each region
            for name, img in regions.items():
                st.write(f"**Section: {name.replace('_', ' ').title()}**")
                # Convert BGR to RGB for streamlit
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                st.image(img_rgb, use_container_width=True)
        else:
            st.error("Failed to detect standard dimensions for cropping.")
            st.info("Supported dimensions include: 740x1200, 758x1200, 820x1200, 838x1200, 840x1200, 858x1200, 904x1200, 910x1200, 940x1200.")

    st.divider()

    # ============================================================================
    # EXTRACTION SECTION
    # ============================================================================
    st.subheader("🔍 Data Extraction")
    
    # Initialize session state for data if not present
    if 'ocr_data' not in st.session_state:
        st.session_state['ocr_data'] = None

    if st.button("🚀 Start OCR Extraction"):
        with st.spinner("Processing with EasyOCR... please wait (10-20 seconds)"):
            try:
                # Extract metrics directly from the regions dictionary
                data = improved_ocr_extractor.extract_metrics_from_cropped(regions)
                st.session_state['ocr_data'] = data
                
                if data:
                    st.success(f"Successfully extracted {len(data)} fields!")
                else:
                    st.warning("No data could be extracted. Please check the crop preview.")
                    
            except Exception as e:
                st.error(f"Error during extraction: {e}")
                st.exception(e)
    
    # Display results from session state if available
    if st.session_state['ocr_data']:
        data = st.session_state['ocr_data']
        
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Sort columns
        def sort_key(col):
            if col == 'image_name': return '000'
            if col.startswith('cf_'): return '200' + col
            if col.startswith('cb_'): return '300' + col
            # Group Pachy Metrics
            pachy_prefixes = ['PupilCenter', 'PachyApex', 'ThinnestLoc', 'KMax']
            for p in pachy_prefixes:
                if col.startswith(p):
                    return '400' + col
            if col in ['First Name', 'Last Name', 'ID', 'DOB', 'Date of Birth', 'Exam Date', 'Eye', 'Time']:
                return '100' + col
            return '900' + col
        
        sorted_cols = sorted(df.columns, key=sort_key)
        df = df[sorted_cols]
        
        # Display Results
        st.markdown("### 📊 Extracted Metrics")
        st.dataframe(df, use_container_width=True)
        
        # Download Section
        st.markdown("### 💾 Export Data")
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Metrics')
        processed_data = output.getvalue()
        
        st.download_button(
            label="📥 Download Results as Excel",
            data=processed_data,
            file_name=f"extracted_metrics_{uploaded_file.name.split('.')[0]}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    # Final Cleanup of original temp
    if os.path.exists(temp_path):
        os.remove(temp_path)

else:
    # Landing page info
    st.info("👈 Please upload a Pentacam or Oculyzer 4-Maps scan in the sidebar to begin.")
    
    st.markdown("""
    ### How it works:
    1. **Upload**: Select an image of a clinical 4-map scan.
    2. **Crop**: The app automatically identifies the left numeric panel based on image dimensions.
    3. **OCR**: EasyOCR scans the panel sections (Patient Info, Cornea Front/Back, Pachymetry, Global).
    4. **Download**: Review the table and download the results as a ready-to-use Excel file.
    """)
