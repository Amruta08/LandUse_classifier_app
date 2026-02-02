import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from fpdf import FPDF
import io

# Title
st.title("Land-Use Classifier")

# Cache the model once its loaded to avoid reloading
@st.cache_resource(show_spinner=False)
def load_model():
    return tf.keras.models.load_model("eurosat_efficientnetv2b0.keras")

model = load_model()

# class names
CLASS_NAMES = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
               'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

# class info
CLASS_INFO = {
    "AnnualCrop": "Fields used for seasonal crops and annual agricultural cycles.",
    "Forest": "Natural woodland areas and managed forestry.",
    "HerbaceousVegetation": "Areas covered with grass and non-woody plants.",
    "Highway": "Major road networks and highways visible from above.",
    "Industrial": "Factories, warehouses, and large commercial structures.",
    "Pasture": "Grasslands used primarily for livestock grazing.",
    "PermanentCrop": "Long-term agricultural land like orchards and vineyards.",
    "Residential": "Dense housing and urban neighborhoods.",
    "River": "Natural flowing water bodies and streams.",
    "SeaLake": "Large static bodies of water, including coastal areas and lakes."
}


# Function to Preprocess image
def preprocess_image(img):
    if not isinstance(img, Image.Image): # if image is not pillow image object then convert it
        img = Image.open(img)
    image = img.convert("RGB")
    image = image.resize((224, 224))
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0) #expand dim from (224,224,3) to (1,224,224,3), since model expects a batch image for one image
    return image_array


# Function to generate pdf report
def generate_pdf_report(pred_class, confidence, class_info, input_image):
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("Arial", "B", 20)
    pdf.cell(200, 15, txt="Classification Report", ln=True, align="C")
    pdf.ln(10)
    
     # Input Image
    img_buffer = io.BytesIO()
    input_image.save(img_buffer, format="PNG")
    temp_img_path = "temp_report_img.png"
    input_image.save(temp_img_path)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, txt="Input Image", ln=True, align="L")
    pdf.image(temp_img_path, x=10, y=None, w=100)
    pdf.ln(10)
    
    # Results
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, txt="Results", ln=True, align="L")
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Predicted Class: {pred_class}", ln=True)
    pdf.cell(200, 10, txt=f"Confidence Score: {confidence:.2%}", ln=True)
    pdf.ln(5)
    
    # Class Description
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt='Description', ln=True)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 10, txt=class_info)
    
    return pdf.output(dest='S').encode('latin-1')


# Session state for sample images
if "selected_image" not in st.session_state:
    st.session_state.selected_image = None
    

# Function to clear session state of image file handler once reset button is pressed
def reset_uploader():
    if "image_uploader" in st.session_state:
        del st.session_state["image_uploader"]
    st.session_state.selected_image = None
  

# Tab layout  
tab1, tab2, tab3 = st.tabs(["Classifier", "Model & Dataset Information", "Use cases"])
     
with tab1:        
    # Image upload
    st.subheader("📂Upload a satellite image")
    input_img = st.file_uploader("Choose an image", 
                        type=["jpg", "jpeg", "png"],
                        key="image_uploader")

    # Sample images
    st.subheader("📂Or try a sample image:-")
    SAMPLE_IMAGES={
        "img 1":"sample_images/img1.jpg",
        "img 2":"sample_images/img2.jpg",
        "img 3":"sample_images/img3.jpg",
        "img 4":"sample_images/img4.jpg",
        "img 5":"sample_images/img5.jpg",
        "img 6":"sample_images/img6.jpg",
    }

    # Display the sample images
    cols = st.columns(len(SAMPLE_IMAGES))
    for col, (label, path) in zip(cols, SAMPLE_IMAGES.items()):
        with col:
            img = Image.open(path)
            st.image(img, width="stretch")
            if st.button(f"Use {label}", key=label):
                st.session_state.selected_image = img

        
    # Image handling
    image = None
    if input_img is not None:
        image = Image.open(input_img)
    elif st.session_state.selected_image is not None:
        image = st.session_state.selected_image


    # Prediction from image
    if image is not None:
        st.subheader("📷 Input Image:-")
        display_image = image.resize((300,300))
        st.image(display_image, caption="Uploaded Satellite Image")
        
        predict_btn = st.button("Predict Land-Use", icon="🔎")
        
        if predict_btn:
            img_input = preprocess_image(image)
            preds = model.predict(img_input)[0]
            pred_idx = np.argmax(preds)
            
            pred_class = CLASS_NAMES[pred_idx]
            confidence = preds[pred_idx]
            
            # Display results
            st.subheader("📜 Prediction results:-")
            col1, col2 = st.columns(2)
            
            # Prediction results with light css
            with col1:
                st.markdown(
                    f"""
                    <div style="background-color:#0f172a; padding:14px; border-radius:10px; border-left:6px solid #22c55e; margin-top:16px;">
                        <h4 style="margin:0;">Prediction</h4>
                        <p style="font-size:18px; margin:4px 0;">
                            <b>Class:</b> {pred_class}
                        </p>
                        <p style="font-size:18px; margin:4px 0;">
                            <b>Confidence:</b> {confidence:.2%}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
            )
            
            # Class info with light css
            with col2:
                st.markdown(
                    f"""
                    <div style="background-color:#0f172a; padding:14px; border-radius:10px; border-left:6px solid #22c55e; margin-top:16px;">
                        <h4 style="margin:0;">Class Information</h4>
                        <p style="margin-top:8px;">
                            {CLASS_INFO[pred_class]}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            # Class Probabilities
            st.text("") #empty line break
            st.subheader("📊 Class Probabilities:-")
            prob_dict = {
                CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))
            }
            st.bar_chart(prob_dict , horizontal=True)
            
            
            # Download report button
            report = generate_pdf_report(pred_class, confidence, CLASS_INFO[pred_class], display_image)
            st.download_button(label="📥 Download PDF Report", data=report, file_name=f"EuroSAT_Report_{pred_class}.pdf", mime="application/pdf")
        
            
            # Clear btn
            clear_btn = st.button("Clear", icon="🧹", on_click=reset_uploader)
            if clear_btn:
                st.rerun()
          
          
  

with tab2:
    st.subheader("📘 Model Information")
    st.markdown("""
    **Key points:**
    - **CNN Architecture:** EfficientNetV2-B0
    - **Reason:** EfficientNetV2-B0 is a modern CNN-architecture optimized for both accuracy and training efficiency
    - **Training:** Two-phase transfer learning
    - **Phase 1:** Learned dataset-specific class boundaries by training the classifier head
    - **Phase 2:** Refined high-level features through partial backbone fine-tuning
    - **Validation Accuracy:** ~98%
    """)
    st.markdown("**Confusion Matrix:**")
    st.image("images/cm.png", caption="Confusion Matrix", width="stretch")
    
    st.markdown("**Classification Report:**")
    st.image("images/cf_report.png", caption="Classification Report", width="stretch")
    

    st.subheader("📊 Dataset Information:")
    st.markdown("""
    **Key points:**
    - **Dataset:** EuroSAT
    - **Source:** Sentinel-2 satellite imagery (ESA)
    - **Image Type:** RGB satellite images
    - **Image Resolution:** 64x64 pixels
    - **Total Images:** ~27,000
    - **No. of Classes** 10 land-use categories
    - **Class Labels** Annual Crop, Forest, Herbaceous Vegetation, Highway, Industrial Buildings, Pasture, Permanent Crop, Residential Buildings, River, SeaLake
    """)
    
    st.markdown("**Classes Images:**")
    st.image("images/class_images.png", caption="Class Images", width="stretch")

    st.write("**Information about Classes:**")
    for cls, desc in CLASS_INFO.items():
        st.markdown(f"- **{cls}** — {desc}")



with tab3:
    st.subheader("💡 Real-World Use Cases")
    st.markdown("""
    **Key points:**
    - **Primary Task:** Land use and land cover classification from satellite imagery
    - **Land Monitoring:** Large-scale monitoring of agricultural, urban, forest, and water regions
    - **Change Detection:** Identifying land-use changes over time using multi-temporal satellite images
    - **Urban Development:** Tracking expansion or removal of residential and industrial areas
    - **Environmental Monitoring:** Detecting deforestation and vegetation changes
    - **Map Assistance:** Verifying, correcting, and updating geographical maps (e.g., OpenStreetMap)
    """)



