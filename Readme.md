## Land-Use Classifer app
<p>A Streamlit-based satellite land-use classification web app trained on the EuroSAT dataset using an EfficientNetV2-B0 CNN with two-phase transfer learning.</p>
<p><b>Live Demo:</b> https://landuseclassifierapp-mmccnndg4395rmcrxosj63.streamlit.app/</p>
<hr/>

### Progam Ouput :-
<h4> Image Input :-</h4>
<img width="1357" height="745" alt="image" src="https://github.com/user-attachments/assets/3bca8f71-2c0f-4f8e-9a01-7b556ff049f0" />
<h4> Model prediction :-</h4>
<img width="998" height="839" alt="image" src="https://github.com/user-attachments/assets/47f552b7-a98e-4e5e-bd07-a58572172c19" />

<hr/>

### 🛠️ Tech Stack :-

- Python
- TensorFlow / Keras
- Streamlit
- FPDF

<hr/>

### Model Information :-

-  CNN Architecture: EfficientNetV2-B0
-  Reason: EfficientNetV2-B0 is a modern CNN-architecture optimized for both accuracy and training efficiency
-  Training: Two-phase transfer learning
-  Phase 1: Learned dataset-specific class boundaries by training the classifier head
-  Phase 2: Refined high-level features through partial backbone fine-tuning
-  Validation Accuracy: ~98%

<hr/>

### Dataset Information :-

- Dataset: EuroSAT
- Source: Sentinel-2 satellite imagery (ESA)
- Image Type: RGB satellite images
- Image Resolution: 64x64 pixels
- Total Images: ~27,000
- No. of Classes 10 land-use categories
- Class Labels Annual Crop, Forest, Herbaceous Vegetation, Highway, Industrial Buildings, Pasture, Permanent Crop, Residential Buildings, River, SeaLake

<hr>

### Streamlit UI FLow :-
<img width="3860" height="5323" alt="Mermaid Chart - Create complex, visual diagrams with text -2026-02-02-132136" src="https://github.com/user-attachments/assets/b48c25c9-fc09-44b8-b6e4-7bade7bfa74f" />
