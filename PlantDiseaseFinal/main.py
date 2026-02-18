import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import datetime
import uuid
from PIL import Image
import io
import base64
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportLabImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import tempfile
from fpdf import FPDF

# Set page config first - MUST be the first Streamlit command
st.set_page_config(
    page_icon="plant.png",
    page_title="Plant Disease Detection",
    layout="wide",  # This ensures wide layout
    initial_sidebar_state="expanded"  # Makes sure sidebar is expanded by default
)

# Path to the model
MODEL_PATH = "C:/Users/HP/Desktop/Project 2/PlantDiseaseFinal/trained_plant_disease_model.h5"

# Initialize session state for history if it doesn't exist
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=['ID', 'Date', 'Image', 'Prediction'])

# Function to save image as base64 string
def get_image_as_base64(image_file):
    img = Image.open(image_file)
    img = img.resize((100, 100))  # Resize for thumbnail
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

# Function to convert base64 string back to image for PDF
def base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image_stream = io.BytesIO(image_data)
    return Image.open(image_stream)

# Tensorflow Model Prediction
def model_prediction(test_image):
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])  # Convert single image to batch
        predictions = model.predict(input_arr)
        return np.argmax(predictions)  # Return index of max element
    else:
        st.error("Model file not found at: " + MODEL_PATH)
        return None

# Function to add entry to history
def add_to_history(image_file, prediction):
    # Generate a unique ID for this prediction
    unique_id = str(uuid.uuid4())[:8]
    
    # Get current date and time
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Save image as base64 string
    img_str = get_image_as_base64(image_file)
    
    # Create new entry
    new_entry = pd.DataFrame({
        'ID': [unique_id],
        'Date': [current_datetime],
        'Image': [img_str],
        'Prediction': [prediction]
    })
    
    # Append to history
    st.session_state.history = pd.concat([new_entry, st.session_state.history]).reset_index(drop=True)
    
    # Save history to disk (CSV file)
    save_history_to_disk()

# Function to save history to disk
def save_history_to_disk():
    history_file = "plant_disease_history.csv"
    st.session_state.history.to_csv(history_file, index=False)

# Function to load history from disk
def load_history_from_disk():
    history_file = "plant_disease_history.csv"
    try:
        if os.path.exists(history_file):
            history_df = pd.read_csv(history_file)
            st.session_state.history = history_df
    except Exception as e:
        st.error(f"Error loading history: {e}")

# Function to generate PDF report of history - FIXED VERSION
# Add this to your imports
from fpdf import FPDF

def generate_history_pdf_alternative(filtered_df):
    class PDF(FPDF):
        def header(self):
            # Arial bold 15
            self.set_font('Arial', 'B', 15)
            # Title
            self.cell(0, 10, 'Plant Disease Detection History Report', 0, 1, 'C')
            # Line break
            self.ln(10)

    # Instantiate PDF object
    pdf = PDF()
    pdf.add_page()
    
    # Add date
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 10, f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1)
    
    # Add summary
    pdf.cell(0, 10, f"Total entries: {len(filtered_df)}", 0, 1)
    pdf.ln(5)
    
    # Add entries
    for i, entry in filtered_df.iterrows():
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, f"Entry ID: {entry['ID']}", 0, 1)
        
        pdf.set_font('Arial', '', 10)
        pdf.cell(0, 10, f"Date: {entry['Date']}", 0, 1)
        pdf.cell(0, 10, f"Prediction: {entry['Prediction']}", 0, 1)
        
        pdf.cell(0, 10, "Image information stored but not displayed in PDF for compatibility reasons.", 0, 1)
        pdf.ln(10)
    
    # Create a temporary file for the PDF
    temp_filename = tempfile.mktemp(suffix='.pdf')
    
    # Save the PDF
    pdf.output(temp_filename)
    
    return temp_filename

# Load history at the start of the app
load_history_from_disk()

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition", "History"])

# Main Page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpeg"
    st.image(image_path, use_container_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍

    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.
    4. **History:** Check your past analyses in the **History** page and export reports in CSV or PDF format.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.
    - **Persistent History:** Keep track of all your plant disease analyses.
    - **Export Options:** Download your history in CSV or PDF format with images for record-keeping.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.

    ### Diseases We Can Detect
    Our model is trained to detect various plant diseases, including but not limited to:
    - **Apple Diseases:** Apple Scab, Black Rot, Cedar Apple Rust, Healthy
    - **Corn Diseases:** Cercospora Leaf Spot, Common Rust, Northern Leaf Blight, Healthy
    - **Tomato Diseases:** Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites (Two-spotted Spider Mite), Target Spot, Yellow Leaf Curl Virus, Tomato Mosaic Virus, Healthy
    - **Potato Diseases:** Early Blight, Late Blight, Healthy
    - **Grape Diseases:** Black Rot, Esca (Black Measles), Leaf Blight (Isariopsis Leaf Spot), Healthy
    - **Peach Diseases:** Bacterial Spot, Healthy
    - **Strawberry Diseases:** Leaf Scorch, Healthy
    - **Pepper Diseases:** Bacterial Spot, Healthy
    - **Blueberry Diseases:** Healthy
    - **Soybean Diseases:** Healthy
    - **Raspberry Diseases:** Healthy
    - **Squash Diseases:** Powdery Mildew, Healthy
    """)


# About Page
elif app_mode == "About":
    st.header("About")
    st.markdown("""
    #### About Dataset
    This dataset consists of approximately 87,867 RGB images of healthy and diseased crop leaves, categorized into 38 different classes. 
    The dataset is divided as follows:
    - **Training Set:** 61,490 images (70%)
    - **Validation Set:** 13,164 images (15%)
    - **Test Set:** 13,213 images (15%)

    #### Project Team
    This project is developed by:
    - **Komaravolu Srirama Chaitanya Murthy**

    I was dedicated to creating an efficient and accurate plant disease recognition system to help in protecting crops and ensuring a healthier harvest.
    """)


# Disease Recognition Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    
    if test_image:
        st.image(test_image, use_container_width=True)
    
        # Predict button
        if st.button("Predict"):
            # Store a copy of the file for later use
            test_image_copy = test_image
            
            result_index = model_prediction(test_image)

            if result_index is not None:
                # Class names for the predictions
                class_name = [
                    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
                    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 
                    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 
                    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 
                    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 
                    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 
                    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 
                    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 
                    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                    'Tomato___healthy'
                ]
                prediction = class_name[result_index]
                st.success(f"Model is predicting it's a {prediction}")
                
                # Add to history
                add_to_history(test_image_copy, prediction)
                
                st.info("This result has been saved to your history. You can view all past detections in the History page.")
            else:
                st.error("Prediction failed, please check the model file.")

# History Page
elif app_mode == "History":
    st.header("Detection History")
    
    if st.session_state.history.empty:
        st.info("No detection history available. Make some predictions in the Disease Recognition page.")
    else:
        # Add a search/filter option
        search_term = st.text_input("Filter by plant or disease type:", "")
        
        # Add date range filter
        st.write("Filter by date range:")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start date", datetime.datetime.now() - datetime.timedelta(days=30))
        with col2:
            end_date = st.date_input("End date", datetime.datetime.now())
        
        # Convert string dates to datetime for filtering
        filtered_df = st.session_state.history.copy()
        filtered_df['Date_Object'] = pd.to_datetime(filtered_df['Date'])
        
        # Apply filters
        if search_term:
            filtered_df = filtered_df[filtered_df['Prediction'].str.contains(search_term, case=False)]
        
        start_datetime = datetime.datetime.combine(start_date, datetime.time.min)
        end_datetime = datetime.datetime.combine(end_date, datetime.time.max)
        filtered_df = filtered_df[(filtered_df['Date_Object'] >= start_datetime) & 
                                 (filtered_df['Date_Object'] <= end_datetime)]
        
        # Drop helper column
        filtered_df = filtered_df.drop(columns=['Date_Object'])
        
        # Display history in a nice format
        st.write(f"Showing {len(filtered_df)} records")
        
        # Use columns to create a grid layout
        for i in range(0, len(filtered_df), 3):
            cols = st.columns(3)
            for j in range(3):
                if i+j < len(filtered_df):
                    with cols[j]:
                        entry = filtered_df.iloc[i+j]
                        st.write(f"**ID:** {entry['ID']}")
                        st.write(f"**Date:** {entry['Date']}")
                        st.write(f"**Prediction:** {entry['Prediction']}")
                        st.image(f"data:image/jpeg;base64,{entry['Image']}", caption="Plant Image")
                        st.divider()
        
        # Export section
        st.subheader("Export Options")
        col1, col2 = st.columns(2)
        
        # Export as CSV
        with col1:
            if not filtered_df.empty:
                # Prepare a version of the dataframe without images for CSV export
                export_df = filtered_df[['ID', 'Date', 'Prediction']]
                csv = export_df.to_csv(index=False)
                st.download_button(
                    label="Download as CSV",
                    data=csv,
                    file_name="plant_disease_history.csv",
                    mime="text/csv",
                )
        
        # Export as PDF
        with col2:
            if not filtered_df.empty:
                if st.button("Generate PDF Report"):
                    with st.spinner('Generating PDF report...'):
                        try:
                            pdf_path = generate_history_pdf_alternative(filtered_df)
                            
                            # Read PDF file
                            with open(pdf_path, "rb") as pdf_file:
                                pdf_bytes = pdf_file.read()
                            
                            # Provide download button
                            st.download_button(
                                label="Download PDF Report",
                                data=pdf_bytes,
                                file_name="plant_disease_history_report.pdf",
                                mime="application/pdf"
                            )
                            
                            # Clean up
                            os.unlink(pdf_path)
                        except Exception as e:
                            st.error(f"Error generating PDF: {str(e)}")
        
        # Add option to clear history
        st.subheader("Clear History")
        if st.button("Clear History"):
            if st.session_state.history.empty:
                st.warning("History is already empty.")
            else:
                st.session_state.history = pd.DataFrame(columns=['ID', 'Date', 'Image', 'Prediction'])
                save_history_to_disk()
                st.success("History cleared successfully!")
                st.rerun()  # Updated from st.experimental_rerun()