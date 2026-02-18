**🌿 Plant Leaf Disease Detection System**

An AI-powered Plant Leaf Disease Detection System built using Deep Learning (CNN) that identifies plant diseases from leaf images in real time.
The system helps farmers, researchers, and agricultural professionals detect diseases early and take preventive action to reduce crop loss.

**📌 Project Overview**

Plant diseases significantly impact agricultural productivity and food security. Traditional disease diagnosis requires expert knowledge and manual inspection, which is time-consuming and not always accessible.
This project leverages Convolutional Neural Networks (CNN) to automatically classify plant leaf diseases through image recognition and provides predictions through an interactive web application.

The system allows users to:

Upload plant leaf images
Detect diseases instantly
View prediction results with high accuracy
Enable early crop disease management

**🚀 Features**

✅ Real-time plant disease prediction
✅ Deep Learning–based image classification
✅ Supports 38 plant disease classes
✅ User-friendly Streamlit web interface
✅ High accuracy model performance
✅ Data augmentation for robust predictions
✅ Cloud-deployable architecture

**🧠 Model Details**

Architecture: Convolutional Neural Network (CNN)
Frameworks: TensorFlow & Keras
Input Image Size: 128 × 128 RGB
Dataset Size: 87,867 images
Classes: 38 plant disease categories
**Dataset Split:**
Training — 70%
Validation — 15%
Testing — 15%
**CNN Components**
Convolutional Layers – Feature extraction
Max Pooling Layers – Dimensionality reduction
Dropout Layers – Prevent overfitting
Fully Connected Layers – Classification
Softmax Output – Multi-class prediction

**📊 Model Performance**
Metric	Result
Training Accuracy	98%
Validation Accuracy	93%
Testing Accuracy	92%
Average F1 Score	0.91
Prediction Time	< 1 second

The model demonstrates strong generalization across unseen plant leaf images. 


**🏗️ System Architecture**
**Workflow**

Image Upload (User Interface)
Image Preprocessing
CNN Model Inference
Disease Classification
Prediction Display with Confidence
The application integrates:
Frontend: Streamlit
Backend: TensorFlow Model
Deployment: Cloud-based inference system

**🖥️ Tech Stack**
Programming & ML
Python
TensorFlow
Keras
NumPy
OpenCV
Scikit-learn
Web & Deployment
Streamlit
FastAPI (Inference Handling)
Cloud Deployment (GCP Compatible)
Docker (Scalable Deployment)

**📂 Project Structure**
plant-disease-detection/
│
├── dataset/
├── model/
│   └── trained_plant_disease_model.h5
├── app.py
├── train_model.py
├── prediction.py
├── requirements.txt
└── README.md
