# README for Footprint Classification System

### This project is an AI-powered footprint classification system designed to analyze footprint images and predict the foot type. It uses a trained deep learning model built with TensorFlow/Keras and integrates a user-friendly interface using Streamlit. The system provides predictions and actionable suggestions based on the detected foot type.

Features
Footprint Classification:
  Predicts foot types such as Flat Foot, High Arch, Normal Arch, Overpronation, Supination, Toe Walker, Heel Walker, Asymmetrical Footprint, and Irregular Pressure Distribution.

Recommendations:
  Provides tailored suggestions for footwear and lifestyle adjustments based on the predicted foot type.

Visualization:
  Displays confidence scores for predictions using bar charts.

Streamlit Interface:
  Allows users to upload footprint images and view predictions interactively.

Installation
Prerequisites
Python 3.8 or higher

TensorFlow 2.x

Streamlit

Required Python libraries (numpy, Pillow, matplotlib)


Usage
Open the Streamlit app in your browser.
Upload a footprint image (formats: .jpg, .png, .jpeg).
View predictions and confidence scores.
Read suggestions tailored to your foot type.


File Structure
app.py: Main Streamlit application file.
footprint_model.h5: Pre-trained TensorFlow model for classification.
F_CNN.ipynb: Notebook containing model training and evaluation code.
sample_footprints/: Folder containing sample footprint images.


Model Details
Architecture:
  Convolutional Neural Network (CNN) with multiple Conv2D layers followed by MaxPooling2D layers.
  Dense layers for classification with a softmax activation function.

Input:
  Footprint images resized to 224x224 pixels.

Classes:
  9 foot types corresponding to different arch structures and walking patterns.


Troubleshooting
Model Not Loaded:
  Ensure footprint_model.h5 is in the correct directory.

Image Upload Error:
  Verify that the uploaded image is in .jpg, .png, or .jpeg format.

Prediction Error:
  Check if the image is correctly preprocessed (RGB format, resized to 224x224 pixels).

Future Improvements
  Expand dataset for better accuracy across diverse demographics.
  Add functionality for real-time analysis using mobile devices.
  Incorporate additional recommendations based on medical insights.


