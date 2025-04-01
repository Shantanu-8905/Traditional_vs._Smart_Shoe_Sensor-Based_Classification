import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load the updated model
model = tf.keras.models.load_model('footprint_model.h5')

# Class names corresponding to foot types
class_names = [
    "Flat Foot", "High Arch", "Normal Arch", "Overpronation", "Supination",
    "Toe Walker", "Heel Walker", "Asymmetrical Footprint", "Irregular Pressure Distribution"
]

def preprocess_image(img):
    # Convert RGBA to RGB (remove alpha channel)
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    
    # Resize to match model input size
    img = img.resize((224, 224))  # Updated to 224x224 pixels
    
    # Convert image to numpy array
    img_array = np.array(img)

    # Ensure the image is 3D (H, W, C) and has 3 channels
    if img_array.shape[-1] != 3:
        raise ValueError("Image does not have 3 color channels (RGB). Please upload a valid image.")

    # Expand dimensions for model input
    img_array = np.expand_dims(img_array, axis=0)

    # Normalize pixel values to [0, 255] as EfficientNet expects this range
    img_array = img_array.astype(np.float32)  # Ensure the array is of type float32

    return img_array

# Streamlit UI
st.title("AI Footprint Classifier")
st.write("Upload a footprint image to predict the foot type and receive recommendations.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Footprint Image", use_column_width=True)
    
    try:
        img_array = preprocess_image(img)
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        predicted_label = class_names[predicted_class]
        
        # Display prediction results
        st.subheader(f"Predicted Foot Type: {predicted_label}")
        
        # Debugging: Show probabilities as a bar chart
        fig, ax = plt.subplots()
        ax.bar(class_names, prediction[0], color='skyblue')
        ax.set_ylabel("Confidence Score")
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        st.pyplot(fig)
        
        # Suggestions based on prediction
        suggestions = {
            "Flat Foot": "Consider using arch support insoles to prevent foot fatigue.",
            "High Arch": "Try using cushioning insoles to reduce pressure on the foot.",
            "Normal Arch": "Maintain a balanced lifestyle with proper footwear.",
            "Overpronation": "Use stability shoes to correct excessive inward foot rolling.",
            "Supination": "Consider using orthotic insoles for better support.",
            "Toe Walker": "Check for neurological factors and use soft sole shoes.",
            "Heel Walker": "Look into heel cushions to prevent discomfort.",
            "Asymmetrical Footprint": "Consult a podiatrist for gait analysis.",
            "Irregular Pressure Distribution": "Custom orthotics may help redistribute pressure."
        }
        
        st.write(f"**Suggestion:** {suggestions.get(predicted_label, 'No suggestion available.')}")
        
    except Exception as e:
        st.error(f"Error during prediction: {e}")

else:
    st.write("Please upload an image to classify the footprint.")
