import os
import tensorflow as tf
from tensorflow.keras.models import load_model
preprocess_input = tf.keras.applications.efficientnet_v2.preprocess_input
import numpy as np
import gradio as gr
# Load your trained model
model = load_model("model/best_model.keras")

# Class labels
class_names = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']

# Classify image and return top prediction with styled output
def classify_image(img):
    try:
        if img is None:
            return "<div style='color:red;'>No image received. Please upload or capture an image.</div>"

        img = img.resize((224, 224))
        img_array = np.array(img, dtype=np.float32)

        if img_array.shape != (224, 224, 3):
            return f"<div style='color:red;'>Unexpected input shape: {img_array.shape}</div>"

        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)[0]

        predicted_class_index = np.argmax(prediction)
        predicted_class_name = class_names[predicted_class_index]
        confidence = float(prediction[predicted_class_index])

        # Return styled HTML result
        return f"""
        <div style='text-align: center; font-size: 28px; font-weight: bold; color: #00FFAA; font-family: Arial, sans-serif;'>
            Prediction: {predicted_class_name}<br>
            <span style='font-size: 20px; color: #888;'>Confidence: {confidence:.2%}</span>
        </div>
        """
    except Exception as e:
        return f"<div style='color:red;'>Error during prediction: {str(e)}</div>"

# 3. Build the Gradio Interface
# Gradio interface with webcam and upload support
iface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil", sources=["upload", "webcam"],),  #
    outputs=gr.HTML(),
    title="Waste Material Classifier",
    description="Upload or capture an image and this AI model will classify it into one of six recyclable waste categories.",
    article="""
    <br><br>
    <div style='display: flex; justify-content: center; gap: 60px; color: #b0b0b0; font-size: 14px; flex-wrap: wrap;'>
        <div>Powered by EfficientNetV2</div>
        <div>Built with TensorFlow and Gradio</div>
    </div>
   
    """
)

iface.launch()
