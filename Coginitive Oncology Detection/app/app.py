from flask import Flask, request, jsonify, render_template
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image # type: ignore
import cv2

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model(r'C:\Users\krish\OneDrive\Desktop\Coginitive Oncology Detection\braintumor.h5')  # Path to your saved model

# Define labels for predictions
labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image uploaded'})

    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'})

    try:
        # Save the uploaded file to a temporary location
        file_path = os.path.join('static', file.filename)
        file.save(file_path)

        # Preprocess the image
        img = cv2.imread(file_path)
        img = cv2.resize(img, (150, 150))  # Resize to match model input
        img_array = np.array(img) / 255.0  # Normalize
        img_array = img_array.reshape(1, 150, 150, 3)  # Add batch dimension

        # Predict using the model
        predictions = model.predict(img_array)
        predicted_label = labels[np.argmax(predictions)]

        # Clean up the temporary file
        os.remove(file_path)

        return jsonify({'success': True, 'prediction': predicted_label})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
