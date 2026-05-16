import os
import io
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
import cv2
from sklearn.cluster import KMeans
import tensorflow as tf

app = Flask(__name__)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Load your pre-trained event theme detection model (update path as needed)
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'event_theme_model.h5')
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    model = None  # For development without model

# Example event classes (update as per your model)
EVENT_CLASSES = ['wedding', 'party', 'office', 'beach', 'other']

# Example color dataset (expand as needed)
COLOR_DATASET = [
    {'name': 'Red', 'rgb': (255, 0, 0), 'hex': '#FF0000'},
    {'name': 'Green', 'rgb': (0, 255, 0), 'hex': '#00FF00'},
    {'name': 'Blue', 'rgb': (0, 0, 255), 'hex': '#0000FF'},
    {'name': 'White', 'rgb': (255, 255, 255), 'hex': '#FFFFFF'},
    {'name': 'Black', 'rgb': (0, 0, 0), 'hex': '#000000'},
    # Add more colors as needed
]

# Example outfit recommendations (expand as needed)
RECOMMENDATIONS = {
    'wedding': ['Formal dress', 'Suit', 'Gown'],
    'party': ['Cocktail dress', 'Smart casual', 'Trendy outfit'],
    'office': ['Business suit', 'Blazer', 'Formal shirt'],
    'beach': ['Beachwear', 'Sundress', 'Shorts'],
    'other': ['Casual wear', 'Comfortable outfit']
}

def preprocess_image(image_bytes):
    # Resize and preprocess image for model
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224))  # Update size as per your model
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def detect_event_theme(image_bytes):
    if model is None:
        return 'other'  # Fallback if model not loaded
    arr = preprocess_image(image_bytes)
    preds = model.predict(arr)
    idx = np.argmax(preds)
    return EVENT_CLASSES[idx]

def extract_dominant_colors(image_bytes, n_colors=3):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img_np = np.array(img)
    img_np = img_np.reshape((-1, 3))
    kmeans = KMeans(n_clusters=n_colors, n_init=10)
    kmeans.fit(img_np)
    centers = kmeans.cluster_centers_.astype(int)
    return [tuple(center) for center in centers]

def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb

def match_colors(dominant_colors):
    matched = []
    for color in dominant_colors:
        min_dist = float('inf')
        best = None
        for entry in COLOR_DATASET:
            dist = np.linalg.norm(np.array(color) - np.array(entry['rgb']))
            if dist < min_dist:
                min_dist = dist
                best = entry
        matched.append({'name': best['name'], 'rgb': best['rgb'], 'hex': best['hex']})
    return matched

@app.route('/recommend', methods=['POST'])
def recommend():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    image_file = request.files['image']
    image_bytes = image_file.read()

    # 1. Detect event theme
    event_theme = detect_event_theme(image_bytes)

    # 2. Extract dominant colors
    dominant_colors = extract_dominant_colors(image_bytes)
    matched_colors = match_colors(dominant_colors)

    # 3. Recommend outfits
    outfits = RECOMMENDATIONS.get(event_theme, RECOMMENDATIONS['other'])

    return jsonify({
        'event_theme': event_theme,
        'dominant_colors': [
            {'rgb': color, 'hex': rgb_to_hex(color)} for color in dominant_colors
        ],
        'matched_colors': matched_colors,
        'outfit_recommendations': outfits
    })

if __name__ == '__main__':
    app.run(debug=True)
