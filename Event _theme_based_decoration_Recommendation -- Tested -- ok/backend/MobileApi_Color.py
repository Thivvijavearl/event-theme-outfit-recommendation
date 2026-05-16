from flask import Flask, request, jsonify
import os
import numpy as np
from PIL import Image
import joblib
import tempfile
import traceback
import json
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Base paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "models", "color_histogram_classifier.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "color_histogram_scaler.pkl")
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "models", "color_histogram_label_encoder.pkl")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "backend", "temp_uploads")
CONTROL_DATA_PATH = os.path.join(BASE_DIR, "data", "color_controls.json")

# Allowed extensions
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

def allowed_file(filename):
    """Check if uploaded file has an allowed extension"""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Create uploads directory if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the model, scaler, and label encoder
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found.")
if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(f"Scaler file '{SCALER_PATH}' not found.")
if not os.path.exists(LABEL_ENCODER_PATH):
    raise FileNotFoundError(f"Label encoder file '{LABEL_ENCODER_PATH}' not found.")

classifier = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)
print("Color histogram classifier, scaler, and label encoder loaded successfully!")

def extract_color_histogram(image_path, bins=32):
    """Extract color histogram features from image"""
    try:
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)

        # Calculate histograms for each channel
        hist_r = np.histogram(img_array[:, :, 0], bins=bins, range=(0, 256))[0]
        hist_g = np.histogram(img_array[:, :, 1], bins=bins, range=(0, 256))[0]
        hist_b = np.histogram(img_array[:, :, 2], bins=bins, range=(0, 256))[0]

        # Concatenate histograms
        features = np.concatenate([hist_r, hist_g, hist_b])

        # Normalize
        features = features.astype(float)
        features = features / np.sum(features) if np.sum(features) > 0 else features

        return features

    except Exception as e:
        print(f"Error extracting features from {image_path}: {e}")
        return None

def predict_species(img_path):
    """Predict dominant color (class) from image path using histogram classifier"""
    if not os.path.exists(img_path):
        return None, 0.0

    try:
        # Extract color histogram features
        features = extract_color_histogram(img_path)
        if features is None:
            return None, 0.0

        # Reshape for prediction
        features = features.reshape(1, -1)

        # Scale features
        features_scaled = scaler.transform(features)

        # Make prediction
        prediction_idx = classifier.predict(features_scaled)[0]
        predicted_class = label_encoder.inverse_transform([prediction_idx])[0]

        # Get confidence
        prediction_probabilities = classifier.predict_proba(features_scaled)[0]
        confidence = float(np.max(prediction_probabilities) * 100)

        return predicted_class, confidence

    except Exception as e:
        print(f"Error during prediction: {e}")
        traceback.print_exc()
        return None, 0.0


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": os.path.exists(MODEL_PATH),
        "label_encoder_loaded": os.path.exists(LABEL_ENCODER_PATH),
        "endpoint_type": "direct_image_upload"
    })


@app.route("/occasions", methods=["GET"])
def get_occasions():
    return jsonify({"occasions": control_data.get("occasions", [])})


# Load control data from JSON
if not os.path.exists(CONTROL_DATA_PATH):
    raise FileNotFoundError(f"Control data file '{CONTROL_DATA_PATH}' not found.")

with open(CONTROL_DATA_PATH, "r") as f:
    control_data = json.load(f)

# Load dataset-driven recommendations if generated from new CSV data
DATASET_CONTROL_DATA_PATH = os.path.join(BASE_DIR, 'data', 'color_dataset_recommendations.json')
if os.path.exists(DATASET_CONTROL_DATA_PATH):
    with open(DATASET_CONTROL_DATA_PATH, 'r', encoding='utf-8') as f:
        dataset_control_data = json.load(f)
    # Use only dataset occasions and color mappings
    control_data['occasions'] = dataset_control_data.get('occasions', [])
    for key, value in dataset_control_data.items():
        if key == 'occasions':
            continue
        control_data[key] = value
    print(f"Loaded dataset-derived control data from '{DATASET_CONTROL_DATA_PATH}'")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        print("Prediction request received")
        if "image" not in request.files:
            return jsonify({"error": "No image file provided in request"}), 400

        image_file = request.files["image"]

        if image_file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        if not allowed_file(image_file.filename):
            return jsonify({"error": "Only .jpg and .jpeg files are allowed"}), 400

        # Save uploaded file
        file_ext = os.path.splitext(image_file.filename)[1].lower() or ".jpg"
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext, dir=UPLOAD_FOLDER)
        image_file.save(temp_file.name)
        temp_file.close()

        # Prediction
        predicted_species, confidence = predict_species(temp_file.name)
        print(f"Prediction result: {predicted_species}, {confidence}")

        # Cleanup temp file
        try:
            os.remove(temp_file.name)
        except Exception as e:
            print(f"Warning: could not delete temp file {temp_file.name}: {e}")

        if predicted_species is None:
            return jsonify({"error": "Prediction failed"}), 500

        # Get occasion from request (optional)
        occasion = request.form.get("occasion", "Wedding")  # Default to first occasion
        
        # Validate occasion against available options
        available_occasions = control_data.get("occasions", [])
        if occasion not in available_occasions:
            occasion = available_occasions[0] if available_occasions else "Wedding"

        # Lookup color data
        color_info = control_data.get(predicted_species, {
            "description": "No color data available.",
            "dress_pattern_description": "Suitable patterns and styles for this dominant color.",
            "recommendations": {}
        })

        # Pull best-matching outfit colors (complementary/neutral) from control data
        matching_colors = color_info.get("matching_colors", [])
        suitable_colors = color_info.get("suitable_colors", {}).get(occasion, [])
        if not suitable_colors:
            suitable_colors = matching_colors

        # Use occasion-specific recommendations directly from control_data
        recommendations = color_info.get("recommendations", {})
        men_recommendations = recommendations.get(occasion, {}).get("men", [])
        women_recommendations = recommendations.get(occasion, {}).get("women", [])

        # Enhance description with general styling logic if we have a matching palette
        description_text = color_info.get("description", "")
        if matching_colors:
            description_text += (
                " In styling, it's usually better to choose complementary, neutral, or contrasting colors "
                "rather than wearing the exact same shade as the event background; this makes the wearer "
                "stand out in photos and keeps the look balanced."
            )

        result = {
            "predicted_species": predicted_species,
            "confidence": round(confidence, 2),
            "description": description_text,
            "dress_pattern_description": color_info.get("dress_pattern_description", ""),
            "matching_colors": matching_colors,
            "suitable_colors": suitable_colors,
            "occasion": occasion,
            "men_recommendations": men_recommendations,
            "women_recommendations": women_recommendations,
            "status": "success"
        }

        return jsonify(result)

    except Exception as e:
        print(f"Server error: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Server error: {str(e)}"}), 500


if __name__ == "__main__":
    print("Starting Color Prediction API with direct image upload support...")
    app.run(host="0.0.0.0", port=5002, debug=True)
