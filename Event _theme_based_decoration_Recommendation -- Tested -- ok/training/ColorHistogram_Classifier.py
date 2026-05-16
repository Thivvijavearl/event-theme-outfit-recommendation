import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from PIL import Image
import glob
from sklearn.model_selection import cross_val_score

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print("Creating color histogram-based classifier...")

# Data directory
DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')

# Color classes based on folder names
color_classes = ['Black', 'Blue', 'Green', 'Purble', 'Red', 'Red-Yellow', 'Yellow']

def extract_color_histogram(image_path, bins=32):
    """Extract color histogram features from image"""
    try:
        # Open image
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
        print(f"Error processing {image_path}: {e}")
        return None

print("Loading images and extracting color features...")

# Create dataset
image_paths = []
labels = []
features = []

for color_class in color_classes:
    class_dir = os.path.join(DATA_DIR, color_class)
    if os.path.exists(class_dir):
        # Get all image files
        class_images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.avif', '*.webp']:
            class_images.extend(glob.glob(os.path.join(class_dir, ext)))

        print(f"Processing {len(class_images)} images for class '{color_class}'")

        for img_path in class_images:
            # Extract features
            feature_vector = extract_color_histogram(img_path)
            if feature_vector is not None:
                features.append(feature_vector)
                labels.append(color_class)
                image_paths.append(img_path)

print(f"\nTotal processed images: {len(features)}")
print("Class distribution:")
for color_class in color_classes:
    count = labels.count(color_class)
    print(f"  {color_class}: {count} images")

# Convert to numpy arrays
X = np.array(features)
y = np.array(labels)

print(f"\nFeature matrix shape: {X.shape}")

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print(f"Classes: {label_encoder.classes_}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\nTraining set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest classifier
print("\nTraining Random Forest classifier...")
rf_classifier = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)

rf_classifier.fit(X_train_scaled, y_train)

# Train SVM classifier
print("Training SVM classifier...")
svm_classifier = SVC(
    kernel='rbf',
    C=10,
    gamma='scale',
    probability=True,
    random_state=42
)

svm_classifier.fit(X_train_scaled, y_train)

# Evaluate models
print("\nEvaluating models...")

# Random Forest evaluation
rf_train_acc = rf_classifier.score(X_train_scaled, y_train)
rf_test_acc = rf_classifier.score(X_test_scaled, y_test)

print(f"Random Forest - Train accuracy: {rf_train_acc:.4f}")
print(f"Random Forest - Test accuracy: {rf_test_acc:.4f}")

# SVM evaluation
svm_train_acc = svm_classifier.score(X_train_scaled, y_train)
svm_test_acc = svm_classifier.score(X_test_scaled, y_test)

print(f"SVM - Train accuracy: {svm_train_acc:.4f}")
print(f"SVM - Test accuracy: {svm_test_acc:.4f}")

# Choose the better model
if rf_test_acc > svm_test_acc:
    best_model = rf_classifier
    model_name = "Random Forest"
    best_accuracy = rf_test_acc
else:
    best_model = svm_classifier
    model_name = "SVM"
    best_accuracy = svm_test_acc

print(f"\nBest model: {model_name} with {best_accuracy:.4f} test accuracy")

# Cross-validation
cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Detailed classification report
y_pred = best_model.predict(X_test_scaled)
print(f"\nClassification Report for {model_name}:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix for {model_name}:")
print(cm)

# Save the best model and preprocessing objects
joblib.dump(best_model, os.path.join(BASE_DIR, 'models', 'color_histogram_classifier.pkl'))
joblib.dump(scaler, os.path.join(BASE_DIR, 'models', 'color_histogram_scaler.pkl'))
joblib.dump(label_encoder, os.path.join(BASE_DIR, 'models', 'color_histogram_label_encoder.pkl'))

print("\nModel saved as 'models/color_histogram_classifier.pkl'")
print("Scaler saved as 'models/color_histogram_scaler.pkl'")
print("Label encoder saved as 'models/color_histogram_label_encoder.pkl'")

# Test on a few examples
print("\nTesting on sample images...")
test_samples = [
    (os.path.join(BASE_DIR, 'data', 'raw', 'Blue', 'Blue00001.jpg'), 'Blue'),
    (os.path.join(BASE_DIR, 'data', 'raw', 'Red', 'Red00001.jpg'), 'Red'),
    (os.path.join(BASE_DIR, 'data', 'raw', 'Yellow', 'Yellow00001.jpg'), 'Yellow'),
    (os.path.join(BASE_DIR, 'data', 'raw', 'Purble', 'Purble00006.jpg'), 'Purble')
]

for img_path, expected in test_samples:
    if os.path.exists(img_path):
        features = extract_color_histogram(img_path)
        if features is not None:
            features_scaled = scaler.transform([features])
            prediction = best_model.predict(features_scaled)[0]
            predicted_class = label_encoder.inverse_transform([prediction])[0]
            confidence = np.max(best_model.predict_proba(features_scaled)) * 100
            status = "✓" if predicted_class == expected else "✗"
            print(f"{status} {expected} -> {predicted_class} ({confidence:.1f}%)")

print("\nColor histogram-based classifier training completed!")