import os
import math
import pandas as pd
import joblib
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from PIL import Image

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
image_dir = os.path.join(BASE_DIR, 'data', 'raw')
labels_path = os.path.join(BASE_DIR, 'data', 'colors_dataset.csv')
model_path = os.path.join(BASE_DIR, 'models', 'color_dominance_vgg16.h5')
encoder_path = os.path.join(BASE_DIR, 'models', 'color_label_encoder.pkl')

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
if not os.path.exists(encoder_path):
    raise FileNotFoundError(f"Label encoder not found: {encoder_path}")
if not os.path.exists(labels_path):
    raise FileNotFoundError(f"Labels CSV not found: {labels_path}")

# Load labels CSV
labels_df = pd.read_csv(labels_path)
labels_df = labels_df[pd.notnull(labels_df['image_path'])]

# Build full paths and filter missing images
full_paths = []
classes = []
for fname, label in zip(labels_df['image_path'], labels_df['label']):
    path = os.path.join(image_dir, str(fname))
    if os.path.exists(path):
        try:
            with Image.open(path) as img:
                img.verify()
            full_paths.append(path)
            classes.append(label)
        except Exception:
            print(f"Unreadable image, skipping: {path}")
    else:
        print(f"Missing image, skipping: {path}")

if len(full_paths) == 0:
    raise RuntimeError('No valid images found for evaluation.')

df = pd.DataFrame({'filename': full_paths, 'class': classes})

# Parameters (should match training)
img_size = (224, 224)
batch_size = 32

# Create generator
val_datagen = ImageDataGenerator(rescale=1.0/255)
validation_generator = val_datagen.flow_from_dataframe(
    dataframe=df,
    x_col='filename',
    y_col='class',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Load model and encoder
model = load_model(model_path)
label_encoder = joblib.load(encoder_path)

# Evaluate
steps = math.ceil(len(df) / batch_size)
loss, accuracy = model.evaluate(validation_generator, steps=steps, verbose=1)
print(f"\nEvaluation results on {len(df)} samples:")
print(f"Loss: {loss:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")
