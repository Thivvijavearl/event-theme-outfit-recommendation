import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import joblib
import augmentation as au  # custom augmentation module
import plots  # plotting utilities (plot_history, evaluate_model, etc.)
import tensorflow as tf
from PIL import Image

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

print(tf.__version__)

# -------------------------
# Set random seed for reproducibility
np.random.seed(42)

# -------------------------
# Paths for dataset
image_dir = os.path.join(BASE_DIR, 'data', 'raw')
labels_path = os.path.join(BASE_DIR, 'data', 'colors_dataset.csv')

# -------------------------
# Load labels CSV (match your CSV headers)
labels_df = pd.read_csv(labels_path)

# Drop rows with missing image paths
labels_df = labels_df[pd.notnull(labels_df['image_path'])]

# Keep only supported image formats

# Validate images and filter labels

valid_image_paths = []
valid_labels = []
for fname, label in zip(labels_df['image_path'], labels_df['label']):
    path = os.path.join(image_dir, str(fname))
    try:
        with Image.open(path) as img:
            img.verify()
        valid_image_paths.append(path)
        valid_labels.append(label)
    except Exception:
        print(f"Skipping invalid image: {path}")

image_paths = valid_image_paths
labels = np.array(valid_labels)

# Verify and filter valid image paths
## Removed duplicate/incorrect filtering

# -------------------------
# Encode class labels (convert species names to integers)
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
num_classes = len(label_encoder.classes_)

# -------------------------
# Split into training and validation sets (stratified by label)

# Ensure image_paths and labels are the same length and use them for split
assert len(image_paths) == len(valid_labels), f"image_paths ({len(image_paths)}) and labels ({len(valid_labels)}) must match."
X_train, X_val, y_train, y_val = train_test_split(
    image_paths,
    valid_labels,
    test_size=0.2,
    random_state=42,
    stratify=valid_labels
)

# Create DataFrames for ImageDataGenerator
train_df = pd.DataFrame({'filename': X_train, 'class': y_train})
val_df   = pd.DataFrame({'filename': X_val, 'class': y_val})

# -------------------------
# Image parameters
img_width, img_height = 224, 224
batch_size = 32

# -------------------------
# Data augmentation for training + rescaling for validation
train_generator = au.train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='filename',
    y_col='class',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

val_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col='filename',
    y_col='class',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

# -------------------------
# Load pre-trained VGG16 (without top FC layers)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
base_model.trainable = False  # Freeze convolutional base

# Add custom classification layers
x = base_model.output
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Build final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile with Adam optimizer
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -------------------------
# Define callbacks
checkpoint = ModelCheckpoint(
    os.path.join(BASE_DIR, 'models', 'color_dominance_vgg16.h5'),
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# -------------------------
# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=10,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    callbacks=[checkpoint, early_stopping]
)

# -------------------------
# Save label encoder for inference
joblib.dump(label_encoder, os.path.join(BASE_DIR, 'models', 'color_label_encoder.pkl'))

# -------------------------
# Evaluate on validation set
loss, accuracy = model.evaluate(validation_generator)
print(f"\nValidation Accuracy: {accuracy * 100:.2f}%")

# Show class index → label mapping
print("\nClass mapping:")
for i, class_name in enumerate(label_encoder.classes_):
    print(f"{i}: {class_name}")

# -------------------------
# Plot training history (accuracy & loss curves)
plots.plot_history(history)
