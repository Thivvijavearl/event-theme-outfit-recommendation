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
import tensorflow as tf
from PIL import Image
import glob

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

print("TensorFlow version:", tf.__version__)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Data directory
data_dir = os.path.join(BASE_DIR, 'data', 'raw')

# Color classes based on folder names
color_classes = ['Black', 'Blue', 'Green', 'Purble', 'Red', 'Red-Yellow', 'Yellow']

print("Loading images from DATA folder...")

# Create dataset from folder structure
image_paths = []
labels = []

for color_class in color_classes:
    class_dir = os.path.join(data_dir, color_class)
    if os.path.exists(class_dir):
        # Get all image files (jpg, jpeg, png, avif, webp)
        pattern = os.path.join(class_dir, '*')
        class_images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.avif', '*.webp']:
            class_images.extend(glob.glob(os.path.join(class_dir, ext)))

        print(f"Found {len(class_images)} images for class '{color_class}'")

        for img_path in class_images:
            # Verify image is valid
            try:
                with Image.open(img_path) as img:
                    img.verify()
                image_paths.append(img_path)
                labels.append(color_class)
            except Exception as e:
                print(f"Skipping invalid image: {img_path} - {e}")

print(f"\nTotal valid images: {len(image_paths)}")
print("Class distribution:")
for color_class in color_classes:
    count = labels.count(color_class)
    print(f"  {color_class}: {count} images")

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
num_classes = len(label_encoder.classes_)

print(f"\nClasses: {label_encoder.classes_}")
print(f"Number of classes: {num_classes}")

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    image_paths,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

print(f"\nTraining set: {len(X_train)} images")
print(f"Validation set: {len(X_val)} images")

# Create dataframes for generators
train_df = pd.DataFrame({'filename': X_train, 'class': y_train})
val_df = pd.DataFrame({'filename': X_val, 'class': y_val})

# Image parameters
img_width, img_height = 224, 224
batch_size = 16  # Smaller batch size for better training

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Only rescaling for validation
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='filename',
    y_col='class',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

validation_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col='filename',
    y_col='class',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

print("\nBuilding model...")

# Load VGG16 base model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Freeze the convolutional base
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification head
x = base_model.output
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Create the model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Model summary:")
model.summary()

# Callbacks
checkpoint = ModelCheckpoint(
    os.path.join(BASE_DIR, 'models', 'color_dominance_vgg16.h5'),
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# Calculate steps per epoch
train_steps = len(train_df) // batch_size
val_steps = len(val_df) // batch_size

print(f"\nTraining steps per epoch: {train_steps}")
print(f"Validation steps per epoch: {val_steps}")

# Train the model
print("\nStarting training...")
history = model.fit(
    train_generator,
    steps_per_epoch=train_steps,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=val_steps,
    callbacks=[checkpoint, early_stopping],
    verbose=1
)

# Save the final model
model.save(os.path.join(BASE_DIR, 'models', 'color_dominance_vgg16_final.h5'))
print("Fine-tuned model saved as 'models/color_dominance_vgg16_final.h5'")

# Save label encoder
joblib.dump(label_encoder, os.path.join(BASE_DIR, 'models', 'color_label_encoder.pkl'))
print("Label encoder saved as 'models/color_label_encoder.pkl'")

# Save training history
import pickle
with open('color_training_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)
print("Training history saved as 'color_training_history.pkl'")

# Plot training results
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('color_training_plots_improved.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nTraining completed!")
print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")

# Evaluate on validation set
print("\nEvaluating model on validation set...")
val_loss, val_accuracy = model.evaluate(validation_generator, steps=val_steps, verbose=1)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")