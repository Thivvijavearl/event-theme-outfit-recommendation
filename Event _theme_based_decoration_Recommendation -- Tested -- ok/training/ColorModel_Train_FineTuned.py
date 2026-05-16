import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
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

# Split data with stratification
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
batch_size = 16

# Enhanced data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=False,  # Colors might be orientation-sensitive
    fill_mode='nearest',
    brightness_range=[0.8, 1.2],
    channel_shift_range=30.0  # Color augmentation
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

print("\nBuilding improved model with fine-tuning...")

# Load VGG16 base model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Initially freeze all layers
for layer in base_model.layers:
    layer.trainable = False

# Unfreeze the last few convolutional blocks for fine-tuning
for layer in base_model.layers[-4:]:  # Unfreeze last 4 layers
    layer.trainable = True

# Add custom classification head with Global Average Pooling
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Better than Flatten for conv nets
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Create the model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model with lower learning rate for fine-tuning
model.compile(
    optimizer=Adam(learning_rate=1e-5),  # Lower learning rate for fine-tuning
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Model summary:")
model.summary()

# Callbacks
checkpoint = ModelCheckpoint(
    os.path.join(BASE_DIR, 'models', 'color_dominance_vgg16_finetuned.h5'),
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

# Calculate steps per epoch
train_steps = len(train_df) // batch_size
val_steps = len(val_df) // batch_size

print(f"\nTraining steps per epoch: {train_steps}")
print(f"Validation steps per epoch: {val_steps}")

# Train the model
print("\nStarting fine-tuning training...")
history = model.fit(
    train_generator,
    steps_per_epoch=train_steps,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=val_steps,
    callbacks=[checkpoint, early_stopping, reduce_lr],
    verbose=1
)

# Save the final model
model.save(os.path.join(BASE_DIR, 'models', 'color_dominance_vgg16_finetuned_final.h5'))
print("Fine-tuned model saved as 'models/color_dominance_vgg16_finetuned_final.h5'")

# Save label encoder
joblib.dump(label_encoder, os.path.join(BASE_DIR, 'models', 'color_label_encoder.pkl'))
print("Label encoder saved as 'models/color_label_encoder.pkl'")

# Save training history
import pickle
with open('color_training_history_finetuned.pkl', 'wb') as f:
    pickle.dump(history.history, f)
print("Training history saved as 'color_training_history_finetuned.pkl'")

# Plot training results
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Fine-tuned Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Fine-tuned Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('color_training_plots_finetuned.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nFine-tuning completed!")
print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
print(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}")

# Evaluate on validation set
print("\nEvaluating fine-tuned model on validation set...")
val_loss, val_accuracy = model.evaluate(validation_generator, steps=val_steps, verbose=1)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")