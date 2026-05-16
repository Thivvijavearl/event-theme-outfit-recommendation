
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Image dimensions
img_width, img_height = 224, 224
batch_size = 32#training on one image at a time, the model sees 32 images in parallel per training step.

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,# Normalize pixel values (0–255 → 0–1)
    rotation_range=20, # Randomly rotate images up to 20 degrees
    width_shift_range=0.2,# Shift image width up to 20%
    height_shift_range=0.2, # Shift image height up to 20%
    shear_range=0.2,# Shear (slant) transformations
    zoom_range=0.2,# Random zoom in/out
    horizontal_flip=True,# Flip images left-right
    fill_mode='nearest') # Fill in new pixels after rotation/shift
