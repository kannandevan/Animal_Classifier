# Description: This script trains an AI model using images from the 'data' folder.

import tensorflow as tf
import os

# --- Configuration ---
# Set the image dimensions and batch size.
IMG_HEIGHT = 180
IMG_WIDTH = 180
BATCH_SIZE = 5 # How many images to process at a time.

# Set the path to the data folder.
data_dir = 'data'

# --- Data Loading ---
# Load images from the directory.
# 80% of the data will be used for training, and 20% for validation.
print("Loading data...")
train_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

# Get the class names (e.g., 'cats', 'dogs').
class_names = train_dataset.class_names
print("Classes found:", class_names)

# --- Performance Optimization ---
# Cache the data in memory to speed up training.
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# --- Model Building ---
# Create a simple AI model (Convolutional Neural Network).
print("Building the model...")
model = tf.keras.Sequential([
    # Rescale pixel values from the 0-255 range to the 0-1 range.
    tf.keras.layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),

    # Layers for learning patterns from the images.
    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.2), # To reduce overfitting.

    # Layers for making the final decision.
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(class_names))
])

# --- Model Compilation ---
# Prepare the model for training.
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Print a summary of the model.
model.summary()

# --- Model Training ---
# Train the model using the loaded data.
print("\nStarting model training...")
epochs = 10 # How many times to go through the entire dataset.
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs
)
print("Training complete!")

# --- Save the Model ---
# Save the trained model to a file.
model_filename = 'animal_classifier_model.keras'
model.save(model_filename)
print(f"Model saved as '{model_filename}'.")
