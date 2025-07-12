# Description: Predicts a new image using the trained model.

import tensorflow as tf
import numpy as np
import sys

# --- Configuration ---
IMG_HEIGHT = 180
IMG_WIDTH = 180
# The class names obtained during training.
class_names = ['cats', 'dogs'] 
MODEL_PATH = 'animal_classifier_model.keras'

# --- Function to predict image ---
def predict_image(image_path):
    """Loads an image and predicts if it's a cat or a dog."""
    try:
        # Load the saved model.
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        print(f"Error: Could not load model file '{MODEL_PATH}'. Please run the 'train.py' script first.")
        print(e)
        return

    # Load the image.
    try:
        img = tf.keras.utils.load_img(
            image_path, target_size=(IMG_HEIGHT, IMG_WIDTH)
        )
    except FileNotFoundError:
        print(f"Error: Image file not found at '{image_path}'.")
        return

    # Convert the image into a format the model understands.
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    # Make the prediction.
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    # Display the result.
    print(
        "\nThis image is most likely a {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )

# --- Main execution ---
if __name__ == "__main__":
    # Check if an image path was provided in the command line.
    if len(sys.argv) > 1:
        image_to_predict = sys.argv[1]
        predict_image(image_to_predict)
    else:
        # If no path is provided, show an example of how to use the script.
        print("Usage: python predict.py <path_to_image.jpg>")
        print("Example: python predict.py my_dog_photo.jpg")
