# Cat & Dog Image Classifier

A simple deep learning project built with Python and TensorFlow to classify images of cats and dogs. This is a beginner-friendly project to understand the fundamentals of image classification using Convolutional Neural Networks (CNNs).

## About The Project

This project contains two main Python scripts:
* `train.py`: To train a new image classification model from scratch using the images you provide.
* `predict.py`: To use the trained model to predict whether a new image is of a cat or a dog.

## Getting Started

Follow these steps to get the project running on your local machine.

### 1. Prerequisites: Install Python and Libraries

First, ensure you have Python (version 3.8 or newer) installed on your system. You can download it from [python.org](https://www.python.org/).

Once Python is installed, open your terminal or command prompt and install the necessary libraries by running this command:
```bash
pip install tensorflow numpy pillow
```

### 2. Clone the Repository

Next, clone this repository to your local machine using the following git command:
```bash
git clone [https://github.com/kannandevan/Animal_Classifier.git](https://github.com/kannandevan/Animal_Classifier.git)
cd Animal_Classifier
```

### 3. (Important) Add More Images for Better Accuracy

This repository already contains a `data` folder with a few sample images of cats and dogs. However, for the AI model to be accurate, it needs to learn from many more examples.

**Before you start training, it is highly recommended to add more images:**
* Add more cat images to the `data/cats` folder.
* Add more dog images to the `data/dogs` folder.

The more images you add (e.g., 50-100 images for each category), the better your model's predictions will be.

## How to Run the Project

Follow these steps to use the project.

### 1. Train the Model

To train the AI model with the images in the `data` folder, run the `train.py` script from your terminal:
```bash
python train.py
```
**What to Expect:**
You will see the training progress in your terminal. For each "epoch" (training cycle), it will show the accuracy and loss, which indicates how well the model is learning.
```
Epoch 1/10
8/8 [==============================] - 5s 461ms/step - loss: 0.7512 - accuracy: 0.5234
...
Training complete!
Model saved as 'animal_classifier_model.keras'.
```
Once finished, it will save the trained model as `animal_classifier_model.keras`.

### 2. Make a Prediction

After the model is trained, you can use `predict.py` to classify a new image. Place an image you want to test in the main project folder (e.g., `my_test_dog.jpg`).
```bash
python predict.py my_test_dog.jpg
```
**What to Expect:**
The script will analyze the image and print the prediction along with its confidence level.
```
1/1 [==============================] - 0s 103ms/step

This image is most likely a dogs with a 80.73 percent confidence.
```

## Project Structure
```
/Animal_Classifier
|
|-- /data
|   |-- /cats
|   |   |-- cat1.jpg
|   |   `-- ...
|   |
|   `-- /dogs
|       |-- dog1.jpg
|       `-- ...
|
|-- train.py
|-- predict.py
|-- .gitignore
`-- README.md
