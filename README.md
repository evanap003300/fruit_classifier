# Fruit Image Classification

A machine learning project that classifies images of various fruits using TensorFlow and OpenCV.

## Overview

This project implements a convolutional neural network (CNN) to classify images of 8 different types of fruits:
- Apple
- Banana 
- Cherry
- Grapes
- Kiwi
- Mango
- Orange
- Strawberry

The model uses image preprocessing techniques to enhance classification accuracy and is built with TensorFlow and Keras.

## Features

- **Image Preprocessing**: Converts images to grayscale, applies Gaussian blur, and resizes them to a uniform size
- **CNN Architecture**: Uses a sequential model with convolutional and pooling layers
- **Model Persistence**: Saves and loads trained models for future use
- **Easy Testing**: Simple interface to test new fruit images

## Requirements

- Python 3.x
- TensorFlow 2.x
- OpenCV (cv2)
- NumPy

```bash
pip install tensorflow opencv-python numpy
```

## Project Structure

```
├── input-images/          # Raw fruit images organized by fruit type
│   ├── apple-fruit/
│   ├── banana-fruit/
│   └── ...
├── processed-images/      # Preprocessed images used for training
│   ├── apple-fruit/
│   ├── banana-fruit/
│   └── ...
├── test-images/           # Images for testing the model
│   └── test-banana.jpg
├── fruit_classifier.py    # Main Python script
└── fruit_classifier_model.keras  # Saved trained model
```

## Usage

### 1. Preprocess Images

```python
# Uncomment the following line in main() to preprocess your images
# preprocess()
```

This function takes raw images from the input-images directory, applies preprocessing (grayscale conversion, Gaussian blur, resizing), and saves them to the processed-images directory.

### 2. Train the Model

```python
# Uncomment the following line in main() to train the model
# train()
```

This function loads the preprocessed images, creates and trains a CNN model, and saves it as "fruit_classifier_model.keras".

### 3. Test the Model

```python
# This is already called in main()
test()
```

This function loads the trained model and classifies a test image, printing the predicted fruit type.

## Model Architecture

The CNN model has the following architecture:
- Convolutional layer with 32 filters (3x3) and ReLU activation
- Max pooling layer (2x2)
- Convolutional layer with 32 filters (3x3) and ReLU activation
- Max pooling layer (2x2)
- Flatten layer
- Dense layer with 32 neurons and ReLU activation
- Output layer with 8 neurons (one for each fruit) and softmax activation
