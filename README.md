# Fashion MNIST Image Classification with FastAPI and TensorFlow

This project showcases a convolutional neural network (CNN) model for classifying fashion images using the Fashion MNIST dataset. The trained model is integrated into a FastAPI application for real-time predictions.

## Getting Started

To run this project, ensure Python is installed on your system. Install the required dependencies by executing:

`pip install -r requirements.txt
`


## Training the Model

The model is trained using TensorFlow and Keras. We utilize the Fashion MNIST dataset, comprising 60,000 training images and 10,000 test images. To train the model, execute:


# python train_model.py


This script loads the dataset, preprocesses images, constructs the CNN model, compiles it with the Adam optimizer and categorical cross-entropy loss, and trains it for 10 epochs.

## Running the FastAPI Application

Start the FastAPI application for predictions. Run:

# uvicorn app:app --reload


This command launches the FastAPI server on [http://127.0.0.1:8000](http://127.0.0.1:8000).

## Making Predictions

Make predictions by sending POST requests to the `/predict` endpoint with an image file attached.

