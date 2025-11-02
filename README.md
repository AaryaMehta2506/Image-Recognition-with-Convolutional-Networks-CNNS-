Data Science Intermediate Project
# Image Recognition with Convolutional Networks (CNNS)

This project uses a Convolutional Neural Network (CNN) built with TensorFlow and Keras to classify images from the CIFAR-10 dataset into 10 different categories. It also includes a Streamlit web application that allows users to upload an image and get real-time predictions.

## Dataset
Dataset: https://www.kaggle.com/datasets/fedesoriano/cifar10-python-in-csv    (train.csv, test.csv files was downloaded)
CIFAR-10 contains 60,000 32×32 color images across 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.  
The dataset is loaded automatically using:
from tensorflow.keras.datasets import cifar10

## Model Architecture
The CNN consists of convolutional layers with ReLU activation, batch normalization for stable training, max pooling and dropout for regularization, fully connected dense layers, and a softmax output for 10-class classification.  
Data augmentation (random rotation, shift, and flip) is applied to improve generalization.

## Training
The model is trained using:
Optimizer: Adam  
Loss: Sparse Categorical Crossentropy  
Epochs: 50  
Batch size: 64  
After training, the model achieves around 85–90% test accuracy.  
The trained model is saved as:
cifar10_cnn_model_augmented.h5

## Streamlit Web App
The Streamlit app lets you upload any image (JPEG/PNG). It preprocesses the image, resizes it to 32×32, and predicts the top 3 likely classes with confidence scores.

Run the app:
streamlit run app.py

Example Prediction Output:
Predictions:
dog — confidence: 89.21%
cat — confidence: 7.35%
horse — confidence: 3.12%

## File Structure
image_recognition_cnn.ipynb     # Model training and evaluation  
app.py                          # Streamlit interface for prediction  
cifar10_cnn_model_augmented.h5  # Trained model file  
README.md                       # Project documentation

## Customization
You can train this model on your own dataset by modifying the data loading section in the notebook or replacing the CIFAR-10 dataset with your own labeled images.

## Requirements
Install all dependencies with:
pip install tensorflow numpy matplotlib streamlit pillow

## Contributing
Contributions are welcome!
Feel free to fork the repository, improve the game, and open a pull request. Let's grow this classic game together!

## License
This project is licensed under the [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)

## Author
**Aarya Mehta**  
🔗 [GitHub Profile](https://github.com/AaryaMehta2506)


