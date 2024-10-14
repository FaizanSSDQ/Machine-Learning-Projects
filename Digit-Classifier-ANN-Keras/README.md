# Handwritten Digit Classifier

This project is a machine learning model designed to classify handwritten digits (0-9) using the famous **MNIST** dataset. The model is built using **TensorFlow** and **Keras** and is trained on grayscale images of handwritten digits. The classifier aims to predict the digit from an input image with high accuracy.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Results](#results)
- [Conclusion](#conclusion)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This project explores how a neural network can be built to classify handwritten digits using the **MNIST** dataset. The dataset consists of 28x28 pixel images of digits. The model preprocesses these images, builds a neural network with two hidden layers, trains it on labeled data, and then predicts unseen images.

The project demonstrates the following concepts:
- Image flattening and normalization.
- Creating a neural network using **Dense layers**.
- Using **softmax** activation for multi-class classification.
- Model training using **categorical crossentropy** loss function and **SGD optimizer**.
- Evaluating and visualizing results.

## Dataset

The **MNIST dataset** is a large database of handwritten digits commonly used for training various image processing systems. It consists of:
- **60,000 training images**
- **10,000 test images**
Each image is a 28x28 grayscale image, and the associated label is a digit from 0 to 9.

## Model Architecture

The neural network is built using a **Sequential** model with the following layers:
1. **Input Layer**: The input data is flattened from a 28x28 matrix into a 1D vector of size 784.
2. **Hidden Layer 1**: A fully connected (Dense) layer with 400 neurons and ReLU activation.
3. **Hidden Layer 2**: A fully connected (Dense) layer with 20 neurons and ReLU activation.
4. **Output Layer**: A fully connected (Dense) layer with 10 neurons and softmax activation for classification into one of the 10 digit classes.

### Hyperparameters
- **Input size**: 784 (28x28 pixels)
- **Hidden Layer 1**: 400 neurons
- **Hidden Layer 2**: 20 neurons
- **Output Layer**: 10 neurons (for digits 0-9)
- **Batch size**: 200
- **Epochs**: 5

## Installation

To get started, clone this repository and install the necessary dependencies.

1. Clone the repository:
   ```bash
   git clone https://github.com/YourUsername/Digit-Classifier.git
   cd Digit-Classifier
   ```
2. Install the required libraries:
   ```bash
   pip install tensorflow keras matplotlib numpy pandas scikit-learn
   ```
# **How to Run**
1. Open the Jupyter Notebook file Digit_Classifier.ipynb in your environment.
2. Run all cells to train the model, test the predictions, and visualize the results.
3. Optionally, you can modify the model architecture, the number of epochs, or any other hyperparameters and re-run the notebook.

# **Results**
The model achieves a high level of accuracy in recognizing handwritten digits, both on training and test sets. After training for 5 epochs, it is capable of making accurate predictions with softmax probabilities.

# **Sample Output**
The notebook also contains visualization of predictions on a subset of test data where the model successfully identifies the correct digits.

# **Conclusion**
This project demonstrates a simple yet effective neural network for handwritten digit classification. By utilizing Dense layers and activation functions like ReLU and softmax, the model achieves good accuracy. The project provides a solid foundation for understanding deep learning concepts and applying them to image classification problems.

## Contributing
Feel free to contribute to this project by opening a pull request or suggesting improvements. Contributions can include improving model accuracy, adding new features, or optimizing code performance.

# **License**

You can copy this text into your `README.md` file directly for the project. The Markdown format used here ensures proper structure and styling when displayed on GitHub or other Markdown-compatible platforms.

