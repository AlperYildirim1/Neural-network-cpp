MNIST Digit Recognition with Neural Networks in C++
This project implements a neural network from scratch in C++ to recognize handwritten digits from the MNIST dataset. The Eigen library is used for matrix operations, enabling efficient computation in training and prediction phases.

Overview
The project trains a neural network to classify images of handwritten digits (0-9) from the MNIST dataset. The neural network is fully implemented in C++, with the Eigen library handling matrix operations, which allows the model to be trained without relying on deep learning frameworks.

Features
Neural Network: Multi-layer perceptron (MLP) with configurable layers and neurons.
Eigen Library: Used for efficient matrix operations.
MNIST Compatibility: Trained on the MNIST dataset for digit recognition.
From Scratch Implementation: No deep learning libraries, only Eigen for linear algebra.

Results
After training 10 epochs, the model achieves an accuracy of approximately 90,2% on the MNIST test set.