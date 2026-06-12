# Quantum Neural Network

This project implements a simple Quantum Neural Network (QNN) classifier in Python. The program constructs a parameterized quantum circuit, trains the circuit parameters using gradient descent, and applies the trained model to classify simple binary patterns.

The goal of this project was to explore how quantum circuits can be used as trainable machine learning models and to better understand quantum machine learning, parameterized quantum circuits, and optimization-based training.

## Project Overview

The QNN uses layers of parameterized quantum gates to process input data. Each layer applies rotation gates followed by entangling operations. The circuit parameters are updated during training to improve classification performance on a small binary-pattern dataset.

The model includes:

- Parameterized X, Y, and Z rotation gates
- CNOT entangling gates
- A training loop using gradient descent
- A simple binary-pattern classification task
- Evaluation of the trained model on test inputs
