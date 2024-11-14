Quantum Neural Network for Stock Market Direction Prediction Using NASDAQ Data
This repository presents a Quantum Neural Network (QNN) model designed to predict stock market trends using NASDAQ data. By leveraging quantum computing techniques, the project aims to explore the potential of QNNs in accurately forecasting stock price direction, offering new insights for high-frequency trading and financial analysis.

Table of Contents
Overview
Project Features
Data
QNN Architecture
Implementation
Results
Future Scope
Getting Started
Dependencies
Usage
Contributing
License
Overview
Stock market prediction is a complex task due to the nonlinear and high-dimensional nature of financial data. Classical machine learning models often fall short in capturing these patterns, especially for high-frequency data. This project leverages QNNs, harnessing quantum mechanics principles to better model and predict price movement trends based on daily NASDAQ stock data.

Project Features
QNN-based model: Employs Quantum Neural Networks to predict whether the stock price will rise or fall.
PennyLane Framework: Uses PennyLane for quantum circuit implementation, supporting both quantum and classical computation.
Optimization with COBYLA: The COBYLA algorithm is used for training, which is well-suited for handling non-differentiable objective functions in quantum circuits.
Evaluation Metrics: Assesses model performance using accuracy, F1 score, precision, recall, and a confusion matrix.
Data
The dataset consists of NASDAQ daily stock prices, including:

Open and Close Prices: Used to calculate the price direction (up or down) for each day.
High and Low Prices: Included as additional features to improve the model’s predictive capacity.
Volume: Represents daily trade volume, contributing additional context to price movements.
The binary target variable indicates whether the stock price closed higher or lower than the opening price on a given day.

QNN Architecture
The QNN architecture includes:

Angle Embedding: Maps classical stock price data into quantum states.
Strongly Entangling Layers: Three entangling layers are applied to introduce complex relationships between qubits, allowing the QNN to represent intricate dependencies between features.
Circuit Structure: Four qubits with rotation gates (Rx, Ry, Rz) and control gates. This setup enables the model to capture non-linear patterns that are often present in financial time series data.
Implementation
Data Preprocessing:

Clean the dataset by handling missing values and standardizing numerical features.
Encode price movement as a binary variable (1 for an upward trend, 0 for a downward trend).
Split data into training (80%) and testing (20%) sets.
QNN Construction:

Build a quantum circuit using PennyLane with four qubits and three entangling layers.
Parameterize rotation and control gates based on input features to represent stock price patterns.
Training and Evaluation:

Train the QNN model using the COBYLA optimization algorithm with a mini-batch gradient approach.
Evaluate the model on the test set using metrics like accuracy, F1 score, precision, recall, and a confusion matrix.
Results
The QNN model achieved:

Accuracy: 90%
F1 Score: 0.93
These results indicate that QNNs hold promise for financial prediction tasks, as they effectively capture complex relationships within stock data that classical models may miss.

Future Scope
This project sets a foundation for further exploration of QNNs in financial analytics. Future work could involve:

Testing the model on larger datasets and real-time data.
Enhancing scalability and robustness for high-frequency trading applications.
Investigating methods to reduce overfitting and improve generalization.
Getting Started
To get a local copy up and running, follow these steps.

Dependencies
Python 3.8+
PennyLane (for quantum circuit implementation)
NumPy (for data manipulation)
Scikit-learn (for standard preprocessing and evaluation)
Matplotlib (optional, for visualizing results)
