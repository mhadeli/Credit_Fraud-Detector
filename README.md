# Credit Fraud Detector

This repository contains code for detecting credit card fraud using a neural network model. The dataset used is highly imbalanced, and the code demonstrates techniques to handle this imbalance using undersampling and SMOTE (Synthetic Minority Over-sampling Technique).

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Introduction

Credit card fraud detection is a critical application of machine learning. This project showcases how to handle imbalanced datasets and build a neural network model to detect fraudulent transactions.

## Dataset

The dataset used is the [Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) from Kaggle. It contains transactions made by credit cards in September 2013 by European cardholders. The dataset has 284,807 transactions, with only 492 being fraudulent (0.172% of all transactions).

## Preprocessing

1. **Scaling**: The `Amount` and `Time` features are scaled using `RobustScaler` to reduce the influence of outliers.
2. **Splitting**: The dataset is split into training and testing sets.
3. **Handling Imbalance**:
   - **Undersampling**: Reduces the number of majority class instances to balance the dataset.
   - **SMOTE**: Generates synthetic minority class instances to balance the dataset.

## Modeling

Two models are built using Keras:
1. **Undersampling Model**: Trained on the undersampled dataset.
2. **Oversampling Model (SMOTE)**: Trained on the SMOTE-generated dataset.

Both models use a simple neural network architecture with:
- Input layer
- Two hidden layers with ReLU activation
- Output layer with sigmoid activation

## Evaluation

The models are evaluated using a confusion matrix to assess their performance on detecting fraudulent transactions.

## Installation

To run this code, you need to have Python 3 and the required packages installed. Follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/credit-fraud-detector.git
    ```

2. Navigate to the project directory:

    ```bash
    cd credit-fraud-detector
    ```

3. Create and activate a virtual environment (optional but recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

4. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Place the `creditcard.csv` file in the project directory.

2. Run the script:

    ```bash
    python credit_fraud_detector.py
    ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
