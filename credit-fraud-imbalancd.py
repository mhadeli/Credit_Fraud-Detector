import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
import itertools

# Load and preprocess the data
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)

    # Check for null values
    if df.isnull().sum().max() > 0:
        raise ValueError("Dataset contains null values")
    
    # Scale the 'Amount' and 'Time' features
    robust_scaler = RobustScaler()
    df['scaled_amount'] = robust_scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df['scaled_time'] = robust_scaler.fit_transform(df['Time'].values.reshape(-1, 1))
    
    df.drop(['Time', 'Amount'], axis=1, inplace=True)
    df.rename(columns={'scaled_amount': 'Amount', 'scaled_time': 'Time'}, inplace=True)

    return df

# Split the data into training and testing sets
def split_data(df, test_size=0.2, random_state=42):
    X = df.drop('Class', axis=1)
    y = df['Class']
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Handle class imbalance with undersampling
def undersample_data(X_train, y_train):
    undersample = NearMiss(version=1)
    return undersample.fit_resample(X_train, y_train)

# Handle class imbalance with SMOTE
def oversample_data(X_train, y_train):
    smote = SMOTE(sampling_strategy='minority', random_state=42)
    return smote.fit_resample(X_train, y_train)

# Build and compile a neural network model
def build_nn_model(input_dim, learning_rate=0.001):
    model = Sequential([
        Dense(16, input_dim=input_dim, activation='relu'),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train the model
def train_model(model, X_train, y_train, batch_size=25, epochs=20, validation_split=0.2):
    model.fit(X_train, y_train, validation_split=validation_split, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=2)

# Evaluate model performance with a confusion matrix
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def main(file_path):
    # Load and preprocess the data
    df = load_and_preprocess_data(file_path)

    # Split the data
    X_train, X_test, y_train, y_test = split_data(df)

    # Undersample the data
    X_train_undersample, y_train_undersample = undersample_data(X_train, y_train)

    # Build, train, and evaluate the undersampling model
    undersample_model = build_nn_model(X_train_undersample.shape[1])
    train_model(undersample_model, X_train_undersample, y_train_undersample)
    undersample_predictions = undersample_model.predict(X_test, batch_size=200, verbose=0)
    undersample_cm = confusion_matrix(y_test, undersample_predictions.round())
    plot_confusion_matrix(undersample_cm, classes=['No Fraud', 'Fraud'])

    # Oversample the data
    X_train_smote, y_train_smote = oversample_data(X_train, y_train)

    # Build, train, and evaluate the oversampling model
    oversample_model = build_nn_model(X_train_smote.shape[1])
    train_model(oversample_model, X_train_smote, y_train_smote, batch_size=300)
    oversample_predictions = oversample_model.predict(X_test, batch_size=200, verbose=0)
    oversample_smote_cm = confusion_matrix(y_test, oversample_predictions.round())
    plot_confusion_matrix(oversample_smote_cm, classes=['No Fraud', 'Fraud'])

    plt.show()

if __name__ == "__main__":
    main("creditcard.csv")