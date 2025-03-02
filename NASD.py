import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Ensure dataset directory exists
def check_file_exists(filepath):
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found!")
        return False
    return True

# Define NASD model
def create_nasd_model(input_size=21):
    model = Sequential([
        Dense(64, activation=LeakyReLU(alpha=0.1), input_shape=(input_size,)),
        Dense(32, activation=LeakyReLU(alpha=0.05)),
        Dense(16, activation='relu'),
        Dense(4, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model

# Load training data
def load_data(x_file, y_file):
    if check_file_exists(x_file) and check_file_exists(y_file):
        X = pd.read_csv(x_file, header=None).values.T
        Y = pd.read_csv(y_file, header=None).values.T
        print(f"Loaded data: X shape {X.shape}, Y shape {Y.shape}")
        return X, Y
    return None, None

# Train NASD model
def train_nasd_model(model, X, Y, batch_size=2000, epochs=50):
    print("Training model...")
    history = model.fit(X, Y, batch_size=batch_size, epochs=epochs, verbose=1)
    model.save("NASD_model.h5")
    print("Model saved as NASD_model.h5")
    return history

# Plot accuracy
def plot_training_history(history):
    plt.plot(history.history['accuracy'])
    plt.title('NASD Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train'], loc='upper left')
    plt.show()

# Evaluate model
def evaluate_nasd_model(model, test_x_file, test_y_file):
    X_test, Y_test = load_data(test_x_file, test_y_file)
    if X_test is not None:
        loss, acc = model.evaluate(X_test, Y_test, batch_size=128)
        print(f"Test Accuracy: {acc:.4f}")
        return acc
    return None

# Main execution
if __name__ == "__main__":
    train_x_file = "train_x.csv"
    train_y_file = "train_y.csv"
    test_x_file = "test_x.csv"
    test_y_file = "test_y.csv"

    X_train, Y_train = load_data(train_x_file, train_y_file)
    if X_train is not None:
        model = create_nasd_model()
        history = train_nasd_model(model, X_train, Y_train)
        plot_training_history(history)

        # Evaluate on test data
        model = load_model("NASD_model.h5")
        evaluate_nasd_model(model, test_x_file, test_y_file)
