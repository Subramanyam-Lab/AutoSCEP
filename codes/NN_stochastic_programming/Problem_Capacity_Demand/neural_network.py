# neural_network.py

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

def train_neural_network(training_data_file):
    """
    Train a neural network on the training data.
    """
    # Load the training data
    data = pd.read_csv(training_data_file)
    X_train = data['capacity'].values.reshape(-1, 1)
    y_train = data['expected_cost'].values

    # Define the neural network model
    model = keras.Sequential([
        keras.layers.Dense(16, activation='relu', input_shape=(1,)),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1)  # Output layer
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mse')

    # Train the model
    history = model.fit(X_train, y_train, epochs=200, batch_size=4, validation_split=0.2, verbose=0)

    # Plot training history
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.title('Training History')
    plt.show()

    # Save the trained model
    model.save('trained_model.h5')

    return model

if __name__ == "__main__":
    train_neural_network('training_data.csv')
