# neural_network.py

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

def train_neural_network(training_data_file):
    """
    Train a feedforward neural network on the training data.
    """
    # Load the training data
    data = pd.read_csv(training_data_file)
    X_train = data['capacity'].values.reshape(-1, 1)
    y_train = data['expected_cost'].values

    # Normalize the data
    from sklearn.preprocessing import MinMaxScaler
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))

    # Define the feedforward neural network model
    model = keras.Sequential([
        keras.layers.Dense(32, activation='relu', input_shape=(1,)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1)  # Output layer
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mse')

    # Train the model
    history = model.fit(X_train_scaled, y_train_scaled, epochs=500, batch_size=4, validation_split=0.2, verbose=0)

    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.title('Training History')
    plt.show()

    # Save the trained model and scalers
    model.save('trained_model.h5')
    joblib.dump(scaler_X, 'scaler_X.pkl')
    joblib.dump(scaler_y, 'scaler_y.pkl')

    return model

if __name__ == "__main__":
    train_neural_network('training_data.csv')
