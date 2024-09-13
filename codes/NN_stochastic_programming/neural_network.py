import tensorflow as tf
from tensorflow import keras

# Prepare input and output data
X_train = np.array([data[0] for data in training_data])
y_train = np.array([data[1] for data in training_data])

# Define the neural network model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1)  # Output layer for expected cost
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
