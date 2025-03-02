import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Nadam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import RandomNormal, GlorotUniform
from sklearn.model_selection import train_test_split

# Load dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Normalize data
X_train, X_val, X_test = X_train / 255.0, X_val / 255.0, X_test / 255.0

# One-hot encode labels
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_val = tf.keras.utils.to_categorical(y_val, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# User inputs
num_epochs = int(input("Enter number of epochs (5 or 10): "))
num_hidden_layers = int(input("Enter number of hidden layers (3, 4, or 5): "))
hidden_size = int(input("Enter size of every hidden layer (32, 64, 128): "))
weight_decay = float(input("Enter L2 regularization value (0, 0.0005, 0.5): "))
learning_rate = float(input("Enter learning rate (0.001 or 0.0001): "))
batch_size = int(input("Enter batch size (16, 32, 64): "))
optimizer_choice = input("Enter optimizer (sgd, momentum, nesterov, rmsprop, adam, nadam): ")
weight_init_choice = input("Enter weight initialization (random, Xavier): ")
activation_choice = input("Enter activation function (sigmoid, ReLU): ")

# Set weight initialization
if weight_init_choice.lower() == "xavier":
    initializer = GlorotUniform()
else:
    initializer = RandomNormal(mean=0.0, stddev=0.05)

# Set activation function
activation_function = "relu" if activation_choice.lower() == "relu" else "sigmoid"

# Build model
model = Sequential()
model.add(Flatten(input_shape=(32, 32, 3)))
for _ in range(num_hidden_layers):
    model.add(Dense(hidden_size, activation=activation_function, kernel_initializer=initializer,
                    kernel_regularizer=l2(weight_decay)))
model.add(Dense(10, activation='softmax'))

# Set optimizer
if optimizer_choice.lower() == "sgd":
    optimizer = SGD(learning_rate=learning_rate)
elif optimizer_choice.lower() == "momentum":
    optimizer = SGD(learning_rate=learning_rate, momentum=0.9)
elif optimizer_choice.lower() == "nesterov":
    optimizer = SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
elif optimizer_choice.lower() == "rmsprop":
    optimizer = RMSprop(learning_rate=learning_rate)
elif optimizer_choice.lower() == "adam":
    optimizer = Adam(learning_rate=learning_rate)
elif optimizer_choice.lower() == "nadam":
    optimizer = Nadam(learning_rate=learning_rate)
else:
    raise ValueError("Invalid optimizer choice")

# Compile model
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(X_val, y_val))

# Evaluate model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Test Accuracy: {test_acc * 100:.2f}%")