import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import SGD, RMSprop, Adam, Nadam
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.regularizers import l2

# Load CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
num_classes = 10

# Normalize pixel values
X_train, X_test = X_train / 255.0, X_test / 255.0

# One-hot encode labels
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# Take 10% of training data for validation
val_split = int(0.1 * X_train.shape[0])
X_val, y_val = X_train[:val_split], y_train[:val_split]
X_train, y_train = X_train[val_split:], y_train[val_split:]

# Get user inputs
epochs = int(input("Enter number of epochs (5 or 10): "))
hidden_layers = int(input("Enter number of hidden layers (3, 4, or 5): "))
hidden_size = int(input("Enter size of each hidden layer (32, 64, or 128): "))
weight_decay = float(input("Enter weight decay (0, 0.0005, or 0.5): "))
learning_rate = float(input("Enter learning rate (1e-3 or 1e-4): "))
batch_size = int(input("Enter batch size (16, 32, or 64): "))
weight_init = input("Enter weight initialization (random or Xavier): ")
activation = input("Enter activation function (sigmoid or ReLU): ")

# Set weight initializer
initializer = GlorotUniform() if weight_init.lower() == "xavier" else "random_normal"

# Build model
def build_model():
    model = Sequential()
    model.add(Flatten(input_shape=(32, 32, 3)))
    for _ in range(hidden_layers):
        model.add(Dense(hidden_size, activation=activation.lower(), kernel_initializer=initializer, kernel_regularizer=l2(weight_decay)))
    model.add(Dense(num_classes, activation='softmax'))
    return model

# Define optimizers
optimizers = {
    "SGD": SGD(learning_rate=learning_rate, momentum=0.0),
    "Momentum": SGD(learning_rate=learning_rate, momentum=0.9),
    "NAG": SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True),
    "RMSprop": RMSprop(learning_rate=learning_rate),
    "Adam": Adam(learning_rate=learning_rate),
    "Nadam": Nadam(learning_rate=learning_rate)
}

# Train and evaluate each optimizer
results = {}
for opt_name, optimizer in optimizers.items():
    print(f"\nTraining with {opt_name} optimizer...")
    model = build_model()
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=2)
    
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    results[opt_name] = test_acc * 100
    print(f"{opt_name} Test Accuracy: {test_acc * 100:.2f}%")

# Print results
sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
print("\nFinal Accuracy Scores:")
for opt, acc in sorted_results:
    print(f"{opt}: {acc:.2f}%")

# Print ranking order
print("\nOrder of best accuracy:")
print(" > ".join([f"{opt} ({acc:.2f}%)" for opt, acc in sorted_results]))