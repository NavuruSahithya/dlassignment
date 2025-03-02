import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# Load CIFAR-10 dataset
def load_cifar_dataset():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    num_classes = 10

    # Normalize pixel values to [0, 1]
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # Flatten images to 1D vectors (32x32x3 -> 3072)
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    # Convert labels to one-hot encoding
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    return X_train, X_test, y_train, y_test, num_classes

X_train, X_test, y_train, y_test, num_classes = load_cifar_dataset()

# Hyperparameters
input_size = 32 * 32 * 3
hidden_size = 128
output_size = num_classes
learning_rate = 0.001
beta1 = 0.9  # Decay rate for first moment estimate
beta2 = 0.999  # Decay rate for second moment estimate
epsilon = 1e-8
epochs = 100
batch_size = 64

# Initialize weights and biases
np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros((1, output_size))

# Adam optimizer parameters
mW1, vW1 = np.zeros_like(W1), np.zeros_like(W1)
mW2, vW2 = np.zeros_like(W2), np.zeros_like(W2)
mb1, vb1 = np.zeros_like(b1), np.zeros_like(b1)
mb2, vb2 = np.zeros_like(b2), np.zeros_like(b2)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Training loop
for epoch in range(epochs):
    for i in range(0, X_train.shape[0], batch_size):
        X_batch = X_train[i:i + batch_size]
        y_batch = y_train[i:i + batch_size]

        # Forward pass
        hidden_input = np.dot(X_batch, W1) + b1
        hidden_output = sigmoid(hidden_input)
        output_input = np.dot(hidden_output, W2) + b2
        output = softmax(output_input)

        # Compute loss
        loss = -np.mean(np.sum(y_batch * np.log(output + 1e-10), axis=1))

        # Backpropagation
        d_output = (output - y_batch) / batch_size
        d_hidden = np.dot(d_output, W2.T) * sigmoid_derivative(hidden_output)

        dW2 = np.dot(hidden_output.T, d_output)
        db2 = np.sum(d_output, axis=0, keepdims=True)
        dW1 = np.dot(X_batch.T, d_hidden)
        db1 = np.sum(d_hidden, axis=0, keepdims=True)

        # Adam updates
        for param, dparam, m, v in [(W1, dW1, mW1, vW1), (W2, dW2, mW2, vW2),
                                    (b1, db1, mb1, vb1), (b2, db2, mb2, vb2)]:
            m[:] = beta1 * m + (1 - beta1) * dparam
            v[:] = beta2 * v + (1 - beta2) * (dparam ** 2)
            m_hat = m / (1 - beta1 ** (epoch + 1))
            v_hat = v / (1 - beta2 ** (epoch + 1))
            param -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss:.4f}')

# Testing
hidden_input = np.dot(X_test, W1) + b1
hidden_output = sigmoid(hidden_input)
output_input = np.dot(hidden_output, W2) + b2
output = softmax(output_input)

# Predictions
predictions = np.argmax(output, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

# Accuracy calculation
accuracy = accuracy_score(y_test_labels, predictions)
print(f'Accuracy on test set with Adam optimizer: {accuracy * 100:.2f}%')