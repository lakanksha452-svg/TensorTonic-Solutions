
import numpy as np

def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def train_logistic_regression(X, y, lr, steps):
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    N, D = X.shape

    # Initialize parameters
    w = np.zeros(D)
    b = 0.0

    # Gradient descent
    for _ in range(steps):

        # Forward propagation
        z = np.dot(X, w) + b
        p = _sigmoid(z)

        # Compute gradients
        dw = np.dot(X.T, (p - y)) / N
        db = np.sum(p - y) / N

        # Update parameters
        w -= lr * dw
        b -= lr * db

    return w, float(b)
