import numpy as np

X = np.array([
    [20, 6, 2, 386],
    [16, 3, 6, 289],
    [27, 6, 2, 393],
    [19, 1, 2, 110],
    [24, 4, 2, 280],
    [22, 1, 5, 167],
    [15, 4, 2, 271],
    [18, 4, 2, 274],
    [21, 1, 4, 148],
    [16, 2, 4, 198]
], dtype=float)

y = np.array([1,1,1,0,1,0,1,1,0,0], dtype=float)

# Feature scaling
X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

# Perceptron
def sigmoid(z): return 1 / (1 + np.exp(-z))

class Perceptron:
    def __init__(self, input_dim, lr=0.1, epochs=2000):
        self.weights = np.random.randn(input_dim) * 0.01
        self.bias = 0
        self.lr = lr
        self.epochs = epochs

    def predict(self, x):
        return sigmoid(np.dot(x, self.weights) + self.bias)

    def fit(self, X, y):
        for _ in range(self.epochs):
            for xi, target in zip(X, y):
                y_pred = self.predict(xi)
                error = target - y_pred
                self.weights += self.lr * error * xi
                self.bias += self.lr * error

# Train perceptron
p = Perceptron(input_dim=X.shape[1])
p.fit(X, y)
preds_perceptron = [1 if p.predict(xi) >= 0.5 else 0 for xi in X]

print("Perceptron Predictions:", preds_perceptron)
print("Perceptron Accuracy:", np.mean(preds_perceptron == y))


# Add bias column
X_bias = np.c_[np.ones(X.shape[0]), X]
W_pinv = np.linalg.pinv(X_bias) @ y

# Predictions
y_hat = X_bias @ W_pinv
preds_pinv = [1 if val >= 0.5 else 0 for val in y_hat]

print("\nPseudo-Inverse Predictions:", preds_pinv)
print("Pseudo-Inverse Accuracy:", np.mean(preds_pinv == y))
