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

y = np.array([1,1,1,0,1,0,1,1,0,0])

# Feature scaling 
X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

class Perceptron:
    def __init__(self, input_dim, lr=0.1, epochs=1000):
        self.weights = np.random.randn(input_dim) * 0.01
        self.bias = 0
        self.lr = lr
        self.epochs = epochs

    def predict(self, x):
        z = np.dot(x, self.weights) + self.bias
        return sigmoid(z)

    def fit(self, X, y):
        n_samples = X.shape[0]

        for epoch in range(self.epochs):
            for xi, target in zip(X, y):
                z = np.dot(xi, self.weights) + self.bias
                y_pred = sigmoid(z)

                # gradient update
                error = target - y_pred
                self.weights += self.lr * error * xi
                self.bias += self.lr * error

# Initialize perceptron
model = Perceptron(input_dim=X.shape[1], lr=0.1, epochs=2000)

# Train
model.fit(X, y)

# Predictions
preds = [1 if model.predict(xi) >= 0.5 else 0 for xi in X]
print("Predictions:", preds)
print("Actual     :", y.tolist())

# Accuracy
accuracy = np.mean(preds == y)
print(f"Accuracy: {accuracy*100:.2f}%")
