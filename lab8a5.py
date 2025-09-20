import numpy as np

# Activation functions
def step_activation(z):
    return 1 if z >= 0 else 0

def bipolar_step(z):
    return 1 if z >= 0 else -1

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return max(0, z)

class CustomPerceptron:
    def __init__(self, w0=10, w1=0.2, w2=-0.75, lr=0.05, epochs=10, activation="step"):
        self.bias = w0
        self.weights = np.array([w1, w2], dtype=float)
        self.lr = lr
        self.epochs = epochs
        self.activation = activation

    def activate(self, z):
        if self.activation == "step":
            return step_activation(z)
        elif self.activation == "bipolar":
            return bipolar_step(z)
        elif self.activation == "sigmoid":
            return sigmoid(z)
        elif self.activation == "relu":
            return relu(z)
        else:
            raise ValueError("Unknown activation function!")

    def predict(self, x):
        z = np.dot(self.weights, x) + self.bias
        return self.activate(z)

    def fit(self, X, y):
        for epoch in range(self.epochs):
            errors = 0
            for xi, target in zip(X, y):
                y_pred = self.predict(xi)

                if self.activation in ["sigmoid", "relu"]:
                    y_bin = 1 if y_pred >= 0.5 else 0
                elif self.activation == "bipolar":
                    y_bin = 1 if y_pred == 1 else 0  
                else:
                    y_bin = y_pred

                error = target - y_bin
                if error != 0:
                    errors += 1
                    self.weights += self.lr * error * xi
                    self.bias += self.lr * error

            if errors == 0:
                print(f"Converged in {epoch+1} epochs using {self.activation}")
                return epoch + 1
        return self.epochs

# XOR gate
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,0])  

# Test with different activation functions
for act in ["step", "bipolar", "sigmoid", "relu"]:
    print(f"\n=== Activation: {act.upper()} ===")
    p = CustomPerceptron(lr=0.1, epochs=50, w0=10, w1=0.2, w2=-0.75, activation=act)
    epochs = p.fit(X, y)
    preds = [p.predict(xi) for xi in X]
    print(f"Predictions: {preds}")
