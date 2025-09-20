#Bipolar activation function 
def bipolar_step(z):
    return 1 if z >= 0 else -1

#sigmoid activation function 
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#ReLU activation function
def relu(z):
    return max(0, z)

import numpy as np

class CustomPerceptron:
    def __init__(self, w0=10, w1=0.2, w2=-0.75, lr=0.05, epochs=100, activation="step"):
        self.bias = w0
        self.weights = np.array([w1, w2], dtype=float)
        self.lr = lr
        self.epochs = epochs
        self.activation = activation

    def activate(self, z):
        if self.activation == "step":
            return 1 if z >= 0 else 0
        elif self.activation == "bipolar":
            return 1 if z >= 0 else -1
        elif self.activation == "sigmoid":
            return 1 / (1 + np.exp(-z))
        elif self.activation == "relu":
            return max(0, z)
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
                # For sigmoid/ReLU, threshold at 0.5 to compare with AND output
                if self.activation in ["sigmoid", "relu"]:
                    y_bin = 1 if y_pred >= 0.5 else 0
                elif self.activation == "bipolar":
                    y_bin = 1 if y_pred == 1 else 0  # map -1 → 0
                else:
                    y_bin = y_pred

                error = target - y_bin
                if error != 0:
                    errors += 1
                    self.weights += self.lr * error * xi
                    self.bias += self.lr * error
            if errors == 0:
                print(f"Converged in {epoch+1} epochs with {self.activation} activation")
                break

# AND Gate 
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,0,0,1])

# Test with different activation functions
for act in ["step", "bipolar", "sigmoid", "relu"]:
    p = CustomPerceptron(activation=act, epochs=50)
    p.fit(X, y)
    print("Final Weights:", p.weights, "Bias:", p.bias)
    print("Predictions:", [p.predict(xi) for xi in X])
    print("-"*50)
