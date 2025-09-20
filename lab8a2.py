import numpy as np

# Step activation function
def step_activation(z):
    return 1 if z >= 0 else 0

class CustomPerceptron:
    def __init__(self, w0=10, w1=0.2, w2=-0.75, lr=0.05, epochs=10):
        self.bias = w0
        self.weights = np.array([w1, w2], dtype=float)
        self.lr = lr
        self.epochs = epochs

    def predict(self, x):
        z = np.dot(self.weights, x) + self.bias
        return step_activation(z)

    def fit(self, X, y):
        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch+1}")
            for xi, target in zip(X, y):
                y_pred = self.predict(xi)
                error = target - y_pred
                # Weight update 
                self.weights += self.lr * error * xi
                self.bias += self.lr * error
                print(f"Input: {xi}, Target: {target}, Predicted: {y_pred}, Error: {error}")
                print(f"Updated weights: {self.weights}, Bias: {self.bias}")

# AND gate
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([0, 0, 0, 1]) 

# Create perceptron and train
p = CustomPerceptron()
p.fit(X, y)

# Final results
print("\nFinal Weights:", p.weights)
print("Final Bias:", p.bias)

# Test predictions
print("\nTesting AND gate predictions:")
for xi in X:
    print(f"Input: {xi}, Output: {p.predict(xi)}")
