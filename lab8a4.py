import numpy as np
import matplotlib.pyplot as plt

# Step activation function
def step_activation(z):
    return 1 if z >= 0 else 0

class CustomPerceptron:
    def __init__(self, w0=10, w1=0.2, w2=-0.75, lr=0.1, epochs=100):
        self.bias = w0
        self.weights = np.array([w1, w2], dtype=float)
        self.lr = lr
        self.epochs = epochs

    def predict(self, x):
        z = np.dot(self.weights, x) + self.bias
        return step_activation(z)

    def fit(self, X, y):
        for epoch in range(self.epochs):
            errors = 0
            for xi, target in zip(X, y):
                y_pred = self.predict(xi)
                error = target - y_pred
                if error != 0:
                    errors += 1
                    self.weights += self.lr * error * xi
                    self.bias += self.lr * error
            if errors == 0:
                return epoch + 1  
        return self.epochs        

# AND gate 
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,0,0,1])

# Learning rates 
learning_rates = [0.1 * i for i in range(1, 11)]
epochs_to_converge = []

for lr in learning_rates:
    p = CustomPerceptron(lr=lr, epochs=50)  
    epochs = p.fit(X, y)
    epochs_to_converge.append(epochs)
    print(f"LR={lr:.1f} → Converged in {epochs} epochs")

# Plot results
plt.plot(learning_rates, epochs_to_converge, marker='o')
plt.xlabel("Learning Rate (α)")
plt.ylabel("Epochs to Converge")
plt.title("Effect of Learning Rate on Perceptron Convergence (AND Gate)")
plt.grid(True)
plt.show()
