import numpy as np

# Step activation function
def step_function(z):
    return 1 if z >= 0 else 0

class Perceptron:
    def __init__(self, input_size, lr=0.1, epochs=10):
        self.weights = np.zeros(input_size)
        self.bias = 0
        self.lr = lr
        self.epochs = epochs

    def predict(self, x):
        z = np.dot(self.weights, x) + self.bias  
        return step_function(z)                 

    def fit(self, X, y):
        for _ in range(self.epochs):
            for xi, target in zip(X, y):
                y_pred = self.predict(xi)
                #error calculation 
                error = target - y_pred          
                self.weights += self.lr * error * xi
                self.bias += self.lr * error
