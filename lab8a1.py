import numpy as np

# Step activation function
def step_function(z):
    return 1 if z >= 0 else 0

#bipolar sigmoid function
def bipolar_sigmoid(z):
    return (2 / (1 + np.exp(-z))) - 1

def bipolar_sigmoid_derivative(z):
    f = bipolar_sigmoid(z)
    return 0.5 * (1 + f) * (1 - f)

#Tanh function 

def tanh(z):
    return np.tanh(z)

def tanh_derivative(z):
    return 1 - np.tanh(z)**2

#ReLu activation function
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return np.where(z > 0, 1, 0)

#Leaky Relu activation function 
def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)

def leaky_relu_derivative(z, alpha=0.01):
    return np.where(z > 0, 1, alpha)

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
