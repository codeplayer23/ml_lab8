import numpy as np

def sigmoid(z): return 1.0 / (1.0 + np.exp(-z))
def sigmoid_derivative(output): return output * (1.0 - output)

# AND gate + one-hot encoding
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
Y = np.array([[1,0],[1,0],[1,0],[0,1]], dtype=float)

# Hyperparameters
alpha = 0.5
max_iters = 5000
error_threshold = 0.002

# Initialize weights and bias
np.random.seed(42)
V = np.random.uniform(-1, 1, size=(2, 2))   
bh = np.random.uniform(-1, 1, size=(2,))    
W = np.random.uniform(-1, 1, size=(2, 2))   
bo = np.random.uniform(-1, 1, size=(2,))    

for epoch in range(1, max_iters+1):
    se_sum = 0.0
    for xi, target in zip(X, Y):
        # forward
        zh = np.dot(V, xi) + bh
        ah = sigmoid(zh)
        zo = np.dot(W, ah) + bo
        ao = sigmoid(zo)

        # error
        error = target - ao
        se_sum += np.sum(0.5 * (error ** 2))

        # backprop
        delta_o = error * sigmoid_derivative(ao)
        delta_h = sigmoid_derivative(ah) * np.dot(W.T, delta_o)

        # updates
        W += alpha * np.outer(delta_o, ah)
        bo += alpha * delta_o
        V += alpha * np.outer(delta_h, xi)
        bh += alpha * delta_h

    mse = se_sum / len(X)
    if mse <= error_threshold:
        print(f"Converged at epoch {epoch} with MSE={mse:.6f}")
        break
else:
    print(f"Stopped after {max_iters} epochs, final MSE={mse:.6f}")

# Predictions
print("\nFinal predictions:")
for xi in X:
    ah = sigmoid(np.dot(V, xi) + bh)
    ao = sigmoid(np.dot(W, ah) + bo)
    print(f"Input {xi} → Output {np.round(ao,3)} → Class {np.argmax(ao)}")
