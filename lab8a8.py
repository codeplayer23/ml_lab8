import numpy as np

def sigmoid(z): return 1.0 / (1.0 + np.exp(-z))
def sigmoid_derivative(output): return output * (1.0 - output)

# AND dataset
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y = np.array([0,0,0,1], dtype=float)

# hyperparameters
alpha = 0.5          
max_iters = 1000
error_threshold = 0.002

# initialize weights and bias
np.random.seed(21)
V = np.random.uniform(-1, 1, size=(2, 2))
bh = np.random.uniform(-1, 1, size=(2,))
W = np.random.uniform(-1, 1, size=(1, 2))
bo = np.random.uniform(-1, 1, size=(1,))[0]

history = []
for epoch in range(1, max_iters + 1):
    se_sum = 0.0
    for xi, target in zip(X, y):
        # forward
        zh = np.dot(V, xi) + bh
        ah = sigmoid(zh)
        zo = np.dot(W, ah) + bo
        ao = sigmoid(zo)[0]

        # error & backprop
        error = target - ao
        se_sum += 0.5 * (error ** 2)

        delta_o = error * sigmoid_derivative(ao)
        delta_h = sigmoid_derivative(ah) * (W.T.flatten() * delta_o)

        # updates
        W += alpha * delta_o * ah.reshape(1, -1)
        bo += alpha * delta_o
        V += alpha * np.outer(delta_h, xi)
        bh += alpha * delta_h

    mse = se_sum / X.shape[0]
    history.append(mse)
    if mse <= error_threshold:
        print(f"Converged at epoch {epoch} with MSE={mse:.6f}")
        break
else:
    print(f"Stopped after {max_iters} iterations. Final MSE={mse:.6f}")

# results
print("Final V (hidden weights):\n", V)
print("Final bh (hidden biases):\n", bh)
print("Final W (output weights):\n", W)
print("Final bo (output bias):\n", bo)

preds = []
probs = []
for xi in X:
    ah = sigmoid(np.dot(V, xi) + bh)
    ao = sigmoid(np.dot(W, ah) + bo)[0]
    probs.append(ao)
    preds.append(1 if ao >= 0.5 else 0)

print("Predicted probs:", [round(p,4) for p in probs])
print("Predicted classes:", preds)
print("Actual classes   :", y.tolist())
