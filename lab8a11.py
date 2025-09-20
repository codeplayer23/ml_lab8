import numpy as np
from sklearn.neural_network import MLPClassifier

# Input data for AND gate
X_and = np.array([[0,0],
                  [0,1],
                  [1,0],
                  [1,1]])

# Output for AND gate
y_and = np.array([0, 0, 0, 1])

# Create MLPClassifier
and_model = MLPClassifier(hidden_layer_sizes=(), activation='logistic', max_iter=1000, random_state=42)

# Train the model
and_model.fit(X_and, y_and)

# Predictions
pred_and = and_model.predict(X_and)
print("AND Gate Predictions:", pred_and)

# Input data for XOR gate
X_xor = np.array([[0,0],
                  [0,1],
                  [1,0],
                  [1,1]])

# Output for XOR gate
y_xor = np.array([0, 1, 1, 0])

xor_model = MLPClassifier(hidden_layer_sizes=(2,), activation='logistic', max_iter=10000, random_state=42)

# Train the model
xor_model.fit(X_xor, y_xor)

# Predictions
pred_xor = xor_model.predict(X_xor)
print("XOR Gate Predictions:", pred_xor)
