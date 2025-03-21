# 2. In the above implementation use a learning rate of 0.01 and print the loss values, accuracy and weight updates. 
#    Plot the loss curve. Explain the inference you make from the observations regrading the loss values, accuracy and weight updates.

import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Activation Functions and Derivatives
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_deriv(a):
    # Derivative assuming input a is the sigmoid activation output
    return a * (1 - a)

# Hyperparameters and Network Architecture
input_dim = 3        # 3 input features
hidden_dim1 = 4      # 1st hidden layer: 4 neurons
hidden_dim2 = 4      # 2nd hidden layer: 4 neurons
output_dim = 2       # Output layer: 2 neurons

learning_rate = 0.01  # Changed learning rate from 0.1 to 0.01
momentum = 0.9
epochs = 10
batch_size = 10
num_samples = 100

# Initialize Weights and Biases (small random values)
W1 = np.random.randn(input_dim, hidden_dim1) * 0.1
b1 = np.random.randn(1, hidden_dim1) * 0.1

W2 = np.random.randn(hidden_dim1, hidden_dim2) * 0.1
b2 = np.random.randn(1, hidden_dim2) * 0.1

W3 = np.random.randn(hidden_dim2, output_dim) * 0.1
b3 = np.random.randn(1, output_dim) * 0.1

# Initialize momentum velocities for weights and biases (all zeros)
vW1 = np.zeros_like(W1)
vb1 = np.zeros_like(b1)
vW2 = np.zeros_like(W2)
vb2 = np.zeros_like(b2)
vW3 = np.zeros_like(W3)
vb3 = np.zeros_like(b3)

# Create Training Data
# Generate 100 random samples for inputs (features)
X = np.random.rand(num_samples, input_dim)

# Generate random binary targets for 2 classes (one-hot style)
Y = np.random.randint(0, 2, size=(num_samples, output_dim))

# Helper Functions for Loss and Accuracy
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def compute_accuracy(y_true, y_pred):
    # Threshold predictions at 0.5 for binary classification
    preds = (y_pred >= 0.5).astype(int)
    # Count a sample as correct if all output neurons match the target exactly
    correct = np.all(preds == y_true, axis=1)
    return np.mean(correct)

# Training Loop
loss_history = []  # Record loss per epoch

num_batches = num_samples // batch_size

for epoch in range(1, epochs + 1):
    # Shuffle the training data at the start of each epoch
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    X_shuffled = X[indices]
    Y_shuffled = Y[indices]
    
    epoch_loss = 0.0
    update_magnitudes = {'W1': [], 'W2': [], 'W3': []}
    
    for b in range(num_batches):
        start = b * batch_size
        end = start + batch_size
        X_batch = X_shuffled[start:end]
        Y_batch = Y_shuffled[start:end]
        
        # 1. Forward Propagation
        z1 = np.dot(X_batch, W1) + b1
        a1 = sigmoid(z1)
        
        z2 = np.dot(a1, W2) + b2
        a2 = sigmoid(z2)
        
        z3 = np.dot(a2, W3) + b3
        a3 = sigmoid(z3)  # Final output
        
        # 2. Compute Loss
        loss = mse_loss(Y_batch, a3)
        epoch_loss += loss
        
        # 3. Backpropagation
        error3 = a3 - Y_batch  # derivative of MSE loss
        delta3 = error3 * sigmoid_deriv(a3)
        
        # Gradients for W3 and b3
        dW3 = np.dot(a2.T, delta3) / batch_size
        db3 = np.sum(delta3, axis=0, keepdims=True) / batch_size
        
        # Hidden Layer 2
        error2 = np.dot(delta3, W3.T)
        delta2 = error2 * sigmoid_deriv(a2)
        dW2 = np.dot(a1.T, delta2) / batch_size
        db2 = np.sum(delta2, axis=0, keepdims=True) / batch_size
        
        # Hidden Layer 1
        error1 = np.dot(delta2, W2.T)
        delta1 = error1 * sigmoid_deriv(a1)
        dW1 = np.dot(X_batch.T, delta1) / batch_size
        db1 = np.sum(delta1, axis=0, keepdims=True) / batch_size
        
        # 4. Update Parameters using SGD with Momentum
        
        # Update W3 and b3
        vW3 = momentum * vW3 + learning_rate * dW3
        vb3 = momentum * vb3 + learning_rate * db3
        W3_prev = W3.copy()
        b3_prev = b3.copy()
        W3 -= vW3
        b3 -= vb3
        update_magnitudes['W3'].append(np.linalg.norm(W3 - W3_prev))
        
        # Update W2 and b2
        vW2 = momentum * vW2 + learning_rate * dW2
        vb2 = momentum * vb2 + learning_rate * db2
        W2_prev = W2.copy()
        b2_prev = b2.copy()
        W2 -= vW2
        b2 -= vb2
        update_magnitudes['W2'].append(np.linalg.norm(W2 - W2_prev))
        
        # Update W1 and b1
        vW1 = momentum * vW1 + learning_rate * dW1
        vb1 = momentum * vb1 + learning_rate * db1
        W1_prev = W1.copy()
        b1_prev = b1.copy()
        W1 -= vW1
        b1 -= vb1
        update_magnitudes['W1'].append(np.linalg.norm(W1 - W1_prev))
    
    avg_epoch_loss = epoch_loss / num_batches
    loss_history.append(avg_epoch_loss)
    
    # Compute training accuracy on the full training set
    a1_full = sigmoid(np.dot(X, W1) + b1)
    a2_full = sigmoid(np.dot(a1_full, W2) + b2)
    a3_full = sigmoid(np.dot(a2_full, W3) + b3)
    accuracy = compute_accuracy(Y, a3_full)
    
    # Average weight update magnitude for display
    avg_update_W1 = np.mean(update_magnitudes['W1'])
    avg_update_W2 = np.mean(update_magnitudes['W2'])
    avg_update_W3 = np.mean(update_magnitudes['W3'])
    
    print(f"Epoch {epoch:2d} | Loss: {avg_epoch_loss:.4f} | Accuracy: {accuracy * 100:.2f}%")
    print(f"   Avg Weight Update - W1: {avg_update_W1:.4f}, W2: {avg_update_W2:.4f}, W3: {avg_update_W3:.4f}")


# Plot the Loss Curve

plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs + 1), loss_history, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Average Loss (MSE)")
plt.title("Loss Curve over Epochs (Learning Rate 0.01)")
plt.grid(True)
plt.show()


# RESULT

# Epoch  1 | Loss: 0.2462 | Accuracy: 28.00%
#          Avg Weight Update - W1: 0.0000, W2: 0.0000, W3: 0.0006

# Epoch  2 | Loss: 0.2461 | Accuracy: 28.00%
#          Avg Weight Update - W1: 0.0000, W2: 0.0001, W3: 0.0013

# Epoch  3 | Loss: 0.2457 | Accuracy: 28.00%
#          Avg Weight Update - W1: 0.0000, W2: 0.0000, W3: 0.0011

# Epoch  4 | Loss: 0.2455 | Accuracy: 28.00%
#          Avg Weight Update - W1: 0.0000, W2: 0.0000, W3: 0.0009

# Epoch  5 | Loss: 0.2453 | Accuracy: 28.00%
#          Avg Weight Update - W1: 0.0000, W2: 0.0000, W3: 0.0011

# Epoch  6 | Loss: 0.2451 | Accuracy: 28.00%
#          Avg Weight Update - W1: 0.0000, W2: 0.0000, W3: 0.0010

# Epoch  7 | Loss: 0.2450 | Accuracy: 28.00%
#          Avg Weight Update - W1: 0.0000, W2: 0.0000, W3: 0.0007

# Epoch  8 | Loss: 0.2449 | Accuracy: 28.00%
#          Avg Weight Update - W1: 0.0000, W2: 0.0000, W3: 0.0009

# Epoch  9 | Loss: 0.2449 | Accuracy: 28.00%
#          Avg Weight Update - W1: 0.0000, W2: 0.0000, W3: 0.0009

# Epoch 10 | Loss: 0.2448 | Accuracy: 28.00%
#          Avg Weight Update - W1: 0.0000, W2: 0.0000, W3: 0.0006