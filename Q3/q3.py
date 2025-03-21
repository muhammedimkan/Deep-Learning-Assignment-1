# 3. In the MLP implementation described in Question No:1, use binary cross entropy (BCE) loss instead of MSE. 
#    Display the loss and accuracy for MSE and BCE. Explain and justify your observation with loss and accuracy values.

import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Activation Functions and Derivatives

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_deriv(a):
    # This is used in MSE loss, but for BCE with sigmoid, the derivative simplifies.
    return a * (1 - a)


# Loss Functions
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def bce_loss(y_true, y_pred, epsilon=1e-8):
    return -np.mean(y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon))

# Helper Function: Accuracy

def compute_accuracy(y_true, y_pred):
    preds = (y_pred >= 0.5).astype(int)
    correct = np.all(preds == y_true, axis=1)
    return np.mean(correct)

# Hyperparameters and Network Architecture

input_dim = 3        # 3 input features
hidden_dim1 = 4      # 1st hidden layer: 4 neurons
hidden_dim2 = 4      # 2nd hidden layer: 4 neurons
output_dim = 2       # Output layer: 2 neurons

learning_rate = 0.1  # Using BCE, we set learning rate to 0.1
momentum = 0.9
epochs = 10
batch_size = 10
num_samples = 100

# Initialize Weights and Biases
W1 = np.random.randn(input_dim, hidden_dim1) * 0.1
b1 = np.random.randn(1, hidden_dim1) * 0.1

W2 = np.random.randn(hidden_dim1, hidden_dim2) * 0.1
b2 = np.random.randn(1, hidden_dim2) * 0.1

W3 = np.random.randn(hidden_dim2, output_dim) * 0.1
b3 = np.random.randn(1, output_dim) * 0.1

# Initialize momentum velocities
vW1 = np.zeros_like(W1)
vb1 = np.zeros_like(b1)
vW2 = np.zeros_like(W2)
vb2 = np.zeros_like(b2)
vW3 = np.zeros_like(W3)
vb3 = np.zeros_like(b3)

# Create Training Data
X = np.random.rand(num_samples, input_dim)
Y = np.random.randint(0, 2, size=(num_samples, output_dim))

# Training Loop Using BCE Loss
loss_history_bce = []  # to record BCE loss per epoch
loss_history_mse = []  # record MSE loss on predictions per epoch
accuracy_history = []  # record accuracy per epoch

num_batches = num_samples // batch_size

for epoch in range(1, epochs + 1):
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    X_shuffled = X[indices]
    Y_shuffled = Y[indices]
    
    epoch_loss = 0.0
    for b in range(num_batches):
        start = b * batch_size
        end = start + batch_size
        X_batch = X_shuffled[start:end]
        Y_batch = Y_shuffled[start:end]
        
        # Forward Pass
        z1 = np.dot(X_batch, W1) + b1
        a1 = sigmoid(z1)
        
        z2 = np.dot(a1, W2) + b2
        a2 = sigmoid(z2)
        
        z3 = np.dot(a2, W3) + b3
        a3 = sigmoid(z3)  # Network output
        
        # Compute BCE Loss for the mini-batch (for training)
        loss = bce_loss(Y_batch, a3)
        epoch_loss += loss
        
        # Backpropagation for BCE:
        # For BCE with sigmoid, derivative simplifies to (a3 - Y)
        delta3 = a3 - Y_batch
        
        # Gradients for output layer
        dW3 = np.dot(a2.T, delta3) / batch_size
        db3 = np.sum(delta3, axis=0, keepdims=True) / batch_size
        
        # Backpropagate to second hidden layer
        error2 = np.dot(delta3, W3.T)
        delta2 = error2 * sigmoid_deriv(a2)
        dW2 = np.dot(a1.T, delta2) / batch_size
        db2 = np.sum(delta2, axis=0, keepdims=True) / batch_size
        
        # Backpropagate to first hidden layer
        error1 = np.dot(delta2, W2.T)
        delta1 = error1 * sigmoid_deriv(a1)
        dW1 = np.dot(X_batch.T, delta1) / batch_size
        db1 = np.sum(delta1, axis=0, keepdims=True) / batch_size
        
        # Update Parameters using SGD with Momentum
        vW3 = momentum * vW3 + learning_rate * dW3
        vb3 = momentum * vb3 + learning_rate * db3
        W3 -= vW3
        b3 -= vb3
        
        vW2 = momentum * vW2 + learning_rate * dW2
        vb2 = momentum * vb2 + learning_rate * db2
        W2 -= vW2
        b2 -= vb2
        
        vW1 = momentum * vW1 + learning_rate * dW1
        vb1 = momentum * vb1 + learning_rate * db1
        W1 -= vW1
        b1 -= vb1

    # Average BCE loss over the epoch
    avg_epoch_loss_bce = epoch_loss / num_batches
    loss_history_bce.append(avg_epoch_loss_bce)
    
    # After epoch: Evaluate on full training set
    a1_full = sigmoid(np.dot(X, W1) + b1)
    a2_full = sigmoid(np.dot(a1_full, W2) + b2)
    a3_full = sigmoid(np.dot(a2_full, W3) + b3)
    
    # Compute both BCE and MSE losses on the entire training set
    epoch_loss_bce = bce_loss(Y, a3_full)
    epoch_loss_mse = mse_loss(Y, a3_full)
    loss_history_mse.append(epoch_loss_mse)
    
    # Compute accuracy (threshold at 0.5)
    acc = compute_accuracy(Y, a3_full)
    accuracy_history.append(acc)
    
    print(f"Epoch {epoch:2d} | BCE Loss: {epoch_loss_bce:.4f} | MSE Loss: {epoch_loss_mse:.4f} | Accuracy: {acc * 100:.2f}%")

# Plot the Loss Curve (BCE Loss)
plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs + 1), loss_history_bce, marker='o', label="BCE Loss")
plt.plot(range(1, epochs + 1), loss_history_mse, marker='x', label="MSE Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve over Epochs")
plt.legend()
plt.grid(True)
plt.show()

# Plot the Accuracy Curve

plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs + 1), np.array(accuracy_history) * 100, marker='s', color='green')
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy Curve over Epochs")
plt.grid(True)
plt.show()


# RESULT 

# Epoch  1 | BCE Loss: 0.6824 | MSE Loss: 0.2446 | Accuracy: 28.00%
# Epoch  2 | BCE Loss: 0.6824 | MSE Loss: 0.2447 | Accuracy: 28.00%
# Epoch  3 | BCE Loss: 0.6859 | MSE Loss: 0.2464 | Accuracy: 28.00%
# Epoch  4 | BCE Loss: 0.6828 | MSE Loss: 0.2449 | Accuracy: 28.00%
# Epoch  5 | BCE Loss: 0.6842 | MSE Loss: 0.2455 | Accuracy: 28.00%
# Epoch  6 | BCE Loss: 0.6827 | MSE Loss: 0.2448 | Accuracy: 28.00%
# Epoch  7 | BCE Loss: 0.6823 | MSE Loss: 0.2446 | Accuracy: 28.00%
# Epoch  8 | BCE Loss: 0.6845 | MSE Loss: 0.2457 | Accuracy: 28.00%
# Epoch  9 | BCE Loss: 0.6837 | MSE Loss: 0.2453 | Accuracy: 28.00%
# Epoch 10 | BCE Loss: 0.6853 | MSE Loss: 0.2461 | Accuracy: 28.00%