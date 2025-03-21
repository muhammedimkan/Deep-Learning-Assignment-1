# 5. Implement an MLP with 3 inputs, 2 hidden layers with 8 neurons each and an output layer is scalar. 
#    ReLU activation used in hidden layers and sigmoid used in the output layer. Use adam optimizer and binary cross entropy loss. 
#    Use three parameter initialization methods like Xavier, He, Normal. Display the loss and accuracy values for each method.


import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Activation Functions
def relu(z):
    return np.maximum(0, z)

def relu_deriv(z):
    return (z > 0).astype(float)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Loss Functions

def bce_loss(y_true, y_pred, epsilon=1e-8):
    return -np.mean(y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon))

def compute_accuracy(y_true, y_pred):
    preds = (y_pred >= 0.5).astype(int)
    return np.mean(preds == y_true)


# Parameter Initialization Methods

def init_xavier(fan_in, fan_out):
    limit = np.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, (fan_in, fan_out))

def init_he(fan_in, fan_out):
    std = np.sqrt(2 / fan_in)
    return np.random.randn(fan_in, fan_out) * std

def init_normal(fan_in, fan_out, std=0.01):
    return np.random.randn(fan_in, fan_out) * std


# Adam Optimizer Update Function

def adam_update(param, grad, m, v, t, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad ** 2)
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    param = param - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    return param, m, v

# MLP Architecture Parameters

input_dim = 3
hidden_dim1 = 8
hidden_dim2 = 8
output_dim = 1  # Scalar output

# Training hyperparameters
learning_rate = 0.001  # Typical learning rate for Adam
epochs = 10
batch_size = 10
num_samples = 100

# Generate random training data
X = np.random.rand(num_samples, input_dim)
# Binary target as scalar (0 or 1)
Y = np.random.randint(0, 2, size=(num_samples, 1))

# Function to train the MLP given an initialization method for weights
def train_mlp(initializer):
    # Initialize weights using the provided initializer function
    # For each layer, determine fan_in and fan_out.
    W1 = initializer(input_dim, hidden_dim1)
    b1 = np.zeros((1, hidden_dim1))
    W2 = initializer(hidden_dim1, hidden_dim2)
    b2 = np.zeros((1, hidden_dim2))
    W3 = initializer(hidden_dim2, output_dim)
    b3 = np.zeros((1, output_dim))
    
    # Initialize Adam moments (for each weight and bias)
    mW1 = np.zeros_like(W1); vW1 = np.zeros_like(W1)
    mb1 = np.zeros_like(b1); vb1 = np.zeros_like(b1)
    mW2 = np.zeros_like(W2); vW2 = np.zeros_like(W2)
    mb2 = np.zeros_like(b2); vb2 = np.zeros_like(b2)
    mW3 = np.zeros_like(W3); vW3 = np.zeros_like(W3)
    mb3 = np.zeros_like(b3); vb3 = np.zeros_like(b3)
    
    loss_history = []
    accuracy_history = []
    t = 1  # time step for Adam
    
    num_batches = num_samples // batch_size
    
    for epoch in range(epochs):
        # Shuffle data
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
            
            # Forward pass
            Z1 = np.dot(X_batch, W1) + b1
            A1 = relu(Z1)
            
            Z2 = np.dot(A1, W2) + b2
            A2 = relu(Z2)
            
            Z3 = np.dot(A2, W3) + b3
            A3 = sigmoid(Z3)
            
            # Compute BCE loss for this batch
            loss = bce_loss(Y_batch, A3)
            epoch_loss += loss
            
            # Backward pass
            # Output layer delta: derivative of BCE with sigmoid simplifies to (A3 - Y)
            dZ3 = A3 - Y_batch  # shape (batch_size, 1)
            dW3 = np.dot(A2.T, dZ3) / batch_size
            db3 = np.sum(dZ3, axis=0, keepdims=True) / batch_size
            
            dA2 = np.dot(dZ3, W3.T)
            dZ2 = dA2 * relu_deriv(Z2)
            dW2 = np.dot(A1.T, dZ2) / batch_size
            db2 = np.sum(dZ2, axis=0, keepdims=True) / batch_size
            
            dA1 = np.dot(dZ2, W2.T)
            dZ1 = dA1 * relu_deriv(Z1)
            dW1 = np.dot(X_batch.T, dZ1) / batch_size
            db1 = np.sum(dZ1, axis=0, keepdims=True) / batch_size
            
            # Adam update for each parameter
            W3, mW3, vW3 = adam_update(W3, dW3, mW3, vW3, t, learning_rate)
            b3, mb3, vb3 = adam_update(b3, db3, mb3, vb3, t, learning_rate)
            
            W2, mW2, vW2 = adam_update(W2, dW2, mW2, vW2, t, learning_rate)
            b2, mb2, vb2 = adam_update(b2, db2, mb2, vb2, t, learning_rate)
            
            W1, mW1, vW1 = adam_update(W1, dW1, mW1, vW1, t, learning_rate)
            b1, mb1, vb1 = adam_update(b1, db1, mb1, vb1, t, learning_rate)
            
            t += 1
            
        # Average loss for epoch
        avg_epoch_loss = epoch_loss / num_batches
        loss_history.append(avg_epoch_loss)
        
        # Compute full forward pass for evaluation
        Z1_full = np.dot(X, W1) + b1
        A1_full = relu(Z1_full)
        Z2_full = np.dot(A1_full, W2) + b2
        A2_full = relu(Z2_full)
        Z3_full = np.dot(A2_full, W3) + b3
        A3_full = sigmoid(Z3_full)
        
        epoch_acc = compute_accuracy(Y, A3_full)
        accuracy_history.append(epoch_acc)
        
        print(f"Epoch {epoch+1:2d} | Loss: {avg_epoch_loss:.4f} | Accuracy: {epoch_acc*100:.2f}%")
    
    return loss_history, accuracy_history


# Train using different initialization methods

methods = {
    "Xavier": init_xavier,
    "He": init_he,
    "Normal": init_normal
}

results = {}

for name, initializer in methods.items():
    print(f"\nTraining with {name} Initialization:")
    loss_hist, acc_hist = train_mlp(initializer)
    results[name] = (loss_hist, acc_hist)


# Plot Loss and Accuracy for each method

plt.figure(figsize=(10, 4))
for name in results:
    plt.plot(range(1, epochs+1), results[name][0], marker='o', label=name)
plt.xlabel("Epoch")
plt.ylabel("BCE Loss")
plt.title("Loss Curve for Different Initialization Methods")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 4))
for name in results:
    plt.plot(range(1, epochs+1), np.array(results[name][1])*100, marker='s', label=name)
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy Curve for Different Initialization Methods")
plt.legend()
plt.grid(True)
plt.show()


# RESULT 

# 1. Training with Xavier Initialization:

#     Epoch  1 | Loss: 0.7050 | Accuracy: 45.00%
#     Epoch  2 | Loss: 0.7018 | Accuracy: 45.00%
#     Epoch  3 | Loss: 0.6979 | Accuracy: 46.00%
#     Epoch  4 | Loss: 0.6961 | Accuracy: 51.00%
#     Epoch  5 | Loss: 0.6945 | Accuracy: 44.00%
#     Epoch  6 | Loss: 0.6935 | Accuracy: 53.00%
#     Epoch  7 | Loss: 0.6919 | Accuracy: 54.00%
#     Epoch  8 | Loss: 0.6914 | Accuracy: 54.00%
#     Epoch  9 | Loss: 0.6901 | Accuracy: 54.00%
#     Epoch 10 | Loss: 0.6898 | Accuracy: 54.00%

# 2. Training with He Initialization:

#     Epoch  1 | Loss: 0.6846 | Accuracy: 61.00%
#     Epoch  2 | Loss: 0.6843 | Accuracy: 61.00%
#     Epoch  3 | Loss: 0.6843 | Accuracy: 61.00%
#     Epoch  4 | Loss: 0.6833 | Accuracy: 61.00%
#     Epoch  5 | Loss: 0.6839 | Accuracy: 60.00%
#     Epoch  6 | Loss: 0.6829 | Accuracy: 60.00%
#     Epoch  7 | Loss: 0.6825 | Accuracy: 60.00%
#     Epoch  8 | Loss: 0.6821 | Accuracy: 60.00%
#     Epoch  9 | Loss: 0.6818 | Accuracy: 60.00%
#     Epoch 10 | Loss: 0.6815 | Accuracy: 60.00%

# 3. Training with Normal Initialization:

#     Epoch  1 | Loss: 0.6931 | Accuracy: 54.00%
#     Epoch  2 | Loss: 0.6929 | Accuracy: 54.00%
#     Epoch  3 | Loss: 0.6927 | Accuracy: 54.00%
#     Epoch  4 | Loss: 0.6926 | Accuracy: 54.00%
#     Epoch  5 | Loss: 0.6925 | Accuracy: 54.00%
#     Epoch  6 | Loss: 0.6924 | Accuracy: 54.00%
#     Epoch  7 | Loss: 0.6923 | Accuracy: 54.00%
#     Epoch  8 | Loss: 0.6922 | Accuracy: 54.00%
#     Epoch  9 | Loss: 0.6921 | Accuracy: 54.00%
#     Epoch 10 | Loss: 0.6920 | Accuracy: 54.00%