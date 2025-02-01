import sys
sys.path.insert(1, "src")

from datetime import datetime
import matplotlib.pyplot as plt
import simplejson as json
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from util.functions_pt import rosenbrock, rastrigin, ackley

# Define functions and their minima
functions = {
    "Rosenbrock": (lambda x: rosenbrock(x.clone()), np.array([1, 1])),  # Minimum is at (1, 1)
    "Rastrigin": (lambda x: rastrigin(x.clone()), np.array([0, 0])),  # Minimum is at (0, 0)
    "Ackley": (lambda x: ackley(x.clone()), np.array([0, 0]))  # Minimum is at (0, 0)
}

# Track average losses across functions
sgd_losses = []
adam_losses = []

# Normalize the loss values before averaging
def normalize_losses(losses):
    min_loss = np.min(losses)
    max_loss = np.max(losses)
    if max_loss > min_loss:
        return (losses - min_loss) / (max_loss - min_loss)
    return losses  # No normalization needed if the range is 0 (constant losses)

# Define a function to optimise using Gradient Descent
def gradient_descent(func, X_init, lr=0.01, epochs=1000, method="sgd", loss_log=None):
    X = torch.tensor(X_init, dtype=torch.float32, requires_grad=True)
    optimiser = None

    if method == "sgd":
        optimiser = optim.SGD([X], lr=lr)
    elif method == "adam":
        optimiser = optim.Adam([X], lr=lr)
    
    losses = []

    for epoch in range(epochs+1):
        optimiser.zero_grad()
        y_pred = func(X)
        loss = F.mse_loss(y_pred, torch.tensor(0.0))
        
        if torch.isnan(loss) or torch.any(torch.isnan(X)):
            raise Exception(f"NaN encountered at epoch {epoch}. Stopping process.")
        
        loss.backward()
        optimiser.step()
        
        losses.append(loss.item())  # Record the loss

        if epoch % 1000 == 0:  # Log every 1000 epochs
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}, X = {X.detach().numpy()}")

    # Normalize the losses before logging
    normalized_losses = normalize_losses(np.array(losses))
    loss_log.append(normalized_losses)

    return X.detach().numpy()

if __name__ == "__main__":    
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    res = {}
    
    for name, (func, true_min) in functions.items():
        print(f"\nTesting {name}")

        X_init = np.random.uniform(-1.5, 1.5, size=(2))
        
        print()
        print("Testing SGD")
        gradient_descent(func, X_init, lr=1e-7, epochs=5000, method="sgd", loss_log=sgd_losses)
        
        print()
        print("Testing Adam")
        gradient_descent(func, X_init, lr=1e-3, epochs=5000, method="adam", loss_log=adam_losses)
        
    # Average the normalized losses over all functions
    avg_sgd_losses = np.mean(np.array(sgd_losses), axis=0)
    avg_adam_losses = np.mean(np.array(adam_losses), axis=0)
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(avg_adam_losses, label="Adam")
    plt.plot(avg_sgd_losses, label="SGD")
    plt.xlabel("Iteration")
    plt.ylabel("Normalised Loss")
    plt.title("Comparison of SGD and Adam Optimisers (Normalised Loss)")
    plt.legend()
    plt.grid()
    plt.show()