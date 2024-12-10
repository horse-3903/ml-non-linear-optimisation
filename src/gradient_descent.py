# gradient_descent_optimization.py

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from functions import rastrigin, ackley  # Import the functions

# Define a function to optimize using Gradient Descent
def gradient_descent(model, X_init, lr=0.01, epochs=1000, method="sgd"):
    X = torch.tensor(X_init, dtype=torch.float32, requires_grad=True)
    optimiser = None

    if method == "sgd":
        optimiser = optim.SGD([X], lr=lr)
    elif method == "adam":
        optimiser = optim.Adam([X], lr=lr)
    
    for epoch in range(epochs):
        optimiser.zero_grad()
        y_pred = model(X)
        loss = F.mse_loss(y_pred, torch.tensor([0.0]))  # Minimize the output value (we aim to approach the global minimum)
        loss.backward()
        optimiser.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}, X = {X.detach().numpy()}")

    return X.detach().numpy()

if __name__ == "__main__":
    # Load the trained model for Rastrigin (example)
    model_path = 'models/rastrigin_model.pt'
    model = torch.load(model_path)
    model.eval()

    X_init = np.random.uniform(-5, 5, size=(2))  # Initial guess
    optimized_X = gradient_descent(model, X_init, lr=0.01, epochs=1000, method="adam")
    print(f"Optimized X: {optimized_X}")
