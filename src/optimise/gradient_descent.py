import sys
sys.path.insert(1, "src")

from datetime import datetime

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

# Define a function to optimise using Gradient Descent
def gradient_descent(func, X_init, lr=0.01, epochs=1000, method="sgd"):
    X = torch.tensor(X_init, dtype=torch.float32, requires_grad=True)
    optimiser = None

    if method == "sgd":
        optimiser = optim.SGD([X], lr=lr)
    elif method == "adam":
        optimiser = optim.Adam([X], lr=lr)
    
    for epoch in range(epochs+1):
        optimiser.zero_grad()
        y_pred = func(X)
        loss = F.mse_loss(y_pred, torch.tensor(0.0))
        
        if torch.isnan(loss) or torch.any(torch.isnan(X)):
            print(f"NaN encountered at epoch {epoch}. Stopping process.")
            return
        
        loss.backward()
        optimiser.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}, X = {X.detach().numpy()}")

    return X.detach().numpy()

if __name__ == "__main__":    
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    res = {}
    
    for name, (func, true_min) in functions.items():
        print(f"\nTesting {name}")

        X_init = np.random.uniform(-1.5, 1.5, size=(2))
        
        print()
        print("Testing SGD")
        sgd_optimised_X = gradient_descent(func, X_init, lr=1e-7, epochs=25000, method="sgd")
        print()
        print("Testing Adam")
        adam_optimised_X = gradient_descent(func, X_init, lr=1e-3, epochs=25000, method="adam")
        
        print()
        print(f"SGD Optimised X: {sgd_optimised_X}")        
        print(f"Adam Optimised X: {adam_optimised_X}")
        print(f"Actual Minimum for {name}: {true_min}")
        
        print("bong")
        
        res[name] = {
            "SGD": ([round(v, 5) for v in sgd_optimised_X.tolist()], round(func(torch.from_numpy(sgd_optimised_X)).item(), 5)),
            "Adam": ([round(v, 5) for v in adam_optimised_X.tolist()], round(func(torch.from_numpy(adam_optimised_X)).item(), 5)),
            "Actual": (true_min.tolist(), round(func(torch.from_numpy(true_min)).item(), 5))
        }
        
    print()
    
    res_path = f"results/gradient_descent/{now}.json"
    
    print(f"Saving Results to {res_path}")
    with open(res_path, "w+") as f:
        f.write(json.dumps(res, ignore_nan=True, indent=4))