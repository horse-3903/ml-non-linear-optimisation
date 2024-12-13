import os

import simplejson as json

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from functions import rosenbrock, rastrigin, ackley

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.model(x)

# Define a function to optimise using Gradient Descent
def gradient_descent(model, X_init, lr=0.01, epochs=1000, method="sgd"):
    X = torch.tensor(X_init, dtype=torch.float32, requires_grad=True)
    optimiser = None

    if method == "sgd":
        optimiser = optim.SGD([X], lr=lr)
    elif method == "adam":
        optimiser = optim.Adam([X], lr=lr)
    
    for epoch in range(epochs+1):
        optimiser.zero_grad()
        y_pred = model(X)
        loss = F.mse_loss(y_pred, torch.tensor([0.0]))
        loss.backward()
        optimiser.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}, X = {X.detach().numpy()}")

    return X.detach().numpy()

if __name__ == "__main__":
    input_dim = 2
    hidden_dim = 128
    output_dim = 1
    
    model_lst = sorted(os.listdir('models/'))
    model_name = model_lst[-1]
    model_parent_dir = f"models/{model_name}"
    
    # Define functions and their minima
    functions = {
        "Rosenbrock": (lambda x: rosenbrock(x.clone()), np.array([1, 1])),  # Minimum is at (1, 1)
        "Rastrigin": (lambda x: rastrigin(x.clone()), np.array([0, 0])),  # Minimum is at (0, 0)
        "Ackley": (lambda x: ackley(x.clone()), np.array([0, 0]))  # Minimum is at (0, 0)
    }
    
    res = {}
    
    for name, (func, true_min) in functions.items():
        print(f"\nTesting {name} model")
        
        model_path = f"{model_parent_dir}/{name.lower()}_model.pt"
        model = MLP(input_dim, hidden_dim, output_dim)
        model.load_state_dict(torch.load(model_path, weights_only=True))

        X_init = np.random.uniform(-5, 5, size=(2))
        
        print()
        print("Testing SGD")
        sgd_optimised_X = gradient_descent(model, X_init, lr=1e-8, epochs=25000, method="sgd")
        print()
        print("Testing Adam")
        adam_optimised_X = gradient_descent(model, X_init, lr=1e-3, epochs=15000, method="adam")
        
        print()
        print(f"SGD Optimised X: {sgd_optimised_X}")        
        print(f"Adam Optimised X: {adam_optimised_X}")
        print(f"Actual Minimum for {name}: {true_min}")
        
        res[name] = {
            "SGD": (sgd_optimised_X.tolist(), round(func(torch.from_numpy(sgd_optimised_X)), 5)),
            "Adam": (adam_optimised_X.tolist(), round(func(torch.from_numpy(adam_optimised_X)), 5)),
            "Actual": (true_min.tolist(), round(func(torch.from_numpy(true_min)), 5))
        }
        
    print()
    
    res_path = f"results/gradient_descent/{model_name}.json"
    
    print(f"Saving Results to {res_path}")
    with open(res_path, "w+") as f:
        f.write(json.dumps(res, ignore_nan=True, indent=4))