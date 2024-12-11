import os

import simplejson as json

import numpy as np

import torch

from functions import rosenbrock, rastrigin, ackley

from skopt import gp_minimize

# Define functions and their minima
functions = {
    "Rosenbrock": (lambda x: rosenbrock(torch.tensor(x)), np.array([1, 1])),  # Minimum is at (1, 1)
    "Rastrigin": (lambda x: rastrigin(torch.tensor(x)), np.array([0, 0])),  # Minimum is at (0, 0)
    "Ackley": (lambda x: ackley(torch.tensor(x)), np.array([0, 0]))  # Minimum is at (0, 0)
}

# Define the optimisation function for Bayesian optimisation
def bayesian_optimisation(func, bounds, n_calls=50):
    result = gp_minimize(func, bounds, n_calls=n_calls, verbose=True)
    return result.x, result.fun

if __name__ == "__main__":
    model_lst = sorted(os.listdir('models/'))
    model_parent_dir = f"models/{model_lst[-1]}"

    res = {}
    
    for name, (func, true_min) in functions.items():
        print(f"\nTesting {name} using Bayesian optimisation")
        
        model_path = f"{model_parent_dir}/{name.lower()}_model.pt"

        # Define the bounds for the optimisation
        bounds = [(-5.0, 5.0), (-5.0, 5.0)]
        
        # Perform Bayesian optimisation
        def wrapped_func(x):
            return func(x)
        
        optimised_X, optimised_value = bayesian_optimisation(wrapped_func, bounds)
        
        print(f"Optimised X: {optimised_X}")
        print(f"Actual Minimum for {name}: {true_min}")
        
        res[name] = {
            "Optimised": (optimised_X, round(optimised_value, 5)),
            "Actual": (true_min.tolist(), round(func(true_min), 5))
        }

    print()
    
    res_path = f"results/bayesian_optimisation/{model_lst[-1]}.json"
    
    print(f"Saving Results to {res_path}")
    with open(res_path, "w+") as f:
        f.write(json.dumps(res, ignore_nan=True, indent=4))