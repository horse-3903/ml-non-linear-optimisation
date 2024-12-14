import os

import torch
import numpy as np
import simplejson as json
from scipy.optimize import differential_evolution

from functions import rosenbrock, rastrigin, ackley

# Define the optimisation function using Differential Evolution
def evolutionary_optimisation(func, bounds, strategy='best1bin', maxiter=1000, popsize=15):
    result = differential_evolution(
        func, bounds, strategy=strategy, maxiter=maxiter, popsize=popsize, tol=1e-7, seed=None, disp=True
    )
    return result.x, result.fun

# Define functions and their minima
functions = {
    "Rosenbrock": (lambda x: rosenbrock(torch.tensor(x)), np.array([1, 1])),  # Minimum is at (1, 1)
    "Rastrigin": (lambda x: rastrigin(torch.tensor(x)), np.array([0, 0])),  # Minimum is at (0, 0)
    "Ackley": (lambda x: ackley(torch.tensor(x)), np.array([0, 0]))  # Minimum is at (0, 0)
}

if __name__ == "__main__":
    
    model_lst = sorted(os.listdir('models/'))
    model_name = model_lst[-1]
    model_parent_dir = f"models/{model_name}"

    res = {}
    
    for name, (func, true_min) in functions.items():
        
        print(f"\nTesting {name} using Evolutionary Optimisation")

        # Define the bounds for the optimisation
        bounds = [(-5.0, 5.0), (-5.0, 5.0)]
        
        # Perform Evolutionary optimisation
        def wrapped_func(x):
            return func(x)
        
        optimised_X, optimised_value = evolutionary_optimisation(wrapped_func, bounds)
        
        print(f"Optimised X: {optimised_X}")
        print(f"Actual Minimum for {name}: {true_min}")
        
        res[name] = {
            "Optimised": (optimised_X.tolist(), float(round(optimised_value, 5))),
            "Actual": (true_min.tolist(), round(func(true_min), 5))
        }

    print()
    
    res_path = f"results/evolutionary_optimisation/{model_name}.json"
    
    print(f"Saving Results to {res_path}")
    with open(res_path, "w+") as f:
        f.write(json.dumps(res, ignore_nan=True, indent=4))
