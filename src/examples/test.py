import sys
sys.path.insert(1, "src")

from datetime import datetime
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

from util.functions import rosenbrock, rastrigin, ackley

# Define the optimisation function using Differential Evolution
def evolutionary_algo(func, bounds, strategy='best1bin', maxiter=80, popsize=200):
    best_function_values = []  # To store the best function value at each generation
    
    def callback(xk, convergence):
        # Compute and store the current function value at 'xk'
        best_function_values.append(func(xk))
    
    result = differential_evolution(
        func, bounds, strategy=strategy, maxiter=maxiter, popsize=popsize, tol=1e-7,
        seed=None, disp=True, callback=callback
    )
    return result.x, result.fun, best_function_values

# Plot for a single function
def plot_single_convergence(values, title, ax):
    ax.plot(values, label="Rosenbrock Function", color="blue")
    ax.set_xlabel('Generations')
    ax.set_ylabel('Function Value')
    ax.legend()
    ax.grid(True)

# Combined Convergence Plot
def plot_combined_convergence(convergence_data, ax):
    ax.plot(convergence_data["Rastrigin"], label="Rastrigin Function", color="orange")
    ax.plot(convergence_data["Ackley"], label="Ackley Function", color="green")
    ax.set_xlabel('Generations')
    ax.set_ylabel('Function Value')
    ax.legend()
    ax.grid(True)

# Define functions and their minima
functions = {
    "Rosenbrock": (lambda x: rosenbrock(torch.tensor(x)), np.array([1, 1])),
    "Rastrigin": (lambda x: rastrigin(torch.tensor(x)), np.array([0, 0])),
    "Ackley": (lambda x: ackley(torch.tensor(x)), np.array([0, 0]))
}

if __name__ == "__main__":
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    res = {}
    
    # Rosenbrock (Separate)
    print("\nTesting Rosenbrock using Evolutionary Optimisation")
    bounds = [(-5.0, 5.0), (-5.0, 5.0)]
    func, true_min = functions["Rosenbrock"]
    
    def wrapped_func(x):
        return func(x)
    
    optimised_X, optimised_value, function_values = evolutionary_algo(wrapped_func, bounds)
    
    print(f"Optimised X: {optimised_X}")
    print(f"Actual Minimum for Rosenbrock: {true_min}")
    
    res["Rosenbrock"] = {
        "Optimised": ([round(v, 5) for v in optimised_X], round(optimised_value, 5)),
        "Actual": (true_min.tolist(), round(func(true_min), 5))
    }
    
    # Create the figure for the combined plot
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot Rosenbrock separately
    plot_single_convergence(function_values, "Function Value Convergence for Rosenbrock Function", axs[0])
    
    # Rastrigin and Ackley (Combined)
    convergence_data = {"Rastrigin": [], "Ackley": []}  # Initialize empty lists for both
    
    for name in ["Rastrigin", "Ackley"]:
        print(f"\nTesting {name} using Evolutionary Optimisation")
        func, true_min = functions[name]

        def wrapped_func(x):
            return func(x)
        
        optimised_X, optimised_value, function_values = evolutionary_algo(wrapped_func, bounds)
        
        print(f"Optimised X: {optimised_X}")
        print(f"Actual Minimum for {name}: {true_min}")
        
        res[name] = {
            "Optimised": ([round(v, 5) for v in optimised_X], round(optimised_value, 5)),
            "Actual": (true_min.tolist(), round(func(true_min), 5))
        }
        
        convergence_data[name] = function_values  # Store the convergence data for Rastrigin and Ackley
    
    # Combined plot for Rastrigin and Ackley
    plot_combined_convergence(convergence_data, axs[1])
    
    # Common title at the top of the top plot
    fig.suptitle('Optimisation Convergence for Evolutionary Algorithm', fontsize=16)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Show the plot
    plt.show()
