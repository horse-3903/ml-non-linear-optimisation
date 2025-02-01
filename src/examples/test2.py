import sys
sys.path.insert(1, "src")

from datetime import datetime
import numpy as np
import torch
import matplotlib.pyplot as plt

from skopt import gp_minimize

from util.functions import rosenbrock, rastrigin, ackley

# Define the optimisation function for Bayesian optimisation
def bayesian_optimisation(func, bounds, n_calls=150):
    result = gp_minimize(func, bounds, n_calls=n_calls, verbose=True)
    return result

# Single Convergence Plot
def plot_single_convergence(values, title, ax):
    ax.plot(values, label="Rosenbrock Function", color="blue")
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Function Value')
    ax.legend()
    ax.grid(True)

# Combined Convergence Plot
def plot_combined_convergence(convergence_data, ax):
    ax.plot(convergence_data["Rastrigin"], label="Rastrigin Function", color="orange")
    ax.plot(convergence_data["Ackley"], label="Ackley Function", color="green")
    ax.set_xlabel('Iterations')
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
    print("\nTesting Rosenbrock using Bayesian Optimisation")
    func, true_min = functions["Rosenbrock"]
    bounds = [(-5.0, 5.0), (-5.0, 5.0)]

    results = bayesian_optimisation(func, bounds)
    optimised_X, optimised_value = results.x, results.fun

    print(f"Optimised X: {optimised_X}")
    print(f"Actual Minimum for Rosenbrock: {true_min}")

    res["Rosenbrock"] = {
        "Optimised": ([round(v, 5) for v in optimised_X], round(optimised_value, 5)),
        "Actual": (true_min.tolist(), round(func(true_min), 5))
    }

    # Create the figure for the combined plot
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))

    # Plot Rosenbrock separately
    plot_single_convergence(results.func_vals, "Function Value Convergence for Rosenbrock Function", axs[0])

    # Rastrigin and Ackley (Combined)
    convergence_data = {"Rastrigin": [], "Ackley": []}  # Initialize empty lists for both
    
    for name in ["Rastrigin", "Ackley"]:
        print(f"\nTesting {name} using Bayesian Optimisation")
        func, true_min = functions[name]

        results = bayesian_optimisation(func, bounds)
        optimised_X, optimised_value = results.x, results.fun

        print(f"Optimised X: {optimised_X}")
        print(f"Actual Minimum for {name}: {true_min}")

        res[name] = {
            "Optimised": ([round(v, 5) for v in optimised_X], round(optimised_value, 5)),
            "Actual": (true_min.tolist(), round(func(true_min), 5))
        }

        convergence_data[name] = results.func_vals

    # Combined plot for Rastrigin and Ackley
    plot_combined_convergence(convergence_data, axs[1])

    # Common title at the top of the top plot
    fig.suptitle('Optimisation Convergence for Bayesian Optimisation', fontsize=16)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Show the plot
    plt.show()