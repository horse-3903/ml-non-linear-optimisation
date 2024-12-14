import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

pops = []

def worker(func, x):
    pops.append(np.copy(x))
    return func(x.T)

# Define the optimisation function using Differential Evolution
def evolutionary_optimisation(func, bounds, strategy='best1bin', maxiter=1000, popsize=10):
    optimiser = differential_evolution(
        func, bounds, strategy=strategy,
        maxiter=maxiter, popsize=popsize, tol=1e-7, seed=None, disp=True,
        workers=worker,
    )
    return optimiser.x, optimiser.fun

def func(x):
    return np.sin(x / 2) * np.cos(x / 2) + 0.5 * np.sin(x) - 0.5 * np.cos(x) + 2 * np.sin(x / 5)

if __name__ == "__main__":

    res = {}

    # Define the bounds for the optimisation
    bounds = [(0, 50)]
    
    optimised_X, optimised_value = evolutionary_optimisation(func, bounds)
    
    print(f"Optimised X: {optimised_X}")
    print(f"Optimized X corresponds: {func(optimised_X)}")
    print(pops)
    input()

    # Plot the graph
    x = np.linspace(0, 50, 1000)
    y = func(x)
    for pop in pops:
        
        manager = plt.get_current_fig_manager()
        manager.window.wm_geometry("+100+100")

        plt.plot(x, y, label="Differential Evolution")
        plt.scatter(pop, func(pop), color="red", zorder=5)
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.grid()
        plt.show()