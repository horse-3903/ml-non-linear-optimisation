import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from math import pi
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import rosen

# Define the Rastrigin function
def rastrigin(x):
    A = 10
    return A * len(x) + sum([(xi**2 - A * torch.cos(2 * pi * xi)).item() for xi in x])

# Define the Ackley function
def ackley(x):
    a = 20
    b = 0.2
    c = 2 * pi
    n = len(x)
    sum1 = sum(xi**2 for xi in x)
    sum2 = sum(torch.cos(c * xi) for xi in x)
    term1 = -a * torch.exp(-b * torch.sqrt(sum1 / n))
    term2 = -torch.exp(sum2 / n)
    return term1 + term2 + a + torch.exp(torch.tensor(1.0))

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

# Generate data and plot predictions
def plot_comparison(func, model, name, range_min=-5, range_max=5, resolution=100):
    x = np.linspace(range_min, range_max, resolution)
    y = np.linspace(range_min, range_max, resolution)
    X, Y = np.meshgrid(x, y)
    points = torch.tensor(np.c_[X.ravel(), Y.ravel()], dtype=torch.float32)

    # Get model predictions
    model.eval()
    predictions = model(points).detach().numpy().reshape(X.shape)

    # Get true function values
    true_values = np.array([func(point) for point in points]).reshape(X.shape)

    # Compute Mean Squared Error
    mse = np.mean((predictions - true_values) ** 2)
    print(f"Average MSE for {name} model: {mse:.4f}")

    # Plot
    fig = plt.figure(figsize=(12, 6))

    # True function
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X, Y, true_values, cmap='viridis', alpha=0.8)
    ax1.set_title(f"True Function: {name}")
    ax1.set_xlabel('X1')
    ax1.set_ylabel('X2')
    ax1.set_zlabel('Output')

    # Predicted function
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(X, Y, predictions, cmap='plasma', alpha=0.8)
    ax2.set_title(f"Predicted Function: {name}")
    ax2.set_xlabel('X1')
    ax2.set_ylabel('X2')
    ax2.set_zlabel('Output')

    plt.show()

# Main script
if __name__ == "__main__":
    input_dim = 2
    hidden_dim = 128
    output_dim = 1

    functions = {
        "Rosenbrock": lambda x: rosen(x),
        "Rastrigin": lambda x: rastrigin(x.clone()),
        "Ackley": lambda x: ackley(x.clone())
    }
    
    for name, func in functions.items():
        print(f"\nTesting {name} model")

        # Load the saved model
        model_path = f"models/{name.lower()}_model.pt"
        model = MLP(input_dim, hidden_dim, output_dim)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        
        # Compare predictions and true values
        plot_comparison(func, model, name)