import sys
sys.path.insert(1, "src")

import os

import numpy as np

import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from util.functions import rosenbrock, rastrigin, ackley

functions = {
    "Rosenbrock": lambda x: rosenbrock(x.clone()),
    "Rastrigin": lambda x: rastrigin(x.clone()),
    "Ackley": lambda x: ackley(x.clone())
}

minimum_points = {
    "Rosenbrock": np.array([1.0, 1.0]),
    "Rastrigin": np.array([0.0, 0.0]),
    "Ackley": np.array([0.0, 0.0])
}

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

    # Compute Error Heatmap
    error = np.abs(predictions - true_values)

    # Compute Mean Squared Error and Mean Absolute Error
    mse = np.mean((predictions - true_values) ** 2)
    mae = np.mean(np.abs(predictions - true_values))
    
    print(f"Average MSE for {name} model: {mse:.4f}")
    print(f"Average MAE for {name} model: {mae:.4f}")
    
    # Get minimum point
    min_point = minimum_points[name]
    min_value = model(torch.tensor(min_point, dtype=torch.float32)).detach().numpy()
    print(f"Function value at the minimum for {name}: {min_value[0]:.4f}")

    # Plot
    fig = plt.figure(figsize=(12, 5))

    # True function
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(X, Y, true_values, cmap='viridis', alpha=0.8)
    ax1.set_title(f"True Function: {name}", fontsize=10)
    ax1.set_xlabel('X1', fontsize=8)
    ax1.set_ylabel('X2', fontsize=8)
    ax1.set_zlabel('Output', fontsize=8)
    ax1.tick_params(axis='both', which='major', labelsize=7)

    # Predicted function
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_surface(X, Y, predictions, cmap='plasma', alpha=0.8)
    ax2.set_title(f"Predicted Function: {name}", fontsize=10)
    ax2.set_xlabel('X1', fontsize=8)
    ax2.set_ylabel('X2', fontsize=8)
    ax2.set_zlabel('Output', fontsize=8)
    ax2.tick_params(axis='both', which='major', labelsize=7)

    # Error Heatmap (3D)
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot_surface(X, Y, error, cmap='coolwarm', alpha=0.8)
    ax3.set_title(f"Error Heatmap: {name}", fontsize=10)
    ax3.set_xlabel('X1', fontsize=8)
    ax3.set_ylabel('X2', fontsize=8)
    ax3.set_zlabel('Absolute Error', fontsize=8)
    ax3.tick_params(axis='both', which='major', labelsize=7)
    
    plt.show()

# Main script
if __name__ == "__main__":
    input_dim = 2
    hidden_dim = 128
    output_dim = 1
    
    model_lst = sorted(os.listdir('models/'))
    model_name = model_lst[-1]
    
    # for model_name in model_lst:
    model_parent_dir = f"models/{model_name}"
    print(f"Testing Model Record {model_name}")
    
    for name, func in functions.items():
        print(f"\nTesting {name} Model")

        # Load the saved model
        model_path = f"{model_parent_dir}/{name.lower()}_model.pt"
        model = MLP(input_dim, hidden_dim, output_dim)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        
        # Compare predictions and true values
        plot_comparison(func, model, name)