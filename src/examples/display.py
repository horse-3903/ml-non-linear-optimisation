import sys
sys.path.insert(1, "src")

import torch
import matplotlib.pyplot as plt

from util.functions import rosenbrock, rastrigin, ackley

if __name__ == "__main__":
    # Generate a mesh grid using PyTorch
    x = torch.linspace(-5, 5, 100)
    y = torch.linspace(-5, 5, 100)
    x, y = torch.meshgrid(x, y, indexing="ij")

    # Calculate function values for the grid
    z_rosenbrock = torch.tensor([[rosenbrock(torch.tensor([xi, yi])) for xi, yi in zip(x_row, y_row)] for x_row, y_row in zip(x, y)])
    z_rastrigin = torch.tensor([[rastrigin(torch.tensor([xi, yi])) for xi, yi in zip(x_row, y_row)] for x_row, y_row in zip(x, y)])
    z_ackley = torch.tensor([[ackley(torch.tensor([xi, yi])) for xi, yi in zip(x_row, y_row)] for x_row, y_row in zip(x, y)])

    # Convert to numpy arrays for plotting
    z_rosenbrock_np = z_rosenbrock.numpy()
    z_rastrigin_np = z_rastrigin.numpy()
    z_ackley_np = z_ackley.numpy()

    # Plotting the functions
    fig = plt.figure(figsize=(12, 5))

    # Rosenbrock Plot
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(x.numpy(), y.numpy(), z_rosenbrock_np, cmap='viridis')
    ax1.set_title('Rosenbrock Function')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    # Rastrigin Plot
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_surface(x.numpy(), y.numpy(), z_rastrigin_np, cmap='viridis')
    ax2.set_title('Rastrigin Function')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    # Ackley Plot
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot_surface(x.numpy(), y.numpy(), z_ackley_np, cmap='viridis')
    ax3.set_title('Ackley Function')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')

    plt.show()