import torch
import numpy as np
import matplotlib.pyplot as plt

# Define the inverse cone equation
def inverse_cone(x, y, h=5, r_max=5):
    r = np.sqrt(x**2 + y**2)
    z = h * r / r_max
    return z

if __name__ == "__main__":
    # Generate a mesh grid using PyTorch
    x = torch.linspace(-5, 5, 100)
    y = torch.linspace(-5, 5, 100)
    x, y = torch.meshgrid(x, y, indexing="ij")

    # Calculate the inverse cone values for the grid
    z_cone = inverse_cone(x.numpy(), y.numpy(), h=5, r_max=5)

    # Plotting the inverse cone
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface of the inverse cone
    ax.plot_surface(x.numpy(), y.numpy(), z_cone, cmap='viridis')

    # Set the labels and title
    ax.set_title('Inverse 3D Cone')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show the plot
    plt.show()