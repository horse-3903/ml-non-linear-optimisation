import numpy as np
import matplotlib.pyplot as plt

# Function Definitions
def rosenbrock_function(x, y):
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

def rastrigin_function(x, y):
    return 10 * 2 + (x ** 2 - 10 * np.cos(2 * np.pi * x)) + (y ** 2 - 10 * np.cos(2 * np.pi * y))

def ackley_function(x, y):
    return -20 * np.exp(-0.2 * np.sqrt(0.5 * (x ** 2 + y ** 2))) - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) + np.e + 20

# Generate a grid of points
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)

# Compute the function values
Z_rosenbrock = rosenbrock_function(X, Y)
Z_rastrigin = rastrigin_function(X, Y)
Z_ackley = ackley_function(X, Y)

# Plot the Rosenbrock Function
fig = plt.figure(figsize=(18, 5))

ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(X, Y, Z_rosenbrock, cmap='viridis')
ax1.set_title('Rosenbrock Function')
ax1.set_xlabel('X-axis')
ax1.set_ylabel('Y-axis')
ax1.set_zlabel('Z-axis')

# Plot the Rastrigin Function
ax2 = fig.add_subplot(132, projection='3d')
ax2.plot_surface(X, Y, Z_rastrigin, cmap='viridis')
ax2.set_title('Rastrigin Function')
ax2.set_xlabel('X-axis')
ax2.set_ylabel('Y-axis')
ax2.set_zlabel('Z-axis')

# Plot the Ackley Function
ax3 = fig.add_subplot(133, projection='3d')
ax3.plot_surface(X, Y, Z_ackley, cmap='viridis')
ax3.set_title('Ackley Function')
ax3.set_xlabel('X-axis')
ax3.set_ylabel('Y-axis')
ax3.set_zlabel('Z-axis')

plt.tight_layout()
plt.show()