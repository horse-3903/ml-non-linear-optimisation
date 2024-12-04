import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import rosen
from math import pi

# Define the Rastrigin function
def rastrigin(x):
    A = 10
    return A * len(x) + sum([(xi**2 - A * torch.cos(2 * pi * xi)).item() for xi in x])

# Define the Ackley function (for optimization testing)
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

# Generate synthetic data for training
def generate_data(func, n_samples=1000, input_dim=2, range_min=-5, range_max=5):
    X = np.random.uniform(range_min, range_max, size=(n_samples, input_dim))
    y = np.array([func(x) for x in X])
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).unsqueeze(1)

# Define the MLP (Multilayer Perceptron) model
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

# Training function
def train_mlp(model, optimizer, loss_fn, X_train, y_train, epochs=1000):
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        predictions = model(X_train)
        loss = loss_fn(predictions, y_train)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# Function to evaluate optimization
def evaluate_optimization(model, func, input_dim, range_min, range_max, iterations=100):
    # Initialize current_point with dtype=float32 to match the model's weights
    current_point = torch.tensor(np.random.uniform(range_min, range_max, size=(1, input_dim)), dtype=torch.float32, requires_grad=True)
    optimizer = optim.SGD([current_point], lr=0.01)

    for _ in range(iterations):
        optimizer.zero_grad()
        prediction = model(current_point)  # Forward pass
        prediction.backward()
        optimizer.step()
    
    return current_point.detach().numpy(), func(current_point.detach().numpy()[0])

# Main script
if __name__ == "__main__":
    input_dim = 2
    hidden_dim = 64
    output_dim = 1
    n_samples = 1000
    range_min, range_max = -5, 5
    
    functions = {
        "Rosenbrock": lambda x: rosen(x),
        "Rastrigin": lambda x: rastrigin(torch.tensor(x)),
        "Ackley": lambda x: ackley(torch.tensor(x))
    }
    
    for name, func in functions.items():
        print(f"\nTraining MLP for {name} function")
        
        # Generate training data
        X_train, y_train = generate_data(func, n_samples, input_dim, range_min, range_max)
        
        # Create model, optimizer, and loss function
        model = MLP(input_dim, hidden_dim, output_dim)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()
        
        # Train the model
        train_mlp(model, optimizer, loss_fn, X_train, y_train, epochs=1000)
        
        # Evaluate optimization
        print(f"Evaluating optimization for {name} function")
        result_point, result_value = evaluate_optimization(model, func, input_dim, range_min, range_max)
        print(f"Optimized Point: {result_point}, Function Value: {result_value:.4f}")