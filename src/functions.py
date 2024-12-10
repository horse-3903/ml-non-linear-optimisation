from math import pi

import torch
from torch import Tensor

# Define the Rosenbrock function
def rosenbrock(x: Tensor):
    a = 1
    b = 100
    return sum([(b * (xi_next - xi**2)**2 + (a - xi)**2).item() for xi, xi_next in zip(x[:-1], x[1:])])


# Define the Rastrigin function
def rastrigin(x: Tensor):
    A = 10
    return A * len(x) + sum([(xi**2 - A * torch.cos(2 * pi * xi)).item() for xi in x])

# Define the Ackley function
def ackley(x: Tensor):
    a = 20
    b = 0.2
    c = 2 * pi
    n = len(x)
    sum1 = sum(xi**2 for xi in x)
    sum2 = sum(torch.cos(c * xi) for xi in x)
    term1 = -a * torch.exp(-b * torch.sqrt(sum1 / n))
    term2 = -torch.exp(sum2 / n)
    return term1 + term2 + a + torch.exp(torch.tensor(1.0))