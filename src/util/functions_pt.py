from math import pi
import torch
from torch import Tensor

# Define the Rosenbrock function
def rosenbrock(x: Tensor):
    a = 1
    b = 100
    return torch.sum(b * (x[1:] - x[:-1]**2)**2 + (a - x[:-1])**2)

# Define the Rastrigin function
def rastrigin(x: Tensor):
    A = 10
    return A * len(x) + torch.sum(x**2 - A * torch.cos(2 * pi * x))

# Define the Ackley function
def ackley(x: Tensor):
    a = 20
    b = 0.2
    c = 2 * pi
    n = len(x)
    sum1 = torch.sum(x**2)
    sum2 = torch.sum(torch.cos(c * x))
    term1 = -a * torch.exp(-b * torch.sqrt(sum1 / n))
    term2 = -torch.exp(sum2 / n)
    return term1 + term2 + a + torch.exp(torch.tensor(1.0))
