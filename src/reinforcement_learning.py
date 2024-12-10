import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from functions import rosenbrock, rastrigin, ackley

# Define the MLP model for the agent (Policy Network)
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.model(x)  # Raw output, we will sample actions from it

# Define the environment (the optimization problem)
class OptimizationEnv:
    def __init__(self, func, bounds):
        self.func = func
        self.bounds = bounds
        self.state = torch.FloatTensor(np.random.uniform(low=bounds[0][0], high=bounds[0][1], size=(2,)))
    
    def reset(self):
        self.state = torch.FloatTensor(np.random.uniform(low=self.bounds[0][0], high=self.bounds[0][1], size=(2,)))
        return self.state
    
    def step(self, action):
        # Action is used to move the current state (optimization point)
        self.state += action
        # Clip the action to stay within bounds
        self.state = torch.clamp(self.state, self.bounds[0][0], self.bounds[0][1])
        
        # Get the reward (negative of the function value at the new state)
        reward = -self.func(self.state)
        return self.state, reward

# Define the agent (policy network) with training loop
def train_rl_agent(env, policy, optimizer, epochs=1000, lr=0.01):
    rewards = []
    for epoch in range(epochs):
        state = env.reset()
        state_tensor = state.clone()
        
        optimizer.zero_grad()
        
        # Forward pass through the policy network to get action (raw output)
        action_logits = policy(state_tensor)  # [Batch, Action space]
        
        # We will assume the action space is continuous
        # Sample an action from the output (you could use a distribution like Normal if necessary)
        action = torch.tanh(action_logits)  # Clip action between -1 and 1

        # Step the environment and get the next state and reward
        next_state, reward = env.step(action.detach())
        
        # Convert reward to tensor to ensure it can be used in backpropagation
        reward_tensor = torch.tensor(reward, dtype=torch.float32)
        
        # Compute loss as the negative reward (maximize reward)
        # Calculate log probability of the action
        dist = torch.distributions.Normal(action_logits, 1.0)  # Assuming a normal distribution
        log_prob = dist.log_prob(action).sum()
        
        # Policy gradient loss: loss = -log_prob * reward
        loss = -log_prob * reward_tensor
        
        # Backpropagate and update policy
        loss.backward()
        optimizer.step()
        
        rewards.append(reward)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}, State: {state}, Reward: {reward:.4f}")
    
    return policy

if __name__ == "__main__":
    input_dim = 2
    hidden_dim = 128
    output_dim = 2  # Output dimension is same as input for action space
    
    model_lst = sorted(os.listdir('models/'))
    model_parent_dir = f"models/{model_lst[-1]}"
    
    functions = {
        "Rosenbrock": (lambda x: rosenbrock(x.clone()), torch.tensor([1.0, 1.0])),  # Minimum is at (1, 1)
        "Rastrigin": (lambda x: rastrigin(x.clone()), torch.tensor([0.0, 0.0])),  # Minimum is at (0, 0)
        "Ackley": (lambda x: ackley(x.clone()), torch.tensor([0.0, 0.0]))  # Minimum is at (0, 0)
    }
    
    res = {}
    
    for name, (func, true_min) in functions.items():
        print(f"\nTesting {name} using Reinforcement Learning")
        
        model_path = f"{model_parent_dir}/{name.lower()}_model.pt"
        policy = PolicyNetwork(input_dim, hidden_dim, output_dim)
        optimizer = optim.Adam(policy.parameters(), lr=0.01)
        
        # Create environment
        env = OptimizationEnv(func, [(-5.0, 5.0), (-5.0, 5.0)])  # Bounds for optimization
        
        # Train RL agent
        trained_policy = train_rl_agent(env, policy, optimizer, epochs=10000)
        
        # Evaluate the trained policy
        final_state = trained_policy(torch.tensor(true_min, dtype=torch.float32))
        
        print(f"Final Optimized X: {final_state}")
        print(f"Actual Minimum for {name}: {true_min}")
        
        res[name] = {
            "Optimized": (final_state.tolist(), round(func(final_state), 5)),
            "Actual": (true_min.tolist(), round(func(true_min), 5))
        }
    
    print(res)

    # Save results
    res_path = f"results/reinforcement_learning/{model_lst[-1]}.json"
    print(f"Saving Results to {res_path}")
    with open(res_path, "w+") as f:
        f.write(json.dumps(res, indent=4))
