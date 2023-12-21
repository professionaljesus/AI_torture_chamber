import torch
import numpy as np
import gymnasium as gym
from gymnasium.wrappers.human_rendering import HumanRendering
from neural_net import ActorCritic
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

env = gym.make('MountainCar-v0', render_mode="rgb_array")
env = HumanRendering(env)

n_inputs = env.observation_space.shape[0]
n_outputs = env.action_space.n

model = ActorCritic(n_inputs, n_outputs, 256).to(device)


obs, info = env.reset()


n_steps = 1000
gamma = 0.99
n_trajectory_steps = 5
for i in range(n_steps):
    log_probs = []
    values = []
    rewards = []
    for _ in range(n_trajectory_steps):
        state = torch.FloatTensor(obs).unsqueeze(0).to(device)
        dist, value = model(state)
        action = dist.sample()

        obs, reward, terminated, truncated, info = env.step(action.item())

        if terminated or truncated:
            break

        log_prob = dist.log_prob(action)

        log_probs.append(log_prob)
        values.append(value)
        rewards.append(reward)

    next_state = torch.FloatTensor(obs).unsqueeze(0).to(device)
    _, next_value = model(next_state)


    Q = []
    q = next_value.item()
    for r in reversed(rewards):
        q = r + gamma  * q
        Q.insert(0, q)

    values = torch.vstack(values)
    rewards = torch.vstack(rewards).to(device)
    advantage = np.array(Q) - values
    log_probs = torch.vstack(log_probs)

