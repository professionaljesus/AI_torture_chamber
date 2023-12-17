import random
import gymnasium as gym
from collections import namedtuple, deque
from neural_net import NeuralNetwork, torch, nn

Memory = namedtuple('Memory', ('state', 'action', 'next_state', 'reward'))
memories = deque(maxlen=10000)

env = gym.make("CartPole-v1", render_mode="human")

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

n_input = env.observation_space.shape[0]
model = NeuralNetwork(n_input, env.action_space.n).to(device)
target_net = NeuralNetwork(n_input, env.action_space.n).to(device)
target_net.load_state_dict(model.state_dict())


old_observation, info = env.reset(seed=42)

gamma = 0.99
TAU = 0.005

n_steps = 20000
batch_size = 128
learning_rate = 0.001
loss_fn = nn.SmoothL1Loss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, amsgrad=True)
def optimize_model(batch):
    with torch.no_grad():
        X = torch.vstack([torch.from_numpy(b.next_state) for b in batch])
        logits = target_net(X)
        max_rewards = logits.max(1).values

    actions = [b.action for b in batch]
    X = torch.vstack([torch.from_numpy(b.state) for b in batch])
    pred = model(X).gather(1, torch.as_tensor([actions]).T)
    y = torch.Tensor([b.reward for b in batch]) + gamma * max_rewards
    loss = loss_fn(pred, y.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_value_(model.parameters(), 100)
    optimizer.step()

    with torch.no_grad():
        target_net_state_dict = target_net.state_dict()
        model_state_dict = model.state_dict()
        for key in model_state_dict:
            target_net_state_dict[key] = model_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

    return loss


stats = deque(maxlen=20)
survived_rounds = 0

for i in range(n_steps):
    epsilon = (1.0 - (i / n_steps))*0.5

    if random.random() < epsilon:
        action = env.action_space.sample()
    else:
        with torch.no_grad():
            X = torch.from_numpy(old_observation)
            logits = model(X)
            action = logits.argmax().item()

    observation, reward, terminated, truncated, info = env.step(action)
    reward = (2*(2.4**2 - observation[0]**2)/(2.4 ** 2) - 1)*0.5 - 0.5*abs(observation[2])/0.2095
    if terminated:
        reward = -1

    new_memory = Memory(old_observation, action, observation, reward)
    memories.append(new_memory)
    old_observation = observation

    loss = None
    if len(memories) > batch_size:
        batch = random.sample(memories, batch_size)
        loss = optimize_model(batch)

    survived_rounds += 1
    if terminated or truncated:
        stats.append(survived_rounds)
        survived_rounds = 0

        print('{:.2f}%,\t max: {},\t min {},\t avg: {:.2f},\t loss: {}'.format(100*i/n_steps, max(stats), min(stats), sum(stats) / len(stats), loss))
        old_observation, info = env.reset()

torch.save(model.state_dict(), 'model.torch_state_dict')
env.close()
