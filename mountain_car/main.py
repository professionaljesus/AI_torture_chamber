import torch
import os
import signal
import argparse
import gymnasium as gym
from gymnasium.wrappers.human_rendering import HumanRendering
from neural_net import ActorCritic


interrupted = False
def signal_handler(signal, frame):
    global interrupted
    interrupted = True

signal.signal(signal.SIGINT, signal_handler)

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--display", action="store_true")
parser.add_argument("-s", "--save", action="store_true")
parser.add_argument("-c", "--cont", action="store_true")
try:
    args = parser.parse_args()
except:
    args = argparse.Namespace(display = None, save=None, cont=None)

work_dir = os.path.dirname(__file__)

env = gym.make('MountainCar-v0', render_mode="rgb_array")
if args.display:
    env = HumanRendering(env)

n_inputs = env.observation_space.shape[0]
n_outputs = env.action_space.n

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

model = ActorCritic(n_inputs, n_outputs, 256).to(device)

if args.cont:
    model_state_dict = torch.load(os.path.join(work_dir,'model.torch_state_dict'))
    model.load_state_dict(model_state_dict)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, amsgrad=True)

obs, info = env.reset()

n_steps = 10000
gamma = 0.99
n_trajectory_steps = 7
for i in range(n_steps):
    if interrupted:
        break
    log_probs = []
    values = []
    rewards = []
    mask = []
    entropy = 0

    for _ in range(n_trajectory_steps):
        state = torch.FloatTensor(obs).unsqueeze(0).to(device)
        dist, value = model(state)
        action = dist.sample()

        obs, reward, terminated, truncated, info = env.step(action.item())

        if terminated or truncated:
            obs, info = env.reset()

        log_prob = dist.log_prob(action)
        entropy += dist.entropy().mean()

        log_probs.append(log_prob)
        values.append(value)
        rewards.append(reward)
        mask.append(not terminated)

    next_state = torch.FloatTensor(obs).unsqueeze(0).to(device)
    _, next_value = model(next_state) 

    Q = []
    q = next_value.item()
    for j in reversed(range(len(rewards))):
        q = rewards[j] + (gamma  * q if mask[j] else 0)
        Q.insert(0, q)

    values = torch.vstack(values)
    advantage = torch.FloatTensor(Q).unsqueeze(1).detach() - values
    log_probs = torch.vstack(log_probs)
    actor_loss = -(log_probs * advantage.detach()).mean()
    critic_loss = advantage.pow(2).mean()

    loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print('{:.2f}%,\t max: {:.2f},\t loss: {}'.format(100*i/n_steps, max(rewards),  loss))

if args.save:
    torch.save(model.state_dict(), os.path.join(work_dir, 'model.torch_state_dict'))
    print("----------------- Model Saved -----------------")
