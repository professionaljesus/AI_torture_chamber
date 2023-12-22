import torch
import os
import signal
import argparse
import numpy as np
import gymnasium as gym
from gymnasium.wrappers.human_rendering import HumanRendering
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder
from neural_net import ActorCritic
from collections import deque


interrupted = False
def signal_handler(signal, frame):
    global interrupted
    interrupted = True

signal.signal(signal.SIGINT, signal_handler)

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--display", action="store_true")
parser.add_argument("-s", "--save", action="store_true")
parser.add_argument("-c", "--cont", action="store_true")
parser.add_argument("-r", "--record", action="store_true")

try:
    args = parser.parse_args()
except:
    args = argparse.Namespace(display = None, save=None, cont=None)

work_dir = os.path.dirname(__file__)

env = gym.make('MountainCarContinuous-v0', render_mode="rgb_array")
if args.display:
    env = HumanRendering(env)
elif args.record:
    recorder = VideoRecorder(env, os.path.join(work_dir, 'video.mp4'))

n_inputs = env.observation_space.shape[0]
n_outputs = env.action_space.shape[0]

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
    path = os.path.join(work_dir,'model.torch_state_dict')
    if os.path.exists(path):
        model_state_dict = torch.load(path)
        model.load_state_dict(model_state_dict)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, amsgrad=True)

obs, info = env.reset()
n_steps = 10000
gamma = 0.99
n_trajectory_steps = 40
all_time_max = 0
times_won = 0

stats = deque(maxlen=100)
last_time = 0

for i in range(n_steps):
    if interrupted:
        break
    log_probs = []
    values = []
    rewards = []
    mask = []
    entropy = 0

    for j in range(n_trajectory_steps):
        time = i*n_trajectory_steps + j
        state = torch.FloatTensor(obs).unsqueeze(0).to(device)
        dist, value = model(state)
        action = dist.sample()

        if args.record:
            recorder.capture_frame()

        obs, reward, terminated, truncated, info = env.step(action.numpy().reshape((1,)))

        #reward += max(obs[0] + 0.5 *  (1.0 - i/n_steps), 0)

        if reward >= 0:
            reward = min(reward, 5)
        else:
            reward = -0.1

        if terminated:
            times_won += 1
            stats.append(time - last_time)

        if terminated or truncated:
            obs, info = env.reset()
            if truncated:
                stats.append(999)
            last_time = time 
            if args.record:
                interrupted = True
                break

        if args.display:
            fmt = [dist.loc.item(), dist.scale.item(), action[0].item(), obs, reward]
            print("[{:.2f},{:.2f}] -> {:.2f}\t x: {} \t r: {:.2f}".format(*fmt))

        log_prob = dist.log_prob(action)
        entropy += dist.entropy().mean()

        log_probs.append(log_prob)
        values.append(value)
        rewards.append(reward)
        mask.append(not terminated and not truncated)

    if args.record:
        continue

    next_state = torch.FloatTensor(obs).unsqueeze(0).to(device)
    _, next_value = model(next_state) 

    Q = []
    q = next_value.item()
    for j in reversed(range(len(rewards))):
        q = rewards[j] + gamma  * q if mask[j] else 0
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
    

    all_time_max = max(max(rewards), all_time_max)
    print('{:.2f}%,\t max: {:.2f}/{:.2f} \t times won {} \t avg_ttw {:.2f},\t loss: {}'.format(100*i/n_steps, max(rewards), all_time_max, times_won, sum(stats)/max(len(stats), 1),loss))

if args.save:
    torch.save(model.state_dict(), os.path.join(work_dir, 'model.torch_state_dict'))
    print("----------------- Model Saved -----------------")

if args.record:
    recorder.close()
