import torch
import os
import signal
import argparse
import cv2
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from gymnasium.wrappers.human_rendering import HumanRendering
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder
from gymnasium.wrappers.transform_observation import TransformObservation
from neural_net import ActorCriticCNN, nn
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
    args = argparse.Namespace(display = None, save=None, cont=None, record=None)

work_dir = os.path.dirname(__file__)

def trans_obs(obs):
    obs = cv2.cvtColor(obs[0:82, 7:89, :], cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 100], dtype="uint8")
    upper = np.array([0, 0, 110], dtype="uint8")
    obs = cv2.resize(obs, (0, 0), fx = 0.5, fy = 0.5)
    obs = cv2.normalize(cv2.inRange(obs, lower, upper), None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return obs[np.newaxis, ...]

env = gym.make('CarRacing-v2', render_mode="rgb_array")
env = TransformObservation(env, trans_obs)


if args.display:
    env = HumanRendering(env)
elif args.record:
    recorder = VideoRecorder(env, os.path.join(work_dir, 'video.mp4'))

n_outputs = env.action_space.shape[0] - 1

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

model = ActorCriticCNN(n_outputs).to(device)

if args.cont:
    path = os.path.join(work_dir,'model.torch_state_dict')
    if os.path.exists(path):
        model_state_dict = torch.load(path)
        model.load_state_dict(model_state_dict)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, amsgrad=True)

obs, info = env.reset()
n_steps = 10000
gamma = 0.99
n_trajectory_steps = 48

steps_to_start = 50
survived = 0
stats = deque([0],maxlen=100)


prev_state = torch.FloatTensor(obs).to(device)
for i in range(n_steps):
    if interrupted:
        break
    log_probs = []
    values = []
    rewards = []
    mask = []
    entropy = 0

    for j in range(n_trajectory_steps):
        if interrupted:
            break

        if args.record:
            recorder.capture_frame()

        state = torch.FloatTensor(obs).to(device)
        dist, value = model(torch.stack([prev_state,state], 1))
        prev_state = state
        action = dist.sample()

        env_action = [action[0][0].item(), max(action[0][1].item(), 0), -min(action[0][1].item(), 0)]

        if survived < steps_to_start:
            env_action = [0,0,0]

        obs, reward, terminated, truncated, info = env.step(env_action)
        survived += 1

        if survived < steps_to_start:
            continue

        if state[0, 30:, 16:25].mean() == 0:
            terminated = True

        if terminated or truncated:
            obs, info = env.reset()
            stats.append(survived)
            survived = 0
            if terminated and args.record:
                interrupted = True
                break


        if args.display:
            fmt = [*dist.loc[0].tolist(), *env_action, reward]
            print(("[" + "{:+.2f}," * len(dist.loc[0]) + "] -> [" + "{:+.2f}," * len(env_action) + "]\t r: {:+.2f}").format(*fmt))

        log_prob = dist.log_prob(action)
        entropy += dist.entropy().mean()

        log_probs.append(log_prob)
        values.append(value)
        rewards.append(reward)
        mask.append(not terminated)

    if args.record:
        continue

    next_state = torch.FloatTensor(obs).to(device)
    _, next_value = model(torch.stack([state, next_state], 1))

    Q = []
    q = next_value.item()
    for j in reversed(range(len(rewards))):
        q = rewards[j] + gamma  * q if mask[j] else 0
        Q.insert(0, q)

    if not len(values):
        continue

    values = torch.vstack(values)
    advantage = torch.FloatTensor(Q).unsqueeze(1).detach().to(device) - values
    log_probs = torch.vstack(log_probs)
    print(log_probs * advantage.detach())
    actor_loss = -(log_probs * advantage.detach()).mean()
    critic_loss = advantage.pow(2).mean()

    loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy

    optimizer.zero_grad()
    loss.backward()

    nn.utils.clip_grad_value_(model.parameters(), 100)
    optimizer.step()

    fmt = [100*i/n_steps, max(rewards),  max(stats), sum(stats)/len(stats), loss]
    print('{:.2f}%,\t max: {:.2f} \t max_ttt {} \t avg_ttt {:.2f},\t loss: {}'.format(*fmt))

if args.save:
    torch.save(model.state_dict(), os.path.join(work_dir, 'model.torch_state_dict'))
    print("----------------- Model Saved -----------------")

if args.record:
    recorder.close()
