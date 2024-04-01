import torch
import os
import signal
import argparse
import cv2
import numpy as np
import gymnasium as gym
import random
import matplotlib.pyplot as plt
from gymnasium.wrappers.human_rendering import HumanRendering
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder
from gymnasium.wrappers.transform_observation import TransformObservation
from neural_net import ActorCritic, nn, F
from collections import deque, namedtuple

interrupted = False
def signal_handler(signal, frame):
    global interrupted
    interrupted = True

def receive(signum, stack):
    print("Received",signum)

signal.signal(signal.SIGUSR1, receive)

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

env = gym.make('Pendulum-v1', render_mode="rgb_array")

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

model = ActorCritic(n_inputs, n_outputs, 64).to(device)

if args.cont:
    path = os.path.join(work_dir,'model.torch_state_dict')
    if os.path.exists(path):
        model_state_dict = torch.load(path, map_location=torch.device(device))
        model.load_state_dict(model_state_dict)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, amsgrad=True)

clip_e = 0.1
def ppo_optim(states, actions, returns, old_log_probs, old_advantages):

    tot_loss = 0
    idx = np.random.choice(range(len(old_log_probs)), min(25, len(old_log_probs)), replace=False)

    iterab = [(states[i], actions[i], returns[i], old_log_probs[i], old_advantages[i]) for i in idx]

    for state, action, ret, old_log_prob, old_adv in iterab:
        dist, value = model(state)
        entropy = dist.entropy().mean()
        log_prob = dist.log_prob(action)

        ratio = (log_prob - old_log_prob).exp()
        unclipped = ratio * old_adv
        clipped = torch.clamp(ratio, 1.0 - clip_e, 1.0 + clip_e)*old_adv

        actor_loss = -torch.min(unclipped, clipped).mean()
        critic_loss = (ret - value).pow(2).mean()

        loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy

        tot_loss += loss

        optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_value_(model.parameters(), 100)
        optimizer.step()

    return tot_loss

obs, info = env.reset()
n_steps = 10000
gamma = 0.95
n_trajectory_steps = 30

steps_to_start = 50
survived = 0
stats = deque([0],maxlen=100)

for i in range(n_steps):
    if interrupted:
        break

    states = []
    log_probs = []
    values = []
    rewards = []
    mask = []
    actions = []
    entropy = 0

    for j in range(n_trajectory_steps):
        if interrupted:
            break

        if args.record:
            recorder.capture_frame()

        state = torch.FloatTensor(obs).to(device).unsqueeze(0)
        dist, value = model(state)
        states.append(state)
        action = dist.sample()
        actions.append(action)

        env_action = 2*F.tanh(action)[0].numpy()

        obs, reward, terminated, truncated, info = env.step(env_action)

        if terminated or truncated:
            obs, info = env.reset()
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
        mask.append(not terminated and not truncated)

    if args.record:
        continue

    if not len(values):
        continue

    values = torch.vstack(values).detach()
    log_probs = torch.vstack(log_probs).detach()

    state = torch.FloatTensor(obs).to(device).unsqueeze(0)

    _, next_value = model(state)

    Q = []
    q = next_value.item()
    for j in reversed(range(len(rewards))):
        q = rewards[j] + gamma  * q if mask[j] else 0
        Q.insert(0, q)

    Q = torch.FloatTensor(Q).unsqueeze(1).detach().to(device)
    advantage = Q - values

    loss = ppo_optim(states, actions, Q, log_probs, advantage)

    fmt = [100*i/n_steps, max(rewards),  max(stats), sum(stats)/len(stats), loss]
    print('{:.2f}%,\t max: {:.2f} \t max_tr {} \t avg_tr {:.2f} \t loss {:.5f}'.format(*fmt))

if args.save:
    torch.save(model.state_dict(), os.path.join(work_dir, 'model.torch_state_dict'))
    print("----------------- Model Saved -----------------")

if args.record:
    recorder.close()