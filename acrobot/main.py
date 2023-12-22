import torch
import os
import signal
import argparse
import numpy as np
import gymnasium as gym
from gymnasium.wrappers.human_rendering import HumanRendering
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder
from gymnasium.wrappers.transform_observation import TransformObservation
from neural_net import ActorCritic, nn
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

env = gym.make('Pendulum-v1', render_mode="rgb_array")
env = TransformObservation(env, lambda obs: [obs[0], obs[1], obs[2] / 8.0] )

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
n_trajectory_steps = 48

stats = deque([0],maxlen=100)

for i in range(n_steps):
    if interrupted:
        break
    log_probs = []
    values = []
    rewards = []
    mask = []
    entropy = 0

    times_at_top = 0
    home_truncated = False
    for j in range(n_trajectory_steps):
        state = torch.FloatTensor(obs).unsqueeze(0).to(device)
        dist, value = model(state)
        if i < 5:
            action = torch.FloatTensor([-obs[2]*2])
        else:
            action = dist.sample()
            if args.record:
                recorder.capture_frame()

        obs, reward, terminated, truncated, info = env.step(action.numpy().reshape((1,)))

        reward /= 16.2736044

        if obs[0] > 0.97:
            times_at_top += 1

        if times_at_top >= n_trajectory_steps:
            print('-------------kill--------------') 
            #obs, info = env.reset()
            home_truncated = True

        if home_truncated:
            if args.record and i > 5:
                interrupted = True
                break

        if args.display:
            fmt = [dist.loc.item(), dist.scale.item(), action[0].item(), *obs, reward]
            print(("[{:.2f},{:.2f}] -> {:+.2f}\t x: [" + "{:+.2f}," * len(obs) + "] \t r: {:.2f}").format(*fmt))

        log_prob = dist.log_prob(action)
        entropy += dist.entropy().mean()

        log_probs.append(log_prob)
        values.append(value)
        rewards.append(reward)
        mask.append(not home_truncated)

    stats.append(times_at_top)
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

    if loss > 1e15:
        breakpoint()

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
