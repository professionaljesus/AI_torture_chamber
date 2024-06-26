import gymnasium as gym
from gymnasium.wrappers.transform_observation import TransformObservation
import argparse
import numpy as np
import matplotlib.pyplot as plt
from itertools import count
import cv2
import os
import random
import signal
from neural_net import NeuralNetwork, torch, nn
from collections import namedtuple, deque
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epsilon", type=float, default=0.9)
parser.add_argument("-s", "--save", action="store_true")
parser.add_argument("-c", "--cont", action="store_true")
args = parser.parse_args()

CONTINUE = args.cont
SAVE = args.save
work_dir = os.path.dirname(__file__)

print("eps: {}, cont {}, save {}".format(args.epsilon, args.cont, args.save))

interrupted = False
def signal_handler(signal, frame):
    global interrupted
    interrupted = True

signal.signal(signal.SIGINT, signal_handler)

Memory = namedtuple('Memory', ('state', 'action', 'next_state', 'reward', 'prev_state'))
memories = deque(maxlen=10000)

turning_bins = [-1, -0.5, 0, 0.5, 1]
n_outputs = len(turning_bins)
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")
image_shape = (83, 96)

model = NeuralNetwork(image_shape, n_outputs).to(device)

if CONTINUE:
    model_state_dict = torch.load(os.path.join(os.path.dirname(__file__),'model.torch_state_dict'))
    model.load_state_dict(model_state_dict)

target_net = NeuralNetwork(image_shape, n_outputs).to(device)

target_net.load_state_dict(model.state_dict())

gamma = 0.99
TAU = 0.005
batch_size = 128
learning_rate = 0.001
loss_fn = nn.SmoothL1Loss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, amsgrad=True)
def optimize_model():
    if len(memories) < batch_size:
        return None
    batch = random.sample(memories, batch_size)
    with torch.no_grad():
        X = torch.stack([torch.vstack([b.state, b.next_state]) for b in batch])
        logits = target_net(X)
        max_rewards = logits.max(1).values

    actions = [b.action for b in batch]
    X = torch.stack([torch.vstack([b.prev_state, b.state]) for b in batch]).to(device)
    pred = model(X).gather(1, torch.as_tensor([actions]).to(device).T)
    y = torch.Tensor([b.reward for b in batch]).to(device) + gamma * max_rewards
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


env = gym.make("CarRacing-v2")#, render_mode="human")

env = TransformObservation(env, lambda obs: cv2.cvtColor(obs[:83,:,:], cv2.COLOR_BGR2HSV))
obs, info = env.reset(seed=42)
steps_to_start = 50

lower = np.array([0, 0, 100], dtype="uint8")
upper = np.array([0, 0, 110], dtype="uint8")

norm_image = cv2.normalize(cv2.inRange(obs, lower, upper), None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
state = torch.from_numpy(norm_image).unsqueeze(0).to(device, torch.float32)
old_state = state.clone()
old_old_state = state.clone()

fwd_road = old_state[0, 50:66, 40:56]
car_road = old_state[0, 66:76, 40:56]


PLOT = False
if PLOT:
    f, ax = plt.subplots(1,3)
    im = [ax[0].imshow(old_state[0]), ax[1].imshow(fwd_road), ax[2].imshow(norm_image)]

EPSILON = args.epsilon

stats = deque([0]*10,maxlen=50)
avg_survival_time = 0
survived = 0
n_steps = 20000
for i in range(n_steps):
    survived += 1
    if interrupted:
        break

    if survived < steps_to_start:
        env.step([0,0,0])
        continue
    action = [0,0.05,0]

    eps = EPSILON * (1.0 - i/n_steps)
    if survived < avg_survival_time/2:
        eps = 0

    if random.random() < eps:
        turn_id = random.randint(0, len(turning_bins) - 1)
    else:
        with torch.no_grad():
            X = torch.vstack([old_old_state,old_state]).unsqueeze(0)
            logits = model(X)
            turn_id = logits.argmax().item()

    action[0] = turning_bins[turn_id]

    obs, reward, terminated, truncated, info = env.step(action)
    norm_image = cv2.normalize(cv2.inRange(obs, lower, upper), None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    state = torch.from_numpy(norm_image).unsqueeze(0).to(device, torch.float32)
    fwd_road = state[0, 50:66, 40:56]
    car_road = state[0, 66:76, 40:56]

    road_reward = fwd_road.mean() - abs(turn_id - n_outputs//2) * 0.05
    #road_reward = (fwd_road.mean() - (181 - car_road.mean()) * (255.0 / 181.5)) / 255.0

    if PLOT:
        print(road_reward)
        im[0].set_data(state[0])
        im[1].set_data(fwd_road)
        im[2].set_data(norm_image)
        plt.show(block=False)
        plt.waitforbuttonpress()

    if state[0, 66:76, 40:56].mean() == 0:
        print('----- Off Road -----')
        terminated = True
        road_reward = -1


    if survived >= avg_survival_time/2:
        memories.append(Memory(old_state, turn_id, state, road_reward, old_old_state))
        memories.append(Memory(old_state[0].fliplr().unsqueeze(0), (n_outputs - 1) - turn_id, state[0].fliplr().unsqueeze(0), road_reward, old_old_state[0].fliplr().unsqueeze(0)))
        loss = optimize_model()

    old_old_state = old_state
    old_state = state

    print('{:.2f}%\t survived: {}\t survived_avg: {:.2f}\t road_reward {:.4f}\t reward: {:.4f}\t loss: {}'\
          .format(100*eps,survived, avg_survival_time,road_reward, reward, loss))

    if terminated or truncated:
        obs, info = env.reset()
        stats.append(survived)
        avg_survival_time = np.mean(stats)
        survived = 0

    if i % 1000 == 0:
        if SAVE:
            torch.save(model.state_dict(), os.path.join(work_dir, 'model.torch_state_dict'))
            print("----------------- Model Saved -----------------")


if SAVE:
    torch.save(model.state_dict(), os.path.join(work_dir, 'model.torch_state_dict'))
    print("----------------- Model Saved -----------------")
env.close()

