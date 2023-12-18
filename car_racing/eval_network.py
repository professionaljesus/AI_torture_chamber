import gymnasium as gym
import numpy as np
import os
import cv2
from itertools import count
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder
from neural_net import torch, NeuralNetwork
from itertools import count

env =gym.make("CarRacing-v2", render_mode='human')
#recorder = VideoRecorder(env, 'video.mp4')
model_state_dict = torch.load(os.path.join(os.path.dirname(__file__),'model.torch_state_dict'))

turning_bins = np.arange(-1.0, 1.25, 0.25)
n_outputs = len(turning_bins)
model = NeuralNetwork(0, n_outputs)
model.load_state_dict(model_state_dict)

obs, info = env.reset()

lower = np.array([0, 0, 100], dtype="uint8")
upper = np.array([0, 0, 110], dtype="uint8")
image = cv2.cvtColor(obs[:83,:,:], cv2.COLOR_BGR2HSV)
old_state = torch.from_numpy(cv2.inRange(image, lower, upper)).unsqueeze(0).to(torch.float32)
for i in count():
    action = [0,0.1,0]
    turn_id = 0
    with torch.no_grad():
        X = old_state.unsqueeze(1)
        logits = model(X)
        turn_id = logits.argmax().item()

    print(turn_id, logits)
    action[0] = turning_bins[turn_id]
    obs, reward, terminated, truncated, info = env.step(action)

    image = cv2.cvtColor(obs[:83,:,:], cv2.COLOR_BGR2HSV)
    state = torch.from_numpy(cv2.inRange(image, lower, upper)).unsqueeze(0).to(torch.float32)

    old_state = state
    if state.mean() < 0.01:
        terminated = True


    if terminated or truncated:
        obs, info = env.reset()

env.close()

#recorder.close()

