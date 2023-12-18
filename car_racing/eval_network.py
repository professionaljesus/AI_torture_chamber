import signal
import gymnasium as gym
import numpy as np
import os
import cv2
from itertools import count
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder
from neural_net import torch, NeuralNetwork
from itertools import count

interrupted = False
def signal_handler(signal, frame):
    global interrupted
    interrupted = True

signal.signal(signal.SIGINT, signal_handler)

env = gym.make("CarRacing-v2", render_mode='rgb_array')
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")
model_state_dict = torch.load(os.path.join(os.path.dirname(__file__),'model.torch_state_dict'), map_location=torch.device(device))

turning_bins = [-1, -0.5, 0, 0.5, 1]
n_outputs = len(turning_bins)
model = NeuralNetwork(0, n_outputs)
model.load_state_dict(model_state_dict)

recorder = VideoRecorder(env, 'video.mp4')
obs, info = env.reset()

lower = np.array([0, 0, 100], dtype="uint8")
upper = np.array([0, 0, 110], dtype="uint8")
image = cv2.cvtColor(obs[:83,:,:], cv2.COLOR_BGR2HSV)
old_state = torch.from_numpy(cv2.inRange(image, lower, upper)).unsqueeze(0).to(torch.float32)
old_old_state = torch.from_numpy(cv2.inRange(image, lower, upper)).unsqueeze(0).to(torch.float32)
survived = 0
for i in count():
    recorder.capture_frame()
    survived += 1
    if survived < 30:
        env.step([0,0,0])
        continue
    acc = 0
    if i % 3 == 0:
        acc = 0.1
    action = [0, acc ,0]
    turn_id = 0
    with torch.no_grad():
        X = torch.vstack([old_old_state, old_state]).unsqueeze(0)
        logits = model(X)
        turn_id = logits.argmax().item()

    print(survived ,turn_id, logits)
    action[0] = turning_bins[turn_id]
    obs, reward, terminated, truncated, info = env.step(action)

    image = cv2.cvtColor(obs[:83,:,:], cv2.COLOR_BGR2HSV)
    state = torch.from_numpy(cv2.inRange(image, lower, upper)).unsqueeze(0).to(torch.float32)

    old_old_state = old_state
    old_state = state
    if state.mean() < 0.01:
        terminated = True


    if terminated or truncated:
        obs, info = env.reset()
        survived = 0

    if truncated:
        break

env.close()

recorder.close()

