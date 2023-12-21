import signal
import gymnasium as gym
import numpy as np
import os
import cv2
import argparse
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder
from gymnasium.wrappers.transform_observation import TransformObservation
from neural_net import torch, NeuralNetwork

parser = argparse.ArgumentParser('record')
args = parser.parse_args()

interrupted = False
def signal_handler(signal, frame):
    global interrupted
    interrupted = True

signal.signal(signal.SIGINT, signal_handler)

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

RECORD = False
env = gym.make("CarRacing-v2", render_mode='rgb_array' if RECORD else 'human')
if RECORD:
    recorder = VideoRecorder(env, 'video.mp4')

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

survived = 0
n_steps = 20000
for i in range(n_steps):
    survived += 1
    if interrupted:
        break
    if RECORD:
        recorder.capture_frame()
    survived += 1
    if survived < 50:
        env.step([0,0,0])
        continue

    action = [0,0.1,0]
    with torch.no_grad():
        X = torch.vstack([old_old_state,old_state]).unsqueeze(0)
        logits = model(X)
        turn_id = logits.argmax().item()

    action[0] = turning_bins[turn_id]

    obs, reward, terminated, truncated, info = env.step(action)
    norm_image = cv2.normalize(cv2.inRange(obs, lower, upper), None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    state = torch.from_numpy(norm_image).unsqueeze(0).to(device, torch.float32)
    fwd_road = state[0, 50:66, 40:56]

    if state[0, 66:76, 40:56].mean() == 0:
        print('----- Off Road -----')
        terminated = True

    old_old_state = old_state
    old_state = state

    print('survived: {}\t road_reward {:.4f}\t'\
          .format(survived, reward))
    if terminated or truncated:
        obs, info = env.reset()
        survived = 0

if RECORD:
    recorder.close()

env.close()

