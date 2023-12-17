import gymnasium as gym
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder
from neural_net import torch, NeuralNetwork
from itertools import count

env =gym.make("CartPole-v1", render_mode='rgb_array')
recorder = VideoRecorder(env, 'video.mp4')
model_state_dict = torch.load('model.torch_state_dict')

model = NeuralNetwork(env.observation_space.shape[0], env.action_space.n)
model.load_state_dict(model_state_dict)

observation, info = env.reset(seed=42)
for i in count():
    with torch.no_grad():
        X = torch.from_numpy(observation)
        logits = model(X)
        action = logits.argmax().item()

    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
        break

recorder.close()

