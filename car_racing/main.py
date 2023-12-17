import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from itertools import count
import cv2


env = gym.make("CarRacing-v2")

observation, info = env.reset(seed=42)
print(env.observation_space.shape)

f, ax = plt.subplots(1,2)
lower = np.array([0, 0, 100], dtype="uint8")
upper = np.array([0, 0, 110], dtype="uint8")
image = cv2.cvtColor(observation, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(image, lower, upper)
ims = [ax[0].imshow(observation), ax[1].imshow(mask)]
for i in count():
    action = env.action_space.sample()

    observation, reward, terminated, truncated, info = env.step(action)

    image = cv2.cvtColor(observation[:83,:,:], cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(image, lower, upper)

    ims[0].set_data(image)
    ims[1].set_data(mask)
    plt.show(block=False)
    plt.pause(0.1)


    if terminated or truncated:
        observation, info = env.reset()
        break

plt.close()

