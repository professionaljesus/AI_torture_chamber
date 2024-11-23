import torch
import numpy as np
import os
from neural_net import AutoEncoder

work_dir = os.path.dirname(__file__)
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

model = AutoEncoder().to(device=device)

X = torch.from_numpy(np.load(os.path.join(work_dir, 'obs.npy'))).to(device=device)
print(X.shape)

y = model(X)
print(y.shape)


