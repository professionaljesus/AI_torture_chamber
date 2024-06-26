from numpy import zeros, zeros_like
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torchvision
import cv2
import matplotlib.pyplot as plt
from torchviz import make_dot

from imagenet_classes import class_names
import os
from torchsummary import summary

from PIL import Image 

pil_transform = torchvision.transforms.ToPILImage()

IMAGENET_MEAN_1 = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD_1 = torch.tensor([0.229, 0.224, 0.225])

LOWER_IMAGE_BOUND = torch.tensor((-IMAGENET_MEAN_1 / IMAGENET_STD_1).reshape(1, -1, 1, 1))
UPPER_IMAGE_BOUND = torch.tensor(((1 - IMAGENET_MEAN_1) / IMAGENET_STD_1).reshape(1, -1, 1, 1))


def unnormalize(img):
    mean = IMAGENET_MEAN_1.reshape(1,3,1,1)
    std = IMAGENET_STD_1.reshape(1,3,1,1)
    img = img * std + mean
    return img

def unpreprocess(img):
    img = unnormalize(img)
    img = (img * 255).to(dtype=torch.uint8)
    img = img.squeeze(0)
    img = pil_transform(img)
    return img

dir = os.path.dirname(os.path.abspath(__file__))

# Initialize the Weight Transforms
weights = torchvision.models.VGG11_Weights.DEFAULT
preprocess = weights.transforms()

# Apply it to the input image
model = torchvision.models.vgg11(weights=weights)
model.eval()
summary(model, (3, 224, 224))

goldfish = Image.open(os.path.join(dir, 'goldfish.jpeg'))
goldfish = preprocess(goldfish).unsqueeze(0)

goldfish.requires_grad = True
optimizer = torch.optim.Adam([goldfish], lr=0.01)

loss_func = nn.HuberLoss()

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output
    return hook

model.classifier[1].register_forward_hook(get_activation('relu'))

for _ in range(300):
    goldfish.grad = None
    model.zero_grad()
    y = model(goldfish)

    loss = loss_func(activation['relu'], torch.zeros_like(activation['relu']))
    loss.backward()

    goldfish.data = goldfish.data + 50*goldfish.grad
    goldfish.data = torch.max(torch.min(goldfish, UPPER_IMAGE_BOUND), LOWER_IMAGE_BOUND)

gf = unpreprocess(goldfish)
gf.save(os.path.join(dir, 'dreamfish.png'))


