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
def unpreprocess(img):
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1,3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(1,3,1,1)
    img = ((img * std + mean) * 255).to(dtype=torch.uint8).squeeze(0)
    img = pil_transform(img)
    return img

dir = os.path.dirname(os.path.abspath(__file__))

# Initialize the Weight Transforms
weights = torchvision.models.VGG11_Weights.DEFAULT
preprocess = weights.transforms()

# Apply it to the input image
model = torchvision.models.vgg11(weights=weights)
model.eval()

#make_dot(y, params=dict(model.named_parameters())).render(os.path.join(dir, "rnn_torchviz"), format="png")

goldfish = Image.open(os.path.join(dir, 'goldfish.jpeg'))
goldfish = preprocess(goldfish).unsqueeze(0)
unpreprocess(goldfish).save(os.path.join(dir, 'gold_goldfish.png'))
y = model(goldfish)
smax = F.softmax(y, dim=1)

loss_func = nn.CrossEntropyLoss()
y_fake = torch.zeros((1,1000))
y_fake[0,7] = 1

while smax[0,7] < 0.95:
    goldfish.requires_grad = True
    goldfish.grad = None
    y = model(goldfish)

    smax = F.softmax(y, dim=1)
    top = torch.topk(smax, 5)
    for i, p in zip(top.indices[0], top.values[0]):
        print(class_names[int(i.item())], p.item())

    loss = loss_func(y, y_fake)
    loss.backward()

    goldfish.requires_grad = False
    goldfish -= goldfish.grad
    #gf.save(os.path.join(dir, 'gf.jpg'))

gf = unpreprocess(goldfish)
gf.save(os.path.join(dir, 'gold_cockfish.png'))


