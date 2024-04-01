import torch
import torch.nn.functional as F
import torchvision

from imagenet_classes import class_names
import os

from PIL import Image 

dir = os.path.dirname(os.path.abspath(__file__))

# Initialize the Weight Transforms
weights = torchvision.models.VGG11_Weights.DEFAULT
preprocess = weights.transforms()

# Apply it to the input image
model = torchvision.models.vgg11(weights=weights)
model.eval()


goldfish_img = Image.open(os.path.join(dir, 'gold_goldfish.png'))
goldfish = preprocess(goldfish_img).unsqueeze(0)

cockfish_img = Image.open(os.path.join(dir, 'gold_cockfish.png'))
cockfish = preprocess(cockfish_img).unsqueeze(0)


y = model(goldfish)
smax = F.softmax(y, dim=1)
top = torch.topk(smax, 5)
for i, p in zip(top.indices[0], top.values[0]):
    print(class_names[int(i.item())], p.item())

goldfish_img.show()
print("----------------------------")

y = model(cockfish)
smax = F.softmax(y, dim=1)
top = torch.topk(smax, 5)
for i, p in zip(top.indices[0], top.values[0]):
    print(class_names[int(i.item())], p.item())

cockfish_img.show()
