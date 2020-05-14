import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torchvision import transforms,datasets
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size=32
num_epochs=15
num_classes=2
model_name="resnet"
data_dir="./hymenoptera_data"
feature_extract=True
input_size=224

all_imgs=datasets.ImageFolder(os.path.join(data_dir,"train"),transforms.Compose([
    transforms.RandomResizedCrop(input_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
]))

loader=torch.utils.data.DataLoader(all_imgs,batch_size,shuffle=True,num_workers=4)

unloader=transforms.ToPILImage()

def set_parameter_requires_grad(model,feature_extract):
    if feature_extract:
        for param in model.parameters():
            param.requires_grad=False

def initialize_model(model_name,num_classes,feature_extract,use_pretrained=True):
    if model_name=="resnet":
        model_ft=models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft,feature_extract)
        num_ftrs=model_ft.fc.in_features
        model_ft.fc=nn.Linear(num_ftrs,num_classes)
        input_size=224
    else:
        print('model not implemented')

    return model_ft,input_size

model_ft,input_size=initialize_model(model_name,num_classes,feature_extract,use_pretrained=True)

print(model_ft)
