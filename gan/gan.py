import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets,transforms
from torchvision import utils as vutils

import numpy as np
import matplotlib.pyplot as plt

manualseed=999
print("random seed:",manualseed)
random.seed(manualseed)
torch.manual_seed(manualseed)

dataroot="./"
num_epochs=1
learning_rate=0.0002
num_works=4
batch_size=128
beta1 = 0.5

dataset=datasets.ImageFolder("/home/mlinux/Datasets/celeba/",
transform=transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
]))

dataloader=torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=4)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

image_batch=next(iter(dataloader))
'''
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("training image")
plt.imshow(np.transpose(vutils.make_grid(image_batch[0].to(device)[:64],normalize=True).cpu(),(1,2,0)))
plt.show()
'''
def weight_init(m):
    classname=m.__class__.__name__
    if classname.find("Conv")!=-1:
        nn.init.normal_(m.weight.data,0.0,0.02)
    elif classname.find("BatchNorm")!=-1:
        nn.init.normal_(m.weight.data,0.0,0.02)
        nn.init.constant_(m.bias.data,0)

class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.main=nn.Sequential(
            nn.ConvTranspose2d(100,64*8,4,1,bias=False),
            nn.BatchNorm2d(64*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(64*8,64*4,4,2,1,bias=False),
            nn.BatchNorm2d(64*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(64*4,64*2,4,2,1,bias=False),
            nn.BatchNorm2d(64*2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(64*2,64,4,2,1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64,3,4,2,1,bias=False),
            nn.Tanh()
        )
    def forward(self,input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.main=nn.Sequential(
            nn.Conv2d(3,64,4,2,1,bias=False),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(64,64*2,4,2,1,bias=False),
            nn.BatchNorm2d(64*2),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(64*2,64*4,4,2,1,bias=False),
            nn.BatchNorm2d(64*4),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(64*4,64*8,4,2,1,bias=False),
            nn.BatchNorm2d(64*8),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(64*8,1,4,1,bias=False),
            nn.Sigmoid()
        )

    def forward(self,input):
        return self.main(input)


netG=Generator()
netD=Discriminator()

netG,netD=netG.to(device),netD.to(device)

netG.apply(weight_init)
netD.apply(weight_init)

loss=nn.BCELoss()

optimizerG=optim.Adam(netG.parameters(),lr=learning_rate,betas=(beta1, 0.999))
optimizerD=optim.Adam(netD.parameters(),lr=learning_rate,betas=(beta1, 0.999))

fixd_noise=torch.randn(64,100,1,1,device=device)

real_label=1
fake_label=0

image_list=[]
D_loss=[]
G_loss=[]
iters=0

for epoch in range(num_epochs):
    for i,data in enumerate(dataloader,0):
        real_cpu=data[0].to(device)
        pred=netD(real_cpu).view(-1)

        b_size=real_cpu.size(0)
        label=torch.full((b_size,),real_label,device=device)
        error_d_real=loss(pred,label)

        optimizerD.zero_grad()
        error_d_real.backward()
        D_x=pred.mean().item()

        noise=torch.randn(b_size,100,1,1,device=device)
        fake=netG(noise)
        label.fill_(fake_label)
        output=netD(fake.detach()).view(-1)
        error_d_fake=loss(output,label)
        error_d_fake.backward()
        D_G_z1=output.mean().item()

        error_D=error_d_real+error_d_fake
        optimizerD.step()

        output=netD(fake).view(-1)
        label.fill_(real_label)
        error_G=loss(output,label)
        optimizerG.zero_grad()
        error_G.backward()
        D_G_z2=output.mean().item()
        optimizerG.step()

        if i%50==0:
            print(epoch,i,error_D,error_G,D_x,D_G_z1,D_G_z2)

print(len(dataloader))