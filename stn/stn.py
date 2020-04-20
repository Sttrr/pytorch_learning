import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets,transforms

import numpy as np
import matplotlib.pyplot as plt

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loader=torch.utils.data.DataLoader(
    datasets.MNIST(root='/home/mlinux/Datasets',train=True,download=True,transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,),(0.3081,))])
            ),batch_size=64,shuffle=True,num_workers=4)

train_loader=torch.utils.data.DataLoader(
    datasets.MNIST(root='/home/mlinux/Datasets',train=False,transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,),(0.3081,))])
            ),batch_size=64,shuffle=True,num_workers=4)


class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=nn.Conv2d(1,10,kernel_size=5)
        self.conv2=nn.Conv2d(10,20,kernel_size=5)
        self.conv2_drop=nn.Dropout2d()
        self.fc1=nn.Linear(320,50)
        self.fc2=nn.Linear(50,10)

        #localization network
        self.localization=nn.Sequential(
            nn.Conv2d(1,8,kernel_size=7),
            nn.MaxPool2d(2,stride=2),
            nn.ReLU(True),
            nn.Conv2d(8,10,kernel_size=5),
            nn.MaxPool2d(2,stride=2),
            nn.ReLU(True)
        )

        #regresor for the 3*2 affine matrix
        self.fc_loc=nn.Sequential(
            nn.Linear(10*3*3,32),
            nn.ReLU(True),
            nn.Linear(32,3*2)
        )

        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1,0,0,0,1,0],dtype=torch.float))


    def stn(self,x):
        xs=self.localization(x)
        xs=xs.view(-1,10*3*3)
        theta=self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid=F.affine_grid(theta,x.size())
        x=F.grid_sample(x,grid)

        return x
        
        #F.relu()和nn.ReLU()的区别？一个是module一个是函数？
    def forward(self,x):
        x=self.stn(x)

        x=F.relu(F.max_pool2d(self.conv1(x),2))
        x=F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)),2))
        x=x.view(-1,320)
        x=F.relu(self.fc1(x))
        x=F.dropout(x,training=self.training)
        x=self.fc2(x)

        return F.log_softmax(x,dim=1)

model=Net().to(device)

optimizer=optim.SGD(model.parameters(),lr=0.01)

def train(epoch):
    model.train()
    for batch_idx,(data,target) in enumerate(train_loader):
        data,target=data.to(device),target.to(device)
        optimizer.zero_grad()

        output=model(data)
        loss=F.nll_loss(output,target)
        loss.backward()
        optimizer.step()

        if batch_idx%50==0:
            print(epoch,batch_idx,loss.item())

for epoch in range(1,10+1):
    train(epoch)



def test():

#it=enumerate(train_loader)
#print(next(it))
#print(len(train_loader.dataset))