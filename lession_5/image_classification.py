import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import transforms,datasets

train_loader=torch.utils.data.DataLoader(datasets.MNIST("/home/mlinux/Datasets",train=True,download=True,
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,))])),batch_size=64,shuffle=True,num_workers=4)

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=nn.Conv2d(1,20,kernel_size=5)
        self.conv2=nn.Conv2d(20,50,kernel_size=5)
        self.fc1=nn.Linear(50*4*4,150)
        self.fc2=nn.Linear(150,10)

    def forward(self,x):
        x=F.max_pool2d(F.relu(self.conv1(x)),kernel_size=2,stride=2)
        x=F.max_pool2d(F.relu(self.conv2(x)),kernel_size=2,stride=2)
        x=x.view(-1,50*4*4)
        x=F.relu(self.fc1(x))
        x=self.fc2(x)
        return F.log_softmax(x,dim=1)

EPOCH=1
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=Net().to(device)
optimzer=optim.Adam(model.parameters(),lr=0.001)

for i in range(EPOCH):
    for idx,(data,target) in enumerate(train_loader):
        model.train()
        data,target=data.to(device),target.to(device)
        
        pred=model(data)
        loss=F.nll_loss(pred,target)

        optimzer.zero_grad()
        loss.backward()
        optimzer.step()
        
        print(idx,loss.item())

#torch.save(model.state_dict(),'mnist_cnn.pt')