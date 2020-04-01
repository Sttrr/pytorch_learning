import torch
import torch.nn as nn
N,D_in,H,D_out=64,1000,100,10

x=torch.randn(N,D_in)
y=torch.randn(N,D_out)

model=torch.nn.Sequential(
    torch.nn.Linear(D_in,H),
    torch.nn.ReLU(),
    torch.nn.Linear(H,D_out)
)

torch.nn.init.normal_(model[0].weight)
torch.nn.init.normal_(model[2].weight)


loss_fn=nn.MSELoss(reduction='sum')

#learning_rate=1e-4
#optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
learning_rate=1e-6
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)

for it in range(500):
    y_pred=model(x)

    loss=loss_fn(y_pred,y)
    print(it,loss.item())

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()
        

    
