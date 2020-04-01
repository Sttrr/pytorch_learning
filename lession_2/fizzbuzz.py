import numpy as np
import torch

def fizz_buzz_encode(i):
    if i%3==0:return 1
    elif i%5==0:return 2
    elif i%15==0:return 3
    else :return 0

def fizz_buzz_decode(i,prediction):
    return [str(i),"fizz","buzz","fizzbuzz"][prediction]

def helper(i):
    print(fizz_buzz_decode(i,fizz_buzz_encode(i)))


NUM_DIGIT=10

def binary_encode(i,num_digit):
    return np.array([i>>d&1 for d in range(num_digit)][::-1])

trX=torch.Tensor([binary_encode(i,NUM_DIGIT) for i in range(101,2**NUM_DIGIT)])
trY=torch.LongTensor([fizz_buzz_encode(i) for i in range(101,2**NUM_DIGIT)])

H=100

model=torch.nn.Sequential(
    torch.nn.Linear(10,100),
    torch.nn.ReLU(),
    torch.nn.Linear(100,4)
).cuda()

learining_rate=0.05

optimizer=torch.optim.SGD(model.parameters(),lr=learining_rate)
loss_fn=torch.nn.CrossEntropyLoss()

batch_size=128
for epoch in range(100):
    for start in range(0,len(trX),batch_size):
        end=start+batch_size
        batchX=trX[start:end]
        batchY=trY[start:end]

        if torch.cuda.is_available():
            batchX=batchX.cuda()
            batchY=batchY.cuda()

        y_pred=model(batchX)
        loss=loss_fn(y_pred,batchY)
        #print(y_pred.shape,batchY.shape)
        print(epoch,loss.item())
    
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
'''
teX=torch.Tensor([binary_encode(i,NUM_DIGIT) for i in range(1,100)])
teY=torch.LongTensor([fizz_buzz_encode(i) for i in range(1,100)])
teX=teX.cuda()
teY=teY.cuda()

y_pre=model(teX)
loss_te=loss_fn(y_pre,teY)

print("test:",loss.item())
'''