import torch

if torch.cuda.is_available():
    device=torch.device('cuda')


N,D_in,H,D_out=60,1000,100,10

x=torch.randn(N,D_in)
y=torch.randn(N,D_out)

w1=torch.randn(D_in,H,requires_grad=True)
w2=torch.randn(H,D_out,requires_grad=True)

learning_rate=1e-6

for it in range(500):
    #forward pass
    y_pred=x.mm(w1).clamp(min=0).mm(w2)

    #compute loss
    loss=(y_pred-y).pow(2).sum()
    print(it,loss.item())

    #backword pass
    #compute gradient
    '''
    grad_y_pred=2.0*(y_pred-y)
    grad_w2=h_relu.t().mm(grad_y_pred)
    grad_h_relu=grad_y_pred.mm(w2.t())
    grad_h=grad_h_relu.clone()
    grad_h[h<0]=0
    grad_w1=x.t().mm(grad_h)
    '''
   
    loss.backward()

    with torch.no_grad():
        #update the parameter of w1 and w2
        w1-=learning_rate*w1.grad
        w2-=learning_rate*w2.grad

        w1.grad.zero_()
        w2.grad.zero_()
    

h=x.mm(w1)
h_relu=h.clamp(min=0)
y_pred=h_relu.mm(w2)

print('y=',y)
print('y_pred=',y_pred)
print('loss=',y-y_pred)
