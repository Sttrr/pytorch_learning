import numpy as np

N,D_in,H,D_out=60,1000,100,10
#随机训练数据
x=np.random.randn(N,D_in)
y=np.random.randn(N,D_out)

w1=np.random.randn(D_in,H)
W2=np.random.randn(H,D_out)

learning_rate=1e-6

for it in range(500):
    #forward pass
    h=x.dot(w1)
    h_relu=np.maximum(h,0)
    y_pred=h_relu.dot(W2)

    #compute loss
    loss=np.square(y_pred-y).sum()
    print(it,loss)

    #backforward pass
    #compute gradient
    grad_y_pred=2.0*(y_pred-y)
    grad_w2=h_relu.T.dot(grad_y_pred)
    grad_h_relu=grad_y_pred.dot(W2.T)
    grad_h=grad_h_relu.copy()
    grad_h[h<0]=0
    grad_w1=x.T.dot(grad_h)

    #update the parameter of w1 and w2
    w1=w1-learning_rate*grad_w1
    W2=W2-learning_rate*grad_w2

h=x.dot(w1)
h_relu=np.maximum(h,0)
y_pred=h_relu.dot(W2)
print('y=',y)
print('y_pred=',y_pred)
print('loss=',y-y_pred)