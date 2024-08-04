import numpy as np
x=np.array([1,2,3,4], dtype=np.float32)
y=np.array([2,4,6,8], dtype=np.float32)

w=0.0

#model prediction
def forward(x):
    return w * x
#loss
def loss(y,y_predicted):
    return ((y_predicted-y)**2).mean()
#gradient
def gradient(x,y,y_predicted):
    return np.dot(2*x,y_predicted-y).mean()

print(f'Prediction before training: f(5)={forward(5):.3f}')

learning_rate=0.02
n_iters=100

for epoch in range(n_iters):
    #prediction
    y_pred=forward(x)
    #loss
    l=loss(y,y_pred)
    #gradients
    dw=gradient(x,y,y_pred)
    #update weights
    w-= learning_rate*dw
    print(f'epoch{epoch+1}:w={w:.3f} ,loss={l:.8f}')
print(f'Prediction after training: f(5)={forward(5):.3f}')