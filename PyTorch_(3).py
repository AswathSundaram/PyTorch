# 1)Design model [forward pass]
# 2)Constrruct loss and optimizer
# 3)Training loop
# forward pass ,backward pass,update weights





import torch
import torch.nn as nn

x=torch.tensor([1,2,3,4], dtype=torch.float32)
y=torch.tensor([2,4,6,8], dtype=torch.float32)

w=torch.tensor(0.0,dtype=torch.float32,requires_grad=True)

#model prediction
def forward(x):
    return w * x

#loss



#gradient


print(f'Prediction before training: f(5)={forward(5):.3f}')

learning_rate=0.01
n_iters=20

loss=nn.MSELoss()
optimizer=torch.optim.SGD([w],lr=learning_rate)

for epoch in range(n_iters):
    #prediction
    y_pred=forward(x)

    #loss
    l=loss(y,y_pred)

    #gradients
    l.backward() #dl/dw

    #update weights
    optimizer.step()

    optimizer.zero_grad()

    print(f'epoch{epoch+1}:w={w:.3f} ,loss={l:.8f}')
print(f'Prediction after training: f(5)={forward(5):.3f}')