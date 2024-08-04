# 1)Design model [forward pass]
# 2)Constrruct loss and optimizer
# 3)Training loop
# forward pass ,backward pass,update weights





import torch
import torch.nn as nn

x=torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
y=torch.tensor([[2],[4],[6],[8]], dtype=torch.float32)


x_test=torch.tensor([5],dtype=torch.float32)
n_samples,n_features=x.shape

input_size=n_features
output_size=n_features

model=nn.Linear(input_size,output_size)
#loss



#gradient


print(f'Prediction before training: f(5)={model(x_test).item():.3f}')

learning_rate=0.01
n_iters=100

loss=nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)

for epoch in range(n_iters):
    #prediction
    y_pred=model(x)

    #loss
    l=loss(y,y_pred)

    #gradients
    l.backward() #dl/dw

    #update weights
    optimizer.step()

    optimizer.zero_grad()
    [w,b]=model.parameters()
    print(f'epoch{epoch+1}:w={w[0][0].item():.3f} ,loss={l:.8f}')
print(f'Prediction after training: f(5)={model(x_test).item():.3f}')