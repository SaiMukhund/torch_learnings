import torch
import torch.nn as nn
import numpy as np 
from sklearn import datasets
import matplotlib.pyplot as plt 


# 0)  prepare data 
x_numpy,y_numpy =datasets.make_regression(n_samples=100,n_features=1,random_state=1)
print(x_numpy.shape,x_numpy)
print(y_numpy.shape,y_numpy)
X=torch.tensor(x_numpy,dtype=torch.float32)
Y=torch.tensor(y_numpy, dtype=torch.float32)
Y=Y.view(-1,1)


# initialize a model 
n_samples,n_features=X.shape
n_input=n_features
n_output=1
model=nn.Linear(n_input,n_output)


### loss and optimizer
loss=nn.MSELoss()
learning_rate=0.001
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)

##  training loop 
n_iters=10000

for epoch in range(n_iters):
    y_pred=model(X)
    l=loss(y_pred,Y)
    l.backward()

    optimizer.step()
    optimizer.zero_grad()
    if(epoch%10 == 0):
        print(f"epoch {epoch} | loss: {l.item():.3f}")


predicted=model(X).detach().numpy()

plt.plot(x_numpy,y_numpy,'ro')
plt.plot(x_numpy,predicted,'b')
plt.show()

