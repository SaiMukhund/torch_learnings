import torch
import torch.nn as nn
import numpy as np 
from sklearn import datasets
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 0)  prepare data 
d=datasets.load_breast_cancer()
x=d.data
y=d.target

X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=1234)

sc=StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train=torch.from_numpy(X_train.astype(np.float32))
X_test=torch.from_numpy(X_test.astype(np.float32))
Y_train=torch.from_numpy(Y_train.astype(np.float32))
Y_test=torch.from_numpy(Y_test.astype(np.float32))

Y_train=Y_train.view(-1,1)
Y_test=Y_test.view(-1,1)

n_samples,n_features=X_train.shape

# model 
n_input=n_features
n_output=1

class LogisticRegression(nn.Module):
    def __init__(self, n_input) -> None:
        super(LogisticRegression,self).__init__()
        self.linear=nn.Linear(n_input,1)
    def forward(self,x):
        y_pred=torch.sigmoid(self.linear(x))
        return y_pred
model=LogisticRegression(n_input)

## loss and optimizer
loss=nn.BCELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.1)

## training
n_iters=10000
for epoch in range(n_iters):
    y_pred=model(X_train)
    l=loss(y_pred,Y_train)
    l.backward()

    optimizer.step()

    optimizer.zero_grad()
    if epoch%10==0:
        print(f'epoch {epoch} | loss: {l.item():.3f}')

with torch.no_grad():
    y_pred=model(X_test)
    y_pred_cls=y_pred.round()
    accuracy=y_pred_cls.eq(Y_test).sum()/float(Y_test.shape[0])
    print(f"aaccuarcy : {accuracy}")
    plt.scatter(X_test[:,0],y_pred_cls)
    plt.scatter(X_test[:,0],Y_test)
    plt.show()