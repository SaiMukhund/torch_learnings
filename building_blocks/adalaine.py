import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap
def plot_decision_regions(X, y, classifier, resolution=0.02):

    # setup marker generator and color map
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=f'Class {cl}', 
                    edgecolor='black')


class Adaline():

    def __init__(self,learning_rate,n_epochs):

        self.learning_rate=learning_rate
        self.n_epochs=n_epochs 

    def fit(self,X,y):

        """
        X: samples [n_samples,n_features]
        y: n_samples 
        """
        n_features=X.shape[1]
        self.w_=np.random.randn(n_features)*0.01
        self.b_=np.random.randn()*0.01

        self.losses=[]

        for epochs in range(self.n_epochs):
            y_pred=self.activation(self.net_input(X))
            errors=y-y_pred
            loss=0.5*(errors**2).mean()
            self.losses.append(loss)
            self.w_+=self.learning_rate*X.T.dot(errors)/X.shape[0]
            self.b_+=self.learning_rate*errors.mean()
            
    
    def net_input(self,x):
        return np.dot(x,self.w_)+self.b_

    def activation(self,x):
        return x 

    def predict(self,x):
        return np.where(self.activation(self.net_input(x))>=0.5,1,0)
def main():
    import pandas as pd 
    import matplotlib.pyplot as plt 
    iris_data=pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",header=None,encoding="utf-8")
    X=iris_data.iloc[:,].values
    y=iris_data.iloc[:,-1].values
    print(X.shape)
    y=np.where(y=="Iris-virginica",0,1)
    iris_classifier=Adaline(learning_rate=0.001,n_epochs=100)
    iris_classifier.fit(X,y)
    plt.plot(range(1,len(iris_classifier.losses)+1),iris_classifier.losses)
    plt.show()
if __name__=="__main__":
    main()