import numpy as np
from tqdm import tqdm 
class Perceptron():

    def __init__(
        self,
        eta:float=0.001,
        n_iter:int=50,
        random_state:int=1):

        self.eta=eta
        self.n_iter=n_iter
        self.random_state=random_state
    
    def fit(self,X,y):

        """
        X: shape is [n_samples,n_features]
        y: shape is [n_samples,1]
        """
        rng=np.random.default_rng(seed=self.random_state)
        self.w_=rng.normal(loc=0.0,scale=0.01,size=X.shape[1])
        #self.w_=np.zeros(shape=X.shape[1],dtype=np.float32)
        self.b_=0.0
        self.errors=[]
        for _ in tqdm(range(self.n_iter)):
            error=0
            precited_out=[]
            for x,target in zip(X,y):
                predict_output=self.predict(x)
                precited_out.append(predict_output)
                loss=target-predict_output
                self.w_ = self.w_+ self.eta*loss*x
                self.b_ = self.b_+self.eta*loss
            for x,target in zip(X,y):
                predict_output=self.predict(x)
                loss=target-predict_output
                error =error+ int(loss!=0.0)
            self.errors.append(error)
        return self 


    def predict(self,X):
        z=np.dot(self.w_,X)+self.b_
        return np.where(z>=0,1,0)


def main():
    import pandas as pd 
    import matplotlib.pyplot as plt 
    iris_data=pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",header=None,encoding="utf-8")
    X=iris_data.iloc[:,:-1].values
    y=iris_data.iloc[:,-1].values
    print(X.shape)
    y=np.where(y=="Iris-virginica",0,1)
    iris_classifier=Perceptron(eta=0.001,n_iter=5)
    iris_classifier.fit(X,y)
    plt.plot(range(1,len(iris_classifier.errors)+1),iris_classifier.errors)
    plt.show()
if __name__=="__main__":
    main()