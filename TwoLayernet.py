import sys,os
import numpy as np
sys.path.append(os.pardir)

from Chapter3 import sigmoid,softmax
from Chapter4 import cross_entropy_error_onehot,numerical_gradient

class TwoLayerNet():

    def __init__(self,input_size,hidden_size,output_size,weight_init_std=0.01):
        #重みの初期化
        #np.random.randnは標準正規分布に従う乱数を生成する関数で例えば(2,3)を渡すと2行3列のランダムな数の行列を生成する
        self.params={}
        self.params['W1']=weight_init_std*np.random.randn(input_size,hidden_size) 
        self.params['b1']=np.zeros(hidden_size) #全ての要素が0の行列を生成する
        self.params['W2']=weight_init_std*np.random.randn(hidden_size,output_size)
        self.params['b2']=np.zeros(output_size)

    def predict(self,x):
        W1,W2=self.params['W1'],self.params['W2']
        b1,b2=self.params['b1'],self.params['b2']

        a1=np.dot(x,W1)+b1 
        z1=sigmoid(a1)
        a2=np.dot(z1,W2)+b2
        y=softmax(a2)

        return y
    
    #x:入力データ,t:教師データ
    def loss(self,x,t): #one_hot対応
        y=self.predict(x)

        return cross_entropy_error_onehot(y,t)
    
    def accuracy(self,x,t):
        y=self.predict(x)
        y=np.argmax(y,axis=1) #np.argmaxは各行の最大値のインデックスを返す関数
        t=np.argmax(t,axis=1)

        accuracy=np.sum(y==t)/float(x.shape[0])
        return accuracy
    
    #x:入力データ,t:教師データ
    def numerical_gradient(self,x,t):
        loss_W=lambda W: self.loss(x,t)
        grads={}
        grads['W1']=numerical_gradient(loss_W,self.params['W1'])
        grads['b1']=numerical_gradient(loss_W,self.params['b1'])
        grads['W2']=numerical_gradient(loss_W,self.params['W2'])
        grads['b2']=numerical_gradient(loss_W,self.params['b2'])

        return grads
