import numpy as np 
from keras.datasets import mnist
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from Chapter3 import sigmoid,softmax
from Chapter4 import cross_entropy_error_onehot,numerical_gradient2
from keras.utils import to_categorical

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
        grads['W1']=numerical_gradient2(loss_W,self.params['W1'])
        grads['b1']=numerical_gradient2(loss_W,self.params['b1'])
        grads['W2']=numerical_gradient2(loss_W,self.params['W2'])
        grads['b2']=numerical_gradient2(loss_W,self.params['b2'])

        return grads


mnist_X, mnist_y = fetch_openml("mnist_784", version=1, data_home=".", return_X_y=True)

X = mnist_X.astype("float64").to_numpy()
y = mnist_y.astype(int).to_numpy()

# 学習用と評価用に分割
X_train, X_test, t_train, t_test = train_test_split(X, y, test_size=10000)
#one-hot表現に変換
t_train_one_hot = to_categorical(t_train, num_classes=10)
t_test_one_hot = to_categorical(t_test, num_classes=10)

train_loss_list=[]

network=TwoLayerNet(input_size=784,hidden_size=50,output_size=10)

#ハイパーパラメータ
iters_num=100 #実行回数
train_size=X_train.shape[0]
batch_size=100
learning_rate=1 #学習率
train_loss_list, train_acc_list, test_acc_list = [], [], []
iter_per_epoch=1 # 精度表示は 1 epoch 毎から 1 iter 毎に変更


for i in range(iters_num):
    #ミニバッチの取得
    batch_mask=np.random.choice(train_size,batch_size)
    x_batch=X_train[batch_mask]
    t_batch=t_train_one_hot[batch_mask]

    #勾配の計算
    grad=network.numerical_gradient(x_batch,t_batch)
    #grad=network.gradient(x_batch,t_batch)#高速版

    #パラメータの更新
    for key in ('W1','b1','W2','b2'):
        network.params[key]-=learning_rate*grad[key]

    #学習経過の記録
    loss=network.loss(x_batch,t_batch)
    train_loss_list.append(loss)

    #精度の表示
    if i % iter_per_epoch==0:
        train_acc=network.accuracy(X_train,t_train_one_hot)
        test_acc=network.accuracy(X_test,t_test_one_hot)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        #表示
        print('[iter='+str(i)+']'+'train_loss='+str(loss)+','+'train_acc='+str(train_acc)+','+'test_acc='+str(test_acc))
