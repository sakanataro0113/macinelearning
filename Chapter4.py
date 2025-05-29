#ニューラルネットワークの学習
import numpy as np
import matplotlib.pyplot as plt

#損失関数の実装

#2乗和誤差

#y:ニューラルネットワークの出力
#t:正解ラベル

def sum_squared_error(y,t):
    return 0.5*np.sum((y-t)**2)

#実際に使ってみる
#「2」を正解とする
t=[0,0,1,0,0,0,0,0,0,0]

#例1:「2」の確率が最も高い場合(0.6)
y=[0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0]
print(sum_squared_error(np.array(y),np.array(t)))

#例2:「7」の確率が最も高い場合(0.6)
y=[0.1,0.05,0.1,0.0,0.05,0.1,0.0,0.6,0.0,0.0]
print(sum_squared_error(np.array(y),np.array(t)))

#一つ目の損失関数の方が値が小さくなっており、教師データとの誤差が小さい、つまり出力結果が教師データにより適合している事が分かる
print("#############################################################################################################")
#交差エントロピー誤差
def cross_entropy_error(y,t):
    delta=1e-7
    return -np.sum(t*np.log(y+delta))

#実際に使ってみる
#「2」を正解とする
t=[0,0,1,0,0,0,0,0,0,0]

#例1:「2」の確率が最も高い場合(0.6)
y=[0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0]
print(cross_entropy_error(np.array(y),np.array(t)))

#例2:「7」の確率が最も高い場合(0.6)
y=[0.1,0.05,0.1,0.0,0.05,0.1,0.0,0.6,0.0,0.0]
print(cross_entropy_error(np.array(y),np.array(t)))

#(バッチ対応版)交差エントロピー誤差
def cross_entropy_error_onehot(y,t):
    if y.ndim==1:
        t=t.reshape(1,t.size)
        y=y.reshape(1,y.size)

    batch_size=y.shape[0]
    return -np.sum(t*np.log(y+1e-7))/batch_size

#教師データがone-hot表現ではなく、ラベル表現の場合
def cross_entropy_error(y,t):
    if y.ndim==1:
        t=t.reshape(1,t.size)
        y=y.reshape(1,y.size)

    batch_size=y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size),t]+1e-7))/batch_size

#微分
#数値微分の悪い実装例
def numerical__diff(f,x):
    h=1e-50
    return (f(x+h)-f(x))/h

#hは0に無限に近づけたいため、できるだけ小さな値を入力したい
#h=1e-50では丸め誤差が発生してしまう
#丸め誤差の例
print(np.float32(1e-50))

#x+hとxの間での関数fの差分を計算しているが、本来の微分ではxの位置での傾き、すなわち接戦に対応するが悪い例の数値微分ではx+hとxの間の傾きに対応している。
#この差異はhを無限に0に近づけることで解消されるが、hを0に近づけることで丸め誤差が発生してしまう。

#誤差を減らすためにx+hとx-hでの関数fの差分を計算する。これを中心差分という
#ここで行っているように微小な差分によって微分を求めることを数値微分という

#数値微分の良い実装例
def numerical_diff(f,x):
    h=1e-4
    return (f(x+h)-f(x-h))/(2*h)

#偏微分
#引数2つの2乗和を計算(引数numpy配列だと仮定)
def function_2(x):
    return x[0]**2+x[1]**2

#↑グラフ出力
from mpl_toolkits.mplot3d import Axes3D
#範囲と間隔の設定
x1=np.arange(-3,3,0.1)
x2=np.arange(-3,3,0.1)

#メッシュ描画設定
x1,x2=np.meshgrid(x1,x2)

#z軸の値を計算
z=function_2(np.array([x1,x2]))

fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')

#プロット
ax.set_zlim(0,18)#z軸の範囲
ax.plot_wireframe(x1,x2,z)
plt.show()

#変数ごとに偏微分を行う
#ex1)x0=3,x1=4のときのx0に対する偏微分
def function_tmp1(x0):
    return x0*x0+4.0**2.0

#ex2)x0=3,x1=4のときのx1に対する偏微分
def function_tmp2(x1):
    return 3.0**2.0+x1*x1

#ex1)出力
print(numerical_diff(function_tmp1,3.0))
#ex2)出力
print(numerical_diff(function_tmp2,4.0))

#変数をまとめて偏微分する 勾配
def numerical_gradient(f,x):
    h=1e-4
    grad=np.zeros_like(x) #xと同じ形状の配列を作成

    for idx in range(x.size):
        tmp_val=x[idx]
        #f(x+h)の計算
        x[idx]=tmp_val+h
        fxh1=f(x)

        #f(x-h)の計算
        x[idx]=tmp_val-h
        fxh2=f(x)

        grad[idx]=(fxh1-fxh2)/(2*h)
        x[idx]=tmp_val #元の値に戻す

    return grad

def numerical_gradient2(f, x):   
        h = 1e-4 # 0.0001
        # 勾配の計算結果を保管するゼロ行列 grad(サイズはxで指定)を準備
        grad = np.zeros_like(x)
        # 行列 x を順次インデックス指定(行と列を指定)する
        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            tmp_val = x[idx]
            x[idx] = tmp_val + h
            fxh1 = f(x) # f(x+h)
        
            x[idx] = tmp_val - h 
            fxh2 = f(x) # f(x-h)
            grad[idx] = (fxh1 - fxh2) / (2*h)
        
            x[idx] = tmp_val # 値を元に戻す
            it.iternext()           
        return grad  

#勾配の計算
#x0=3,x1=4のときの勾配
print(numerical_gradient(function_2,np.array([3.0,4.0])))
#x0=0,x1=2のときの勾配
print(numerical_gradient(function_2,np.array([0.0,2.0])))
#x0=3,x1=0のときの勾配
print(numerical_gradient(function_2,np.array([3.0,0.0])))

#勾配が示す方向は各場所において、関数の値を最も減らす方向

#勾配降下法の実装(この関数では関数の極小値を求めることができ、うまくいけば最小値を求めることができる)
def gradient_descent(f,init_x,lr=0.01,step_num=100):#f:最適化したい関数,init_x:初期値,lr:学習率,step_num:勾配法による学習回数
    x=init_x

    for i in range(step_num):
        grad=numerical_gradient(f,x)#数値微分(関数の勾配を求める)
        x-=lr*grad#学習率をかけて更新
    return x

#ex)f(x0,x1)=x0^2+x1^2の最小値を勾配法で求めよ
init_x=np.array([-3.0,4.0])#初期値
print(gradient_descent(function_2,init_x,lr=0.1,step_num=100))








