import numpy as np
import matplotlib.pyplot as plt

#step関数
def step_function(x):
    if x>0:
        return 1
    else:
        return 0

#↑だとNumpyの配列を引数に取れないから、改良
def step_function_ex(x):
    y=x > 0
    return y.astype(int)

#-1.0,1.0,2.0を渡すと0以上の時1、0より小さい時0になる
x=np.array([-1.0,1.0,2.0])
print(step_function_ex(x))

#step関数のグラフ

def step_function_graph(x):
        return np.array(x>0,dtype=int)

x=np.arange(-5.0,5.0,0.1)
y=step_function_graph(x)
plt.plot(x,y)
plt.ylim(-0.1,1.1)#y軸の範囲を指定
plt.show()

#シグモイド関数:ブロードキャストによりNumPy配列にも対応
def sigmoid(x):
    return 1/(1+np.exp(-x))

x=np.arange(-5.0,5.0,0.1)
y=sigmoid(x)
plt.plot(x,y)
plt.ylim(-0.1,1.1)#y軸の範囲を指定
plt.show()

#ReLu関数
def relu(x):
     return np.maximum(0,x)

#ソフトマックス関数
def softmax(a):
     exp_a=np.exp(a)
     sum_exp_a=np.sum(exp_a)
     y=exp_a/sum_exp_a

     return y





