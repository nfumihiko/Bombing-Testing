# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

h = 0.001 #刻み幅
t = np.arange(0,1,h) #h刻みで時間0~Tを分割
NN = 10**3    # 積分項の分割数

gain = 7 #シグモイド関数内のゲイン


constM_posi = 8.0#獲得情報
constM_nega = 4.0


def sigmoid(x):
    return 1/(1+np.exp(-gain*(x-0.5)))
    
def f_posi_true(v,consta):
    return sigmoid(v)*(1-consta*(1-sigmoid(v)))
def f_nega_false(v,constb):
    return (1-sigmoid(v))*constb*sigmoid(v)
def f_posi_false(v,consta):
    return sigmoid(v)*consta*(1-sigmoid(v))
def f_nega_true(v,constb):
    return (1-sigmoid(v))*(1-constb*sigmoid(v))
    
def info_posi(v,consta,constb):
    return -constM_posi*(f_posi_true(v,consta)/(f_posi_true(v,consta)+f_nega_false(v,constb))*np.log(f_posi_true(v,consta)/(f_posi_true(v,consta)+f_nega_false(v,constb)))+f_nega_false(v,constb)/(f_posi_true(v,consta)+f_nega_false(v,constb))*np.log(f_nega_false(v,constb)/(f_posi_true(v,consta)+f_nega_false(v,constb))))
 
def info_nega(v,consta,constb):
    return -constM_nega*(f_nega_true(v,constb)/(f_nega_true(v,constb)+f_posi_false(v,consta))*np.log(f_nega_true(v,constb)/(f_nega_true(v,constb)+f_posi_false(v,consta)))+f_posi_false(v,consta)/(f_nega_true(v,constb)+f_posi_false(v,consta))*np.log(f_posi_false(v,consta)/(f_nega_true(v,constb)+f_posi_false(v,consta))))
    
    
def G(t,cost,risk,consta,constb):
    return (cost-risk)*f_nega_false(t,constb)+cost*f_nega_true(t,constb)+info_nega(t,consta,constb)*(f_nega_true(t,constb)+f_posi_false(t,consta))+risk*f_posi_true(t,consta)+ info_posi(t,consta,constb)*(f_posi_true(t,consta)+f_nega_false(t,constb))
# グラフで可視化
fig = plt.figure(figsize=(10,5))
plt.subplots_adjust(wspace=0.2, hspace=0.3)
ax1 = fig.add_subplot(121) # 左図
ax2 = fig.add_subplot(122) # 右図



sensitivity = [0.7, 0.7, 0.8, 0.8, 0.9, 0.9] #感度
specificity = [0.9, 0.99, 0.9, 0.99, 0.9, 0.99] #特異度
cost = 3 #治療コスト
risk = 4 #感染リスク
#価値分布の密度関数g(v)を定義
def dist(x):
    #return 4*x*(1-x)
    return x*(1-x)
    
hh = 1/NN    # 微小区間の幅
p = np.linspace(0, 1, NN+1)   # 積分区間を NN 等分する
ff1 = sigmoid(p)*(1-sigmoid(p))*dist(p)
S1 = hh*(np.sum(ff1)-ff1[0]/2-ff1[NN]/2)
ff2 = sigmoid(p)*dist(p)
S2 = hh*(np.sum(ff2)-ff2[0]/2-ff2[NN]/2)
ff3 = (1-sigmoid(p))*dist(p)
S3 = hh*(np.sum(ff3)-ff3[0]/2-ff3[NN]/2)

print("p(1-p):"+str(S1))
print("p:"+str(S2))
print("1-p:"+str(S3))

for i in range(6):
    #比例定数を決定

    consta = (1-sensitivity[i])*S2/S1
    constb = (1-specificity[i])*S3/S1

    print("a:"+str(i+1)+":"+str(consta))
    print("b:"+str(i+1)+":"+str(constb))

    ax1.plot(t,G(t,cost,risk,consta,constb), label="Case"+str(i+1))


#ax1.set_title('sin')
#ax1.set_xlabel('t')
#ax1.set_ylabel('x')
#ax1.set_xlim(-np.pi, np.pi)
#ax1.grid(True)

sensitivity = [0.7, 0.7, 0.8, 0.8, 0.9, 0.9] #感度
specificity = [0.9, 0.99, 0.9, 0.99, 0.9, 0.99] #特異度
cost = 4 #治療コスト
risk = 3 #感染リスク
#価値分布の密度関数g(v)を定義
def dist(x):
    #return 4*x*(1-x)
    return x*(1-x)

for i in range(6):
    #比例定数を決定
    hh = 1/NN    # 微小区間の幅
    p = np.linspace(0, 1, NN+1)   # 積分区間を NN 等分する
    ff1 = sigmoid(p)*(1-sigmoid(p))*dist(p)
    S1 = hh*(np.sum(ff1)-ff1[0]/2-ff1[NN]/2)
    ff2 = sigmoid(p)*dist(p)
    S2 = hh*(np.sum(ff2)-ff2[0]/2-ff2[NN]/2)
    ff3 = (1-sigmoid(p))*dist(p)
    S3 = hh*(np.sum(ff3)-ff3[0]/2-ff3[NN]/2)

    ff22 = sigmoid(p)*sigmoid(p)*dist(p)
    S22 = hh*(np.sum(ff22)-ff22[0]/2-ff22[NN]/2)

    consta = (1-sensitivity[i])*S2/S1
    constb = (1-specificity[i])*S3/S1

    print("a:"+str(i+7)+":"+str(consta))
    print("b:"+str(i+7)+":"+str(constb))
    
    gg = G(p,cost,risk,consta,constb)*dist(p)
    GS = hh*(np.sum(gg)-gg[0]/2-gg[NN]/2)

    ax2.plot(t,G(t,cost,risk,consta,constb), label="Case"+str(i+1)+"'")




ax1.legend()
ax2.legend()
plt.show()
