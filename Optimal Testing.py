# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

nn = 25 #初期弾数
T = 100 # 終点の時間
n = 1000 #時間の分割数
h = T / n #刻み幅
t = np.arange(0,T,h) #h刻みで時間0~Tを分割

eval = 1.0
NN = 10**3    # 積分項の分割数

prob_lambda = 0.6 #会敵確率

sensitivity = 0.7 #感度
specificity = 0.9 #特異度

gain = 10 #シグモイド関数内のゲイン
cost = 0.5 #治療コスト
risk = 1.0 #感染リスク
constE=2.0 #情報量寄与率


#価値分布の密度関数g(v)を定義
def dist(x):
    for i in range(NN+1):
        if x[i]>0 and x[i]<eval:
  #          x[i]=1/eval
            x[i]=2/eval-2*x[i]/eval**2
        else:
            x[i]=0
    return x

def sigmoid(x):
    return 1/(1+np.exp(-gain*(x-0.5)))
    
def sigmoid_dist(x):
    return gain*sigmoid(x)*(1-sigmoid(x))

def entropy(x):
    return -x*np.log2(x)-(1-x)*np.log2(1-x)

#比例定数を決定
hh = 1/NN    # 微小区間の幅
p = np.linspace(0, 1, NN+1)   # 積分区間を NN 等分する
ff = sigmoid(p)*(1-sigmoid(p))*dist(p)
S = hh*(np.sum(ff)-ff[0]/2-ff[NN]/2)
consta = (1-sensitivity)/S
constb = (1-specificity)/S
print(consta)
print(constb)

def f_positive(v):
    return (1-consta*(1-sigmoid(v)))*risk + constE*entropy(sigmoid(v))
def f_negative(v):
    return consta*sigmoid(v)*(cost-risk)+(1-constb*sigmoid(v))*cost + constE*entropy(sigmoid(v))
def f_posi_diff(v):
    return consta*risk*sigmoid_dist(v) + constE*np.log2((1-sigmoid(v))/sigmoid(v))*sigmoid_dist(v)
def f_nega_diff(v):
    return (consta*(cost-risk)-constb*cost)*sigmoid_dist(v) + constE*np.log2((1-sigmoid(v))/sigmoid(v))*sigmoid_dist(v)

def ExpectValue(v):
    return sigmoid(v)*f_positive(v)+(1-sigmoid(v))*f_negative(v)
    
#F(t,n)を定義
#境界条件F(0,n)=0
F = [[0 for i in range(n)]  for j in range(nn+1)]
for j in range(nn+1):
 #   F[j][0] = (1-j/nn)*100
    F[j][0] = 0.0 + ExpectValue(0)*j
 #   F[j][0] = 0.0
for i in range(n):
 #   F[0][i] = (1-i/(n-1))*100
    F[0][i] = 0.0


C = [[[0 for i in range(n)]  for r in range(2)] for j in range(nn+1)]
#Cの初期条件
for j in range(nn+1):
    C[j][0][0] = 0.0
    C[j][1][0] = 1.0

#微分方程式の右辺
def RHS(R,CC0,CC1,j,i):
    xmin=0
    xmax=eval
    sum10=0
    sum11=0
    sum2=0
    sum3=0
        
    S=0
    xmin = 0
    xmax = CC0
  
    if xmax < 0:
        xmax=0
    if xmax >eval:
        xmax=eval
                
    if xmax==0:
        S=0
    else:
        hh = (xmax - xmin)/NN    # 微小区間の幅
        p = np.linspace( xmin, xmax, NN+1)   # 積分区間を NN 等分する
        ff = dist(p)
        S = hh*(np.sum(ff)-ff[0]/2-ff[NN]/2)
    sum10=sum10+S

      
    S=0
    xmin = CC1
    xmax = eval

    if xmin <0:
        xmin=0
    if xmin >eval:
        xmin=eval
        
    if xmin==eval:
        S=0
    else:
        hh = (xmax - xmin)/NN    # 微小区間の幅
        p = np.linspace( xmin, xmax, NN+1)   # 積分区間を NN 等分する
        ff = dist(p)
        S = hh*(np.sum(ff)-ff[0]/2-ff[NN]/2)
    sum11=sum11+S
        
        
    S=0
    xmin = CC0
    xmax = CC1

    if xmin <0:
        xmin=0
    if xmin >eval:
        xmin=eval
    if xmax < 0:
        xmax=0
    if xmax >eval:
        xmax=eval
        
    if xmin==eval:
        S=0
    else:
        hh = (xmax - xmin)/NN    # 微小区間の幅
        p = np.linspace( xmin, xmax, NN+1)   # 積分区間を NN 等分する
        ff = (ExpectValue(p)+F[j-1][i])*dist(p)
        S = hh*(np.sum(ff)-ff[0]/2-ff[NN]/2)

    sum2=sum2+S

    sum3 = (F[j-1][i+1]-F[j-1][i])/2/h
    
    RHStotal=0
    if R==0:
        RHStotal =prob_lambda*(-(ExpectValue(CC0)+F[j-1][i])* (1-sum10-sum11)+sum2-sum3)/(sigmoid(CC0)*f_posi_diff(CC0)+(1-sigmoid(CC0))*f_nega_diff(CC0)+sigmoid_dist(CC0)*(f_positive(CC0)-f_negative(CC0)))
    else:
        RHStotal =prob_lambda*(-(ExpectValue(CC1)+F[j-1][i])* (1-sum10-sum11)+sum2-sum3)/(sigmoid(CC1)*f_posi_diff(CC1)+(1-sigmoid(CC1))*f_nega_diff(CC1)+sigmoid_dist(CC1)*(f_positive(CC0)-f_negative(CC0)))
    return RHStotal
 
 
 
 
#ルンゲクッタ法の中身
def f(k,R,j,i,CC0,CC1,t,KK0,KK1):
    if k==1:
        return RHS(R,CC0,CC1,j,i)
    elif k==2:
        CC0=CC0+KK0/2
        CC1=CC1+KK1/2
        y = RHS(R,CC0,CC1,j,i)
        CC0=CC0-KK0/2
        CC1=CC1-KK1/2
        return y
    elif k==3:
        CC0=CC0+KK0/2
        CC1=CC1+KK1/2
        y = RHS(R,CC0,CC1,j,i)
        CC0=CC0-KK0/2
        CC1=CC1-KK1/2
        return y
    else:
        CC0=CC0+KK0
        CC1=CC1+KK1
        y = RHS(R,CC0,CC1,j,i)
        CC0=CC0-KK0
        CC1=CC1-KK1
        return y
    
# 方程式を解くための反復計算（ルンゲクッタ法）

for j in range(1,nn+1):
    for i in range(n-1):
        if(ExpectValue(C[j][0][i])<ExpectValue(1.0)):
            k_10 = h * f(1,0,j,i,C[j][0][i],1.0,t[i],0,0)
            k_11 = h * f(1,1,j,i,C[j][0][i],1.0,t[i],0,0)
            k_20 = h * f(2,0,j,i,C[j][0][i],1.0,t[i] + h/2 ,k_10,k_11)
            k_21 = h * f(2,1,j,i,C[j][0][i],1.0,t[i] + h/2 ,k_10,k_11)
            k_30 = h * f(3,0,j,i,C[j][0][i],1.0,t[i] + h/2 ,k_20,k_21)
            k_31 = h * f(3,1,j,i,C[j][0][i],1.0,t[i] + h/2 ,k_20,k_21)
            k_40 = h * f(4,0,j,i,C[j][0][i],1.0,t[i] + h ,k_30,k_31)
            k_41 = h * f(4,1,j,i,C[j][0][i],1.0,t[i] + h ,k_30,k_31)
            C[j][0][i+1] = C[j][0][i] + 1/6 * (k_10 + 2*k_20 + 2*k_30 + k_40)
            C[j][1][i+1] = 1.0

            F[j][i+1] = ExpectValue(C[j][0][i+1])+F[j-1][i+1]
        else:
            k_10 = h * f(1,0,j,i,C[j][0][i],C[j][1][i],t[i],0,0)
            k_11 = h * f(1,1,j,i,C[j][0][i],C[j][1][i],t[i],0,0)
            k_20 = h * f(2,0,j,i,C[j][0][i],C[j][1][i],t[i] + h/2 ,k_10,k_11)
            k_21 = h * f(2,1,j,i,C[j][0][i],C[j][1][i],t[i] + h/2 ,k_10,k_11)
            k_30 = h * f(3,0,j,i,C[j][0][i],C[j][1][i],t[i] + h/2 ,k_20,k_21)
            k_31 = h * f(3,1,j,i,C[j][0][i],C[j][1][i],t[i] + h/2 ,k_20,k_21)
            k_40 = h * f(4,0,j,i,C[j][0][i],C[j][1][i],t[i] + h ,k_30,k_31)
            k_41 = h * f(4,1,j,i,C[j][0][i],C[j][1][i],t[i] + h ,k_30,k_31)
            C[j][0][i+1] = C[j][0][i] + 1/6 * (k_10 + 2*k_20 + 2*k_30 + k_40)
            C[j][1][i+1] = C[j][1][i] + 1/6 * (k_11 + 2*k_21 + 2*k_31 + k_41)

            F[j][i+1] = ExpectValue(C[j][0][i+1])+F[j-1][i+1]

 
 
# グラフで可視化
fig = plt.figure(figsize=(10,5))
ax1 = fig.add_subplot(121) # 左図
ax2 = fig.add_subplot(122) # 右図

for j in range(1,nn+1):
    ax1.plot(t,F[j], label="F"+str(j),color=[j/nn,0.0,1.0-j/nn])
ax1.legend()
#ax1.set_title('sin')
#ax1.set_xlabel('t')
#ax1.set_ylabel('x')
#ax1.set_xlim(-np.pi, np.pi)
#ax1.grid(True)
for j in range(1,nn+1):
    for r in range(2):
        ax2.plot(t,C[j][r], label="C"+str(j)+","+str(r), color=[j/nn,0.0,1.0-j/nn])
#ax2.legend()
#ax2.set_title('cos')
#ax2.set_xlabel('t')
#ax2.set_ylabel('x')
ax2.set_ylim(0, 1)
#ax2.grid(True)
plt.show()
