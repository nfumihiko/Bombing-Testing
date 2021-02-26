# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

nn = 5 #初期弾数
T = 10 # 終点の時間
n = 1000 #時間の分割数
h = T / n #刻み幅
t = np.arange(0,T,h) #h刻みで時間0~Tを分割

eval = 1.0
NN = 10**3    # 積分項の分割数

probability = 0.6 #1発で撃墜できる確率
prob_lambda = 1 #会敵確率


param_posi = [0.3, 0.2, 10, 10, 2.0, 7.796, 0.2892]
param_nega = [0.12, 0.25, 5, 5, 1.0, 4.393146, 0.641468]

def IoSfunc(v,P):
    return P[0]*np.tanh(P[2]*v-P[4])-P[1]*np.tanh(P[3]*v-P[5])+P[6]
def IoSfunc_diff(v,P):
    return P[0]*P[2]/np.cosh(P[2]*v-P[4])**2-P[1]*P[3]/np.cosh(P[3]*v-P[5])**2
#ここが重要なところ
def f_positive(v):
    return IoSfunc(v, param_posi)
def f_negative(v):
    return IoSfunc(v, param_nega)
def f_posi_diff(v):
    return IoSfunc_diff(v, param_posi)
def f_nega_diff(v):
    return IoSfunc_diff(v, param_nega)

#F(t,n)を定義
#境界条件F(0,n)=0
F = [[0 for i in range(n)]  for j in range(nn+1)]
for j in range(nn+1):
 #   F[j][0] = (1-j/nn)*100
    F[j][0] = 0.0 + 0.56*j
 #   F[j][0] = 0.0
for i in range(n):
 #   F[0][i] = (1-i/(n-1))*100
    F[0][i] = 0.0


C = [[[0 for i in range(n)]  for r in range(2)] for j in range(nn+1)]
#Cの初期条件
for j in range(nn+1):
    C[j][0][0] = 0.0
    C[j][1][0] = 0.96
#    C[j][1] = 0.8


#価値分布の密度関数g(v)を定義
def dist(x):
    for i in range(NN+1):
        if x[i]>0 and x[i]<eval:
  #          x[i]=1/eval
            x[i]=2/eval-2*x[i]/eval**2
        else:
            x[i]=0
    return x
    
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

    probability1 = probability
#    probability1 = i/n*(1-i/n)+0.4
    
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
        ff = (probability1*f_positive(p)+(1-probability1)*f_negative(p)+F[j-1][i])*dist(p)
        S = hh*(np.sum(ff)-ff[0]/2-ff[NN]/2)

    sum2=sum2+S

    
    sum3 = (F[j-1][i+1]-F[j-1][i])/2/h
#    if sum3>2.0:
#        sum3=2.0
#        print(str(i)+":"+str(j))



    
    RHStotal=0
    if R==0:
        RHStotal =prob_lambda*(-(probability1*f_positive(CC0)+(1-probability1)*f_negative(CC0)+F[j-1][i])* (1-sum10-sum11)+sum2-sum3)/(probability1*f_posi_diff(CC0)+(1-probability1)*f_nega_diff(CC0))
    else:
        RHStotal =prob_lambda*(-(probability1*f_positive(CC1)+(1-probability1)*f_negative(CC1)+F[j-1][i])* (1-sum10-sum11)+sum2-sum3)/(probability1*f_posi_diff(CC1)+(1-probability1)*f_nega_diff(CC1))

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

#    probability = i/n*(1-i/n)+0.4
#    probability = (1-np.tanh(i/n-0.5)**2)/2.0+1/2.0
for j in range(1,nn+1):
    for i in range(n-1):
#        probability1 = i/n*(1-i/n)+0.4
        probability1 = probability
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
 #       F[j][i+1] = (probability*f_positive(C[j][0][i+1])+(1-probability)*f_negative(C[j][0][i+1])+probability*f_positive(C[j][1][i+1])+(1-probability)*f_negative(C[j][1][i+1]))/2+F[j-1][i+1]
        F[j][i+1] = (probability1*f_positive(C[j][0][i+1])+(1-probability1)*f_negative(C[j][0][i+1])+probability1*f_positive(C[j][1][i+1])+(1-probability1)*f_negative(C[j][1][i+1]))/2+F[j-1][i+1]


 
 
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
ax2.legend()
#ax2.set_title('cos')
#ax2.set_xlabel('t')
#ax2.set_ylabel('x')
ax2.set_ylim(0, 1)
#ax2.grid(True)
plt.show()
