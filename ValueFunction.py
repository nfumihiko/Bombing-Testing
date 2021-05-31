# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

h = 0.001 #刻み幅
t = np.arange(0,1,h) #h刻みで時間0~Tを分割
NN = 10**3    # 積分項の分割数

sensitivity = 0.7 #感度
specificity = 0.9 #特異度

gain = 7 #シグモイド関数内のゲイン
PLLR = np.log(sensitivity/(1-specificity)) #治療コスト
NLLR = -np.log((1-sensitivity)/specificity) #感染リスク

PLLR=0
NLLR=0

constD = 0.0 #PLLR
constE = 0.0

constM_posi = 1.0 #獲得情報
constM_nega = constM_posi


def sigmoid(x):
    return 1/(1+np.exp(-gain*(x-0.5)))
    
def sigmoid_dist(x):
    return gain*sigmoid(x)*(1-sigmoid(x))

def entropy(x):
    return -x*np.log(x)-(1-x)*np.log(1-x)

#価値分布の密度関数g(v)を定義
def dist(x):
    #return 4*x*(1-x)
    return x*(1-x)
    
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

consta = (1-sensitivity)*S2/S1
constb = (1-specificity)*S3/S1
print("S1:"+str(S1))
print("S2:"+str(S2))
print("S3:"+str(S3))
print("mu:"+str(S22/S2))

print("a:"+str(consta))
print("b:"+str(constb))


    
def f_posi_true(v):
    return sigmoid(v)*(1-consta*(1-sigmoid(v)))
def f_posi_false(v):
    return sigmoid(v)*constb*(1-sigmoid(v))
def f_nega_false(v):
    return (1-sigmoid(v))*consta*sigmoid(v)
def f_nega_true(v):
    return (1-sigmoid(v))*(1-constb*sigmoid(v))
    
def info_posi(v):
    return -constM_posi*(f_posi_true(v)/(f_posi_true(v)+f_posi_false(v))*np.log(f_posi_true(v)/(f_posi_true(v)+f_posi_false(v)))+f_posi_false(v)/(f_posi_true(v)+f_posi_false(v))*np.log(f_posi_false(v)/(f_posi_true(v)+f_posi_false(v))))

def info_nega(v):
    return -constM_nega*(f_nega_true(v)/(f_nega_true(v)+f_nega_false(v))*np.log(f_nega_true(v)/(f_nega_true(v)+f_nega_false(v)))+f_posi_false(v)/(f_nega_true(v)+f_nega_false(v))*np.log(f_nega_false(v)/(f_nega_true(v)+f_nega_false(v))))

def odd_posi(v):
    return np.log(f_posi_true(t)/f_posi_false(t))
    
def odd_nega(v):
    return np.log(f_nega_true(t)/f_nega_false(t))

def G(t):
    return PLLR*f_posi_true(t)+\
    (PLLR-constD)*f_posi_false(t)+\
    (NLLR-constE)*f_nega_false(t)+\
    NLLR*f_nega_true(t)+\
    info_posi(t)*(f_posi_true(t)+f_posi_false(t))+\
    info_nega(t)*(f_nega_true(t)+f_nega_false(t))

gg = G(p)*dist(p)
GS = hh*(np.sum(gg)-gg[0]/2-gg[NN]/2)

print("GS:"+str(GS))

# グラフで可視化
fig = plt.figure(figsize=(10,5))
plt.subplots_adjust(wspace=0.2, hspace=0.3)
ax1 = fig.add_subplot(231) # 左図
ax2 = fig.add_subplot(232) # 右図
ax3 = fig.add_subplot(233) # 左図
ax4 = fig.add_subplot(234) # 左図
ax5 = fig.add_subplot(235) # 左図
ax6 = fig.add_subplot(236) # 左図

ax1.plot(t,sigmoid(t), label="$p(v)$", color='red')
ax1.plot(t,1-sigmoid(t), label="$1-p(v)$", color='blue')
#ax1.plot(t,np.log(f_nega_true(t)/f_nega_false(t)), label="entropy")
#ax1.plot(t,(odd_posi(t)*f_posi_true(t)+(odd_posi(t)-constD)*f_posi_false(t))/(f_posi_true(t)+f_posi_false(t))+ info_posi(t), label="$F_+(v)$", color='red')
#ax1.plot(t,((odd_nega(t)-constE)*f_nega_false(t)+odd_nega(t)*f_nega_true(t))/(f_nega_true(t)+f_nega_false(t))+info_nega(t), label="$F_-(v)$", color='blue')

ax1.legend()
#ax1.set_title('sin')
#ax1.set_xlabel('t')
#ax1.set_ylabel('x')
#ax1.set_xlim(-np.pi, np.pi)
#ax1.grid(True)

ax2.plot(t,f_posi_true(t), label="$f^+_{true}$", color='red')
ax2.plot(t,f_posi_false(t), label="$f^+_{false}$", color='coral')
ax2.plot(t,f_nega_false(t), label="$f^-_{false}$", color='dodgerblue')
ax2.plot(t,f_nega_true(t), label="$f^-_{true}$", color='blue')

ax2.legend()

ax3.plot(t,(PLLR*f_posi_true(t)+(NLLR-constD)*f_posi_false(t))/(f_posi_true(t)+f_posi_false(t)), label="$F_+(v)- H_+(v)$", color='red')
ax3.plot(t,(NLLR*f_nega_true(t)+(NLLR-constE)*f_nega_false(t))/(f_nega_true(t)+f_nega_false(t)), label="$F_-(v)- H_-(v)$", color='blue')
ax3.legend()

#ax2.plot(t,(PLLR-NLLR)*f_nega_false(t)+PLLR*f_nega_true(t), label="-")
#ax2.plot(t,NLLR*f_posi_true(t), label="+")
#ax2.legend()
#ax2.set_title('cos')
#ax2.set_xlabel('t')
#ax2.set_ylabel('x')
#ax2.set_ylim(0, 1)
#ax2.grid(True)
ax4.plot(t,info_posi(t), label="$H_+(+)$", color='red')
ax4.plot(t,info_nega(t), label="$H_-(v)$", color='blue')
ax4.legend()

#ax4.plot(t,info_nega(t)*(f_nega_true(t)+f_nega_false(t)), label="-")
#ax4.plot(t,info_posi(t)*(f_posi_true(t)+f_posi_false(t)), label="+")
#ax4.legend()

#ax4.plot(t,((PLLR-NLLR)*f_nega_false(t)+PLLR*f_nega_true(t))/(f_posi_true(t)+f_posi_false(t))+ info_nega(t), label="f_- + kH-")
#ax4.plot(t,NLLR*f_posi_true(t)/(f_nega_true(t)+f_nega_false(t))+ info_posi(t), label="f_+ + kH+")
ax5.plot(t,(PLLR*f_posi_true(t)+(PLLR-constD)*f_posi_false(t))/(f_posi_true(t)+f_posi_false(t))+ info_posi(t), label="$F_+(v)$", color='red')
ax5.plot(t,((NLLR-constE)*f_nega_false(t)+NLLR*f_nega_true(t))/(f_nega_true(t)+f_nega_false(t))+info_nega(t), label="$F_-(v)$", color='blue')

ax5.legend()

#ax6.plot(t,(PLLR-NLLR)*f_nega_false(t)+PLLR*f_nega_true(t)+info_nega(t)*(f_nega_true(t)+f_nega_false(t)), label="-")
#ax6.plot(t,NLLR*f_posi_true(t)+ info_posi(t)*(f_posi_true(t)+f_posi_false(t)), label="+")
ax6.plot(t,G(t),label="$G(v)$", color='green')
#ax6.set_ylim(0, 1.5)
#ax4.plot(t,((PLLR-NLLR)*f_nega_false(t)+PLLR*f_nega_true(t)+ info_nega(t)+NLLR*f_posi_true(t)+ info_posi(t))/2, label="G")
#ax4.plot(t,(1-sigmoid(t))*f_negative(t)+ info_nega(t)+sigmoid(t)*f_positive(t)+ info_posi(t), label="G")
#ax4.plot(t,sigmoid(t)*(f_positive(t)-constM * np.log2(sigmoid(t))), label="G+")
#ax4.plot(t,(1-sigmoid(t))*(f_negative(t)-constM * np.log2(1-sigmoid(t))), label="G-")
ax6.legend()
fig.suptitle("$\mu_+=$"+str(sensitivity)+", "+"$\mu_-=$"+str(specificity)+", "+"$\log(C_+)=$"+str(constD)+", "+"$\log(C_-)=$"+str(constE)+",")
plt.show()


