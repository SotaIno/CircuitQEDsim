import numpy as np
import matplotlib.pyplot as plt

from CZwave3 import CZpulse

#CZsystem=CZpulse()
#sq=CZpulse.squarepulse()

Tg = 100
t_list = np.linspace(0,Tg,Tg+1)
t_ramp=1
shape = []
t_ini=10
fq2=0
qFreq20=1

for t in t_list:
    if t < t_ini:
        a = fq2
    elif t <= t_ini + t_ramp:
        a = fq2 - (fq2-qFreq20)/t_ramp*(t-t_ini)
    elif t > Tg - (t_ini):
        a = fq2
    elif t >= Tg - (t_ramp+t_ini):
        a = fq2 - (fq2-qFreq20)  / t_ramp*(Tg-(t+t_ini))
    else:
        a = qFreq20
    shape.append(round(a,5))

plt.plot(t_list,shape)
plt.show()

FIRfilter = np.load('C:/Sota_Ino/Tsai_Lab/fir_coe.npy')

# ベタ書きによるFIRフィルタ
d = np.zeros(len(shape))

for i in range(len(shape)):
    d[i] = 0
    for j in range(len(FIRfilter)):
        if(i-j)>=0:
            d[i] += FIRfilter[j]*shape[i-j]

plt.plot(t_list,d,color='b')
plt.show()