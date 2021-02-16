import os
import numpy as np
import systemConst
import matplotlib.pyplot as plt
import japanize_matplotlib
from systemConst import transmon,Tunabletransmon,QQ

iDir=os.path.abspath(os.path.dirname(__file__))

pi=np.pi
Nq=3
EJ1 = 17
EJ2 = 22
EC1 = 0.27
EC2 = 0.27
g=0.015*2*pi

Xmon1=transmon(EC=EC1,EJ=EJ1,N=10)
#print(Xmon1.anh)
#print(Xmon1.calcChargeQubitLevels(EC=EC1,EJ=EJ1,N=10))

Xmon2=transmon(EC=EC2,EJ=EJ2,N=10)
#print(Xmon2.anh)
#print(Xmon2.calcChargeQubitLevels(EC=EC2,EJ=EJ2,N=10))
nglist,zerolevel=Xmon1.calcNthlevelenergy(EC1,EJ1,10,0)
firstlevel=Xmon1.calcNthlevelenergy(EC1,EJ1,10,1)[1]
secondlevel=Xmon1.calcNthlevelenergy(EC1,EJ1,10,2)[1]
thirdlevel=Xmon1.calcNthlevelenergy(EC1,EJ1,10,3)[1]
plt.rcParams["font.size"] = 14
fig,ax=plt.subplots()
ax.plot(nglist,zerolevel,label='zero level')
ax.plot(nglist,firstlevel,label='1st level')
ax.plot(nglist,secondlevel,label='2nd level')
#ax.plot(nglist,thirdlevel,label='3rd level')
ax.set_xlabel('ng')
ax.legend(fontsize=15)
ax.set_xticks([-1, -0.5, 0, 0.5, 1])
ax.set_title('Enegydiagram(xaxis=ng)')
ax.set_ylabel('Frequency[GHz]')
plt.savefig(iDir+'/levels.png')
"""
QQsystem1=QQ(Xmon1,Xmon2,g)
#QQsystem1.plotDressedEnergyLevels()
staZZ=QQsystem1.calcStaticZZ(None)
print(staZZ)
"""
tXmon1=Tunabletransmon(EC=EC1,EJmax=EJ1,N=10)
#print(tXmon1.anh)
#print(tXmon1.calcChargeQubitLevels(EC=EC1,EJmax=EJ1,N=10,Phi=0))

tXmon2=Tunabletransmon(EC=EC2,EJmax=EJ2,N=10)
#print(tXmon2.anh)
#print(tXmon2.calcChargeQubitLevels(EC=EC2,EJmax=EJ2,N=10,Phi=0))

philist,zerolevel=tXmon1.calcNthlevelenergy(EC1,EJ1,10,0)
firstlevel=tXmon1.calcNthlevelenergy(EC1,EJ1,10,1)[1]
secondlevel=tXmon1.calcNthlevelenergy(EC1,EJ1,10,2)[1]
thirdlevel=tXmon1.calcNthlevelenergy(EC1,EJ1,10,3)[1]
fig,ax=plt.subplots()
ax.plot(philist,zerolevel,linewidth=5)
ax.plot(philist,firstlevel,linewidth=5)
ax.plot(philist,secondlevel,linewidth=5)
#ax.plot(philist,thirdlevel,label='3rd level')
ax.set_xlabel('$\pi \Phi/\Phi_0$',fontsize=18)
ax.legend(fontsize=18,loc='upper right')
ax.set_xlim([-0.3,0.3])
#ax.set_xticks([-1, -0.5, 0, 0.5, 1])
#ax.set_title('量子ビットの外部磁場応答',fontsize=18)
#ax.set_ylabel('周波数[GHz]',fontsize=18)
plt.savefig('C:/Users/Sota/Master_thesis/img/levels_tune_happyoyo.png')

philist,zerolevel=tXmon2.calcNthlevelenergy(EC2,EJ2,10,0)
firstlevel=tXmon2.calcNthlevelenergy(EC2,EJ2,10,1)[1]
secondlevel=tXmon2.calcNthlevelenergy(EC2,EJ2,10,2)[1]
thirdlevel=tXmon1.calcNthlevelenergy(EC1,EJ1,10,3)[1]
fig,ax=plt.subplots()
ax.plot(philist,zerolevel,linewidth=5)
ax.plot(philist,firstlevel,linewidth=5)
ax.plot(philist,secondlevel,linewidth=5)
#ax.plot(philist,thirdlevel,label='3rd level')
ax.set_xlabel('$\pi \Phi/\Phi_0$',fontsize=18)
ax.legend(fontsize=18,loc='upper right')
ax.set_xlim([-0.3,0.3])
#ax.set_xticks([-1, -0.5, 0, 0.5, 1])
#ax.set_title('量子ビットの外部磁場応答',fontsize=18)
#ax.set_ylabel('周波数[GHz]',fontsize=18)
plt.savefig('C:/Users/Sota/Master_thesis/img/levels_tune_happyoyo2.png')