import csv
import os
import math
import numpy as np
import qutip as qt
import scipy
from scipy import constants
from scipy import interpolate
from scipy import integrate
import sympy as sym

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import sys
import quantum_okiba as qo
from tqdm import tqdm

#Create classification folder
iDir = os.path.abspath(os.path.dirname(__file__))
ver = '1'

pi = np.pi
e = 1.6021766208*10**(-19) # [C]
h = 6.62607004*10**(-34) # [m^2 kg/s]
hbar=h/(2*pi)
opts = qt.solver.Options(nsteps=100000)

## params design [frequency]
Nq=3
EJ1 = 17
EJ2 = 22
EC1 = 0.27
EC2 = 0.27
EJEC1 = EJ1/EC1
EJEC2 = EJ2/EC2
C1_total = qo.EC_to_C(EC1)
C2_total = qo.EC_to_C(EC2)

# Frequency
fq1,anh1 = qo.wq(EC1,EJ1,10) #5.21 when EJ1=14, EC1=0.27
fq2,anh2 = qo.wq(EC2,EJ2,10) #5.35 when EJ1=18, EC2=0.27

print(fq1,fq2)
wq1 = fq1*(2*pi)
wq2 = fq2*(2*pi)
qAnhar1 = abs(anh1)*(2*pi)
qAnhar2 = abs(anh2)*(2*pi)
qFreq20 = wq1+qAnhar2

#Hamiltonian term
def ket(Nq, i):
    return qt.basis(Nq, i)

def c(Nq):
    cc = 0
    for i in range(Nq-1):
        cc = cc + np.sqrt(i+1) * ( ket(Nq, i) * ket(Nq, i+1).dag() )
    return cc

c1=c(Nq)
c2=c(Nq)
x=qt.sigmax()
y=qt.sigmay()
z=qt.sigmaz()

Iq1 = qt.qeye(Nq)
Iq2 = qt.qeye(Nq)

ini_coeff = [0,1e-9,0,1e-9,1,0,0,0,0] # 11
ini_state = ini_coeff[0]*qt.tensor(ket(Nq,0), ket(Nq,0)) \
            + ini_coeff[1]*qt.tensor(ket(Nq,0), ket(Nq,1)) \
            + ini_coeff[2]*qt.tensor(ket(Nq,0), ket(Nq,2)) \
            + ini_coeff[3]*qt.tensor(ket(Nq,1), ket(Nq,0)) \
            + ini_coeff[4]*qt.tensor(ket(Nq,1), ket(Nq,1)) \
            + ini_coeff[5]*qt.tensor(ket(Nq,1), ket(Nq,2)) \
            + ini_coeff[6]*qt.tensor(ket(Nq,2), ket(Nq,0)) \
            + ini_coeff[7]*qt.tensor(ket(Nq,2), ket(Nq,1)) \
            + ini_coeff[8]*qt.tensor(ket(Nq,2), ket(Nq,2))

q1_lab = qo.Hq(Nq, wq1, qAnhar1)
Hq1_lab = qt.tensor(q1_lab, Iq2)

rot2 = qo.Hq(Nq, 0, qAnhar2)
q2Freqs = qt.qdiags(np.arange(0,Nq,1),0)
Hq2_t_ind = qt.tensor(Iq1, rot2) #Hq2_rot(constant term)
Hq2_t_dep = qt.tensor(Iq1, q2Freqs) #Hq2_rot(modulation term)

taus_list=np.linspace(10,210,201)
intterm= ( qt.tensor(c1, c2.dag()) + qt.tensor(c1.dag(), c2) )

the_f=0.88
coeff2=0.13

J0=0.015
#tg_standard=round((1/(np.sqrt(2)*J0)),1)
tg=48

delta = fq2-fq1-abs(anh2)
the_i=math.atan(2*np.sqrt(2)*J0/delta)

outdir=iDir+'/Nz_tg={},J0={:.1f}MHz,the_i={:.2f}'.format(tg,J0*1000,the_i)
if os.path.exists(outdir)==False:
    os.makedirs(outdir)

#xx,yy=np.meshgrid(lambdalist,theta_list)
Hint = J0*(2*pi)*intterm

Hqs=Hq1_lab + Hq2_t_ind +Hint

def MW_shaped(t,args):
    amp = args['mwamp']
    shape = args['shape'] 
    if int(t)>=len(shape):
        n=len(shape)-1
    else:
        n=int(t)
    return amp * shape[n]

coeff1=the_f*(pi/2)
lambdas=[the_i,(coeff1-the_i)/2,coeff2]

def co_optim(taug):

    tau_list=np.linspace(0,taug,int(tg))

    def theta_tau(t): #θを制御するスレピアン関数
        theta=lambdas[0]
        for i in range(1,len(lambdas)):
            theta=theta+lambdas[i]*(1-np.cos(2*pi*t*i/taug))
        return theta

    def sintheta_tau(t):
        theta=theta_tau(t)
        sintheta=np.sin(theta)
        return sintheta
        
    t_tau=[]
    for i in range(len(tau_list)):
        t_tau.append(integrate.quad(sintheta_tau,0,tau_list[i])[0])
    
    return(t_tau,theta_tau(tau_list),taug/t_tau[-1])

theta_list=co_optim(int(tg/2))[1] #静止座標系のθ(0)~θ(tg/2)
theta_half=co_optim(int(tg/2))[1] #静止座標系のθ(tg/2)~θ(tg)
theta_list=np.append(theta_list,theta_half)
t_list=np.linspace(0,tg,len(theta_list)) #theta_listの長さに合うように等分割した時間のリスト

f1=interpolate.interp1d(t_list,theta_list,fill_value="extrapolate") #θ(t) calculated ここまでは関数
y1=f1(t_list) #ここで配列になる
slepian_like = 2*np.sqrt(2)*J0*(2*pi)/np.tan(y1)+qFreq20

args = {'mwamp':1.0,'shape':slepian_like}
H_rot = [Hq1_lab + Hq2_t_ind + Hint, [Hq2_t_dep, MW_shaped]]
res = qt.sesolve(H_rot, ini_state, t_list, e_ops=[], args=args, options=opts, progress_bar=None)
q_state_list=res.states

def PhaseChange(state_list):
    
    final01 = [0] * len(state_list)
    final10 = [0] * len(state_list)
    final11 = [0] * len(state_list)

    phase10 = [0] * len(state_list)
    phase01 = [0] * len(state_list)
    phase11 = [0] * len(state_list)

    phaseDiff = [0] * len(state_list)
    
    for i in range(len(state_list)):
            
        final01[i] = state_list[i][:][1]
        final10[i] = state_list[i][:][3]
        final11[i] = state_list[i][:][4]
        
        phase01[i] = np.angle(final01[i]) / pi
        phase10[i] = np.angle(final10[i]) / pi
        phase11[i] = np.angle(final11[i]) / pi
        
        phaseDiff[i] = phase11[i] - phase10[i] - phase01[i]
        
        # phase ordering
        if i > 0 and phase10[i] - phase10[i-1] < -1:
            phase10[i] = phase10[i] + 2
        if i > 0 and phase10[i] - phase10[i-1] > 1:
            phase10[i] = phase10[i] - 2
            
        if i > 0 and phaseDiff[i] - phaseDiff[i-2] < -1:
            phaseDiff[i] = phaseDiff[i] + 2
        if i > 0 and phaseDiff[i] - phaseDiff[i-2] > 1:
            phaseDiff[i] = phaseDiff[i] - 2
    return(phaseDiff)

pD=PhaseChange(q_state_list)

plt.figure(figsize=(6.0,6.0))
gs=gridspec.GridSpec(2,1)
ax1=plt.subplot(gs[0,0])
ax2=plt.subplot(gs[1,0])
plt.title('J={:.3f}[GHz_2π], tg={:.1f}[ns].jpg'.format(J0,tg))

ax1.set_title('PhaseChange')
ax1.plot(t_list,pD,color='g',label='|11> Phase')
ax1.set_xlabel('t[ns]')
ax1.set_ylabel('Phase/pi')
ax1.legend()
ax1.grid(True)

ax2.set_title('Control')
ax2.plot(t_list,slepian_like,color='b',label='Waveform')
ax2.plot([0,tg],[qFreq20,qFreq20],linestyle='dashed',color='black')
ax2.set_xlabel('t[ns]')
ax2.set_ylabel('frequency')
ax2.legend()
ax2.grid(True)

plt.savefig(outdir+"/wave_and_phase_{:.2f}_{:.3f}.jpg".format(the_f,coeff2))
plt.close()
