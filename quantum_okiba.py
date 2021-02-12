import numpy as np
import qutip as qt
import scipy
import matplotlib.pyplot as plt
from scipy import constants

pi = np.pi
e = constants.e # [C]
h = constants.h # [m^2 kg/s]
hbar = constants.hbar

def EC_to_C(EC):
    return ((e**2)/(2*h*EC*10**9))/(10**(-15))

def ket(Nq, i):
    return qt.basis(Nq, i)

def c(Nq):
    cc = 0
    for i in range(Nq-1):
        cc = cc + np.sqrt(i+1) * ( ket(Nq, i) * ket(Nq, i+1).dag() )
    return cc

c1=c(3)
c2=c(3)
intterm=(qt.tensor(c1, c2.dag()) + qt.tensor(c1.dag(), c2))

def hamiltonian(Ec, Ej, N, ng):
    
    m = np.diag(4 * Ec * (np.arange(-N,N+1)-ng)**2) + 0.5 * Ej * (np.diag(-np.ones(2*N), 1) + np.diag(-np.ones(2*N), -1))
    return qt.Qobj(m)

def Energy_diagram(Ec,Ej,N,steps,lev): #第lev準位までみる lev<N
    ng_list = np.linspace(-1,1,steps)
    Eene_map = np.zeros((lev,len(ng_list)))

    for i in range(len(ng_list)):
        ng = ng_list[i]
        H = hamiltonian(Ec,Ej,N,ng)
        Eene = sorted(H.eigenenergies())
        
        for j in range(lev):
            Eene_map[j][i] = Eene[j+1]-Eene[0]
        
    return(Eene_map)

def ene_level(Ec, Ej, N):
    ng = 0
    enes = hamiltonian(Ec, Ej, N, ng).eigenenergies()
    return [enes[i]-enes[0] for i in range(len(enes))]

def wq(Ec,Ej,N):
    wqs=ene_level(Ec,Ej,N)
    return(wqs[1],wqs[2]-2*wqs[1]) #freq of Q,anharmonicity

def Energyplot_for_2Q(Ec1,Ej1,Ec2,Ej2,path):
    ED1 = Energy_diagram(Ec1,Ej1,10,100,3)
    ED2 = Energy_diagram(Ec2,Ej2,10,100,3)
    ng_list = np.linspace(-1.5,1.5,100)
    plt.figure(figsize=(10, 5))
    plt.subplot(1,2,1)
    plt.plot(ng_list,np.zeros(len(ng_list)))
    plt.plot(ng_list,ED1[0])
    plt.plot(ng_list,ED1[1])
    plt.plot(ng_list,ED1[2])
    plt.title('Qubit1')
    plt.xlabel('ng')
    plt.ylabel('frequency[GHz]')
    plt.text(0,10,'EC={},EJ={}'.format(Ec1,Ej1))
    wq1,anh1=wq(Ec1,Ej1,10)
    plt.text(0,15,'f={:.1f},Δ={:.3f}'.format(wq1,anh1))
    plt.subplot(1,2,2) 
    plt.plot(ng_list,np.zeros(len(ng_list)))
    plt.plot(ng_list,ED2[0])
    plt.plot(ng_list,ED2[1])
    plt.plot(ng_list,ED2[2])
    plt.title('Qubit2')
    plt.xlabel('ng')
    plt.ylabel('frequency[GHz]')
    plt.text(0,10,'EC={},EJ={}'.format(Ec2,Ej2))
    wq2,anh2=wq(Ec2,Ej2,10)
    plt.text(0,15,'f={:.1f},Δ={:.3f}'.format(wq2,anh2))
    plt.savefig(path)

def Hq(Nq, qFreq, qAnhar):
    Hqs = 0
    eigenFreq_list = [0,qFreq,2*qFreq-qAnhar]
    for i in range(Nq):
        Hqs = Hqs + eigenFreq_list[i] * ( ket(Nq, i) * ket(Nq, i).dag() )
    return Hqs

ini_coeff = [0,1e-9,0,1e-9,1,0,0,0,0] # 11
ini_state = ini_coeff[0]*qt.tensor(ket(3,0), ket(3,0)) \
        + ini_coeff[1]*qt.tensor(ket(3,0), ket(3,1)) \
        + ini_coeff[2]*qt.tensor(ket(3,0), ket(3,2)) \
        + ini_coeff[3]*qt.tensor(ket(3,1), ket(3,0)) \
        + ini_coeff[4]*qt.tensor(ket(3,1), ket(3,1)) \
        + ini_coeff[5]*qt.tensor(ket(3,1), ket(3,2)) \
        + ini_coeff[6]*qt.tensor(ket(3,2), ket(3,0)) \
        + ini_coeff[7]*qt.tensor(ket(3,2), ket(3,1)) \
        + ini_coeff[8]*qt.tensor(ket(3,2), ket(3,2))

norm = np.dot(ini_coeff, ini_coeff)

def PhaseChange(state_list):
    
    final00 = [0] * len(state_list)
    final01 = [0] * len(state_list)
    final02 = [0] * len(state_list)
    final10 = [0] * len(state_list)
    final11 = [0] * len(state_list)
    final12 = [0] * len(state_list)
    final20 = [0] * len(state_list)
    final21 = [0] * len(state_list)
    final22 = [0] * len(state_list)
    #finalAdia = [0] * len(state_list)

    pop01 = [0] * len(state_list) # population of the state |01>
    pop10 = [0] * len(state_list) # population of the state |10>
    pop11 = [0] * len(state_list)# population of the state |11>
    pop02 = [0] * len(state_list)# population of the state |02>

    phase00 = [0] * len(state_list)
    phase10 = [0] * len(state_list)
    phase01 = [0] * len(state_list)
    phase11 = [0] * len(state_list)
    #phaseAdia = [0] * len(state_list)
    phaseDiff = [0] * len(state_list)
        
    for i in range(len(state_list)):
            
        final00[i] = state_list[i][:][0]
        final01[i] = state_list[i][:][1]
        final02[i] = state_list[i][:][2]
        final10[i] = state_list[i][:][3]
        final11[i] = state_list[i][:][4]
        final12[i] = state_list[i][:][5]
        final20[i] = state_list[i][:][6]
        final21[i] = state_list[i][:][7]
        final22[i] = state_list[i][:][8]
        #finalAdia[i] = state_list[i][:][2] + state_list[i][:][4] # eigenstate along the adiabatic line
        
        pop01[i] = norm * np.absolute(final01[i])**2 # the population is square of the magnitude.
        pop10[i] = norm * np.absolute(final10[i])**2
        pop11[i] = norm * np.absolute(final11[i])**2
        pop02[i] = norm * np.absolute(final02[i])**2
        
        phase00[i] = np.angle(final00[i]) / pi
        phase01[i] = np.angle(final01[i]) / pi
        phase10[i] = np.angle(final10[i]) / pi
        phase11[i] = np.angle(final11[i]) / pi
        #phaseAdia[i] = np.angle(finalAdia[i]) / pi
        #phaseDiff[i] = phaseAdia[i] - phase10[i] - phase01[i]
        
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
    
    return phaseDiff
    #return(phaseDiff,pop11,pop02)

"""
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
"""

#ketベクトル転写用
def tensor_to_flat(tensorproduct):
    
    d=tensorproduct.shape[0]
    onedvector=0
    for i in range(d):
        onedvector=onedvector+tensorproduct[i][0][0]*ket(d,i)
    
    return onedvector