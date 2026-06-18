import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh


def twomode_bose_hubbard_hamiltonian_vectorized(N, K, U):
    dim = N + 1
    n1 = np.arange(dim)
    n2 = N - n1
    diag = (U / 4.0) * (n1 - n2) ** 2
    H = np.diag(diag)
    k_idx = np.arange(1, dim)
    off_vals = -0.5 * K * np.sqrt(k_idx * (N - k_idx + 1))
    H += np.diag(off_vals, k=1) + np.diag(off_vals, k=-1)
    return H

def numerical_phases(V):
    for i in range(V.shape[1]):
        idx = np.argmax(np.abs(V[:, i]))
        phase = np.sign(V[idx, i])
        if phase != 0:
            V[:, i] *= phase
    return V

def Average_Work(H_i,H_f,state_i):
   
    W=np.dot(state_i,H_f.dot(state_i))-np.dot(state_i,H_i.dot(state_i))

    return W

def Amplitudes(H_i,H_f):
    E_i,V_i=eigh(H_i)
    E_f,V_f=eigh(H_f)
    V_i = numerical_phases(V_i)
    V_f = numerical_phases(V_f)
    return V_f.T @ V_i,E_i,E_f,V_i,V_f


def ClassicalBound(H_i,H_f,state_i):
    Lambda,E_i,E_f,V_i,V_f=Amplitudes(H_i,H_f)
    p_jk=(np.abs(Lambda)**2)*np.abs(V_i.T@state_i)**2
    p_k=np.abs(V_f.T @ state_i)**2
    dim=len(E_i)
    w_bound=0
    for j in range(dim):
        for k in range(dim):
            if E_i[j] >= E_f[k]:
                w= (E_i[j] - E_f[k]) * np.sqrt(p_jk[k, j] * p_k[k])
                w_bound += w
                
    return w_bound

#####################################################################################################################################################
K=1
U_i, U_f, N = 0.1, 0.0, 100

alpha = np.linspace(0, 1, 100)
delta_phi= np.linspace(0, 2 * np.pi, 100)

bound_matrix = np.zeros((len(alpha), len(delta_phi)))
work_matrix =np.zeros((len(alpha),len(delta_phi)))
H_i=twomode_bose_hubbard_hamiltonian_vectorized(N,K,U_i)
H_f=twomode_bose_hubbard_hamiltonian_vectorized(N,K,U_f)

Lambda,E_i,E_f,V_i,V_f=Amplitudes(H_i,H_f)
for i, a in enumerate(alpha):
    for j, phi in enumerate(delta_phi):
    
        state_i = a * V_i[:, 0] + np.sqrt(1 - a**2 + 1e-15) * np.exp(1j * phi) * V_i[:, 2]
        state_i /= np.linalg.norm(state_i)
        
        # Calcola il valore e assegnalo
        bound_matrix[i, j] = ClassicalBound(H_i,H_f,state_i)
        work_matrix[i, j] =Average_Work(H_i,H_f,state_i)

plt.imshow(bound_matrix+work_matrix, aspect='auto', origin='lower', 
           extent=[delta_phi.min(), delta_phi.max(), alpha.min(), alpha.max()])
plt.colorbar(label='Work Bound + Averege Work')
plt.xlabel(r'$\Delta \phi$')
plt.ylabel(r'$\alpha$')
plt.show()