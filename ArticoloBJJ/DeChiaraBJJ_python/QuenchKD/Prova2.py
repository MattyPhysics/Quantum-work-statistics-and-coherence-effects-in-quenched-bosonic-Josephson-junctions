import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt

# =============================================================================
# 1. HAMILTONIANA E SETUP
# =============================================================================
def get_BH_hamiltonian(N, K, U):
    dim = N + 1
    H = np.zeros((dim, dim))
    diag = (U / 4.0) * (np.arange(dim) - (N - np.arange(dim)))**2
    np.fill_diagonal(H, diag)
    off_diag = -0.5 * K * np.sqrt(np.arange(1, dim) * (N - np.arange(1, dim) + 1))
    np.fill_diagonal(H[1:], off_diag)
    np.fill_diagonal(H[:, 1:], off_diag)
    return H

def fix_phases(V):
    for i in range(V.shape[1]):
        V[:, i] *= np.sign(V[np.argmax(np.abs(V[:, i])), i])
    return V

# Parametri
N, K = 100, 1.0
U_i, U_f = 0.1, 0.0
H_i = get_BH_hamiltonian(N, K, U_i)
H_f = get_BH_hamiltonian(N, K, U_f)

E_i, V_i = eigh(H_i)
E_f, V_f = eigh(H_f)
V_i, V_f = fix_phases(V_i), fix_phases(V_f)
S = V_f.T @ V_i # Matrice di sovrapposizione

# =============================================================================
# 2. CALCOLO LAVORO E BOUND
# =============================================================================
def get_work_and_bound(c0, phi):
    c2 = np.sqrt(1 - c0**2)
    # Stato iniziale in base agli autostati di H_i
    psi_i = c0 * V_i[:, 0] + c2 * np.exp(1j * phi) * V_i[:, 2]
    
    # Lavoro medio (Definizione esatta: <psi|Hf|psi> - <psi|Hi|psi>)
    work_avg = np.real(np.vdot(psi_i, H_f @ psi_i) - np.vdot(psi_i, H_i @ psi_i))
    
    # Bound Classico (TPM): W_cl = sum_{m,n} P(m|n)P(n) * (E_i[n] - E_f[m])
    # dove E_i[n] >= E_f[m]
    p_n = np.array([c0**2, 0, c2**2]) # Popolazioni iniziali (n=0, 1, 2)
    bound = 0.0
    for n_idx, n in enumerate([0, 2]):
        for m in range(N + 1):
            w_mn = E_i[n] - E_f[m]
            if w_mn > 0:
                p_m_n = S[m, n]**2
                bound += w_mn * p_n[n_idx] * p_m_n
                
    return work_avg, bound

# =============================================================================
# 3. ESECUZIONE E PLOT
# =============================================================================
phases = np.linspace(0, 2*np.pi, 100)
w_vals, b_vals = [], []

for p in phases:
    w, b = get_work_and_bound(0.7, p)
    w_vals.append(w)
    b_vals.append(b)

plt.figure(figsize=(9, 5))
plt.plot(phases/np.pi, -np.array(w_vals), 'b-', label='Lavoro Estratto (-W)')
plt.plot(phases/np.pi, b_vals, 'r--', label='Classical Bound (TPM)')
plt.fill_between(phases/np.pi, -np.array(w_vals), b_vals, 
                 where=(-np.array(w_vals) > b_vals), color='green', alpha=0.2, label='Quantum Advantage')

plt.xlabel(r'Phase $\phi$ [$\pi$]')
plt.ylabel('Energia')
plt.title(f'N={N}, Bose-Hubbard Quench (U={U_i} -> {U_f})')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.show()