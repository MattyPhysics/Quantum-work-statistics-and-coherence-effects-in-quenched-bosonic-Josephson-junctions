import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# =============================================================================
#  1. MOTORE NUMERICO (BOSE-HUBBARD)
# =============================================================================

def twomode_bose_hubbard_hamiltonian(N, K, U):
    dim = N + 1
    H = np.zeros((dim, dim))
    for n1 in range(N + 1):
        n2 = N - n1
        H[n1, n1] = (U / 4) * (n1 - n2)**2
    for n1 in range(1, N + 1):
        H[n1, n1 - 1] = H[n1 - 1, n1] = -K/2 * np.sqrt(n1 * (N - n1 + 1))
    return H

def fix_phases(V):
    for i in range(V.shape[1]):
        idx = np.argmax(np.abs(V[:, i]))
        V[:, i] *= np.sign(V[idx, i])
    return V

# Parametri Quench
N = 100
K = 1.0
U_i = 0.1* K
U_f = 0.0

# Diagonalizzazione Numerica
H_i = twomode_bose_hubbard_hamiltonian(N, K, U_i)
H_f = twomode_bose_hubbard_hamiltonian(N, K, U_f)
E_i, V_i = eigh(H_i)
E_f, V_f = eigh(H_f)
V_i = fix_phases(V_i)
V_f = fix_phases(V_f)

# Matrice di Squeezing Numerica (Overlap tra autostati iniziali e finali)
# S[m, n] = <psi_f^m | psi_i^n>
S_num = V_f.T @ V_i 

# =============================================================================
#  2. CALCOLO METRICHE (KDQ E BOUND)
# =============================================================================

def calculate_numerical_metrics(c0_mod, phi):
    """Calcola W_KDQ e Bound usando esclusivamente i dati numerici di BH."""
    c2_mod = np.sqrt(1 - c0_mod**2)
    # Stato iniziale: superposizione degli autostati n=0 e n=2 di H_i
    coeffs = {0: c0_mod, 2: c2_mod * np.exp(1j * phi)}
    
    W_kdq_total = 0.0
    Bound_total = 0.0
    
    # Indici degli stati iniziali coinvolti
    n_states = [0, 2]
    
    for n in n_states:
        p_n_tpm = abs(coeffs[n])**2
        energy_i = E_i[n]
        
        for m in range(N + 1):
            energy_f = E_f[m]
            W_extracted = energy_i - energy_f  # Lavoro estratto
            
            # --- Calcolo KDQ Numerico q_{mn} ---
            # q_mn = S_mn * sum_s ( rho_ns * conj(S_ms) )
            interf_term = 0.0
            for s in n_states:
                rho_ns = coeffs[n] * np.conj(coeffs[s])
                interf_term += rho_ns * S_num[m, s] # Nota: S è reale in BH, ma usiamo la logica generale
            
            q_mn_real = np.real(S_num[m, n] * interf_term)
            W_kdq_total += q_mn_real * W_extracted
            
            # --- Calcolo Bound Classico ---
            if W_extracted > 0:
                # Probabilità TPM: p(m|n)*p(n)
                p_joint_tpm = p_n_tpm * (S_num[m, n]**2)
                # Probabilità coerente: |<m|psi>|^2
                amp_m = sum(coeffs[s] * S_num[m, s] for s in n_states)
                p_m_coherent = abs(amp_m)**2
                
                Bound_total += W_extracted * np.sqrt(p_joint_tpm * p_m_coherent)
                
    return W_kdq_total, Bound_total

# =============================================================================
#  3. ANALISI E PLOT
# =============================================================================

user_c0 = 0.8
phases = np.linspace(0, 2*np.pi, 100)
w_list, b_list = [], []

for p in phases:
    w, b = calculate_numerical_metrics(user_c0, p)
    w_list.append(w)
    b_list.append(b)

w_list = np.array(w_list)
b_list = np.array(b_list)

# Grafico
plt.rcParams.update({'font.size': 14, 'font.family': 'serif'})
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(phases/np.pi, w_list, 'b-', lw=2.5, label=r'$W_{KDQ}$ (Bose-Hubbard)')
ax.plot(phases/np.pi, b_list, 'r--', lw=2, label='Classical Bound')
ax.fill_between(phases/np.pi, w_list, b_list, where=(w_list > b_list), 
                 color='green', alpha=0.2, label='Quantum Violation')

# Punto a phi=0
w0, b0 = calculate_numerical_metrics(user_c0, 0)
ax.plot(0, w0, 'ko', markersize=8)
ax.annotate(f'W={w0:.3f}', xy=(0, w0), xytext=(0.15, w0+0.1),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1))

ax.set_xlabel(r'Relative Phase $\phi$ [$\pi$]')
ax.set_ylabel('Extracted Energy')
ax.set_title(f'Numerical Analysis (BH Model, N={N}, $c_0={user_c0:.2f}$)')
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc='best')

plt.tight_layout()
plt.show()

print(f"Risultati Numerici (phi=0): W={w0:.5f}, Bound={b0:.5f}, Violazione={w0-b0:.5f}")
