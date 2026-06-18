import numpy as np
from scipy.linalg import eigh
from scipy.special import factorial
import matplotlib.pyplot as plt

def twomode_bose_hubbard_hamiltonian(N, K, U):
    dim = N + 1
    H = np.zeros((dim, dim))
    for n_l in range(N + 1):
        n_r = N - n_l
        H[n_l, n_l] = (U / 4.0) * (n_l - n_r)**2
    for n_l in range(N):
        elem = -K/2.0 * np.sqrt((n_l + 1) * (N - n_l))
        H[n_l + 1, n_l] = elem
        H[n_l, n_l + 1] = elem
    return H

def fix_phases_robust(evecs, N):
    dim = evecs.shape[0]
    fixed_evecs = evecs.copy()
    center = N // 2
    for n in range(dim):
        vec = fixed_evecs[:, n]
        right_part = vec[center:] 
        idx_max = np.argmax(np.abs(right_part))
        if right_part[idx_max] < 0:
            fixed_evecs[:, n] *= -1
    return fixed_evecs

# --- Parametri ---
N = 100
K = 1.0
U_f = 0.1 * K
alpha = 2.0 

H_f = twomode_bose_hubbard_hamiltonian(N, K, U_f)
omega_f = K * np.sqrt(1 + U_f * N / K)

# U range esteso per vedere bene il breakdown
U_range = np.geomspace(1e-3 * K, 1e2 * K, 50)

res_avg_num, res_avg_ana = [], []
res_var_num, res_var_ana = [], []

for U_i in U_range:
    H_i = twomode_bose_hubbard_hamiltonian(N, K, U_i)
    omega_i = K * np.sqrt(1 + U_i * N / K)
    
    evals_i, evecs_i = eigh(H_i)
    evecs_i = fix_phases_robust(evecs_i, N)
    
    psi_0 = np.zeros(N + 1, dtype=complex)
    cutoff = min(50, N)
    pref = np.exp(-abs(alpha)**2 / 2.0)
    for n in range(cutoff):
        coeff = pref * (alpha**n) / np.sqrt(factorial(n))
        psi_0 += coeff * evecs_i[:, n]
    psi_0 /= np.linalg.norm(psi_0)
    
    # Numerico
    W_op = H_f - H_i
    W_op_sq = W_op @ W_op
    avg_W = np.vdot(psi_0, W_op @ psi_0).real
    sq_W = np.vdot(psi_0, W_op_sq @ psi_0).real
    var_W = sq_W - avg_W**2
    
    res_avg_num.append(avg_W)
    res_var_num.append(var_W)
    
    # Analitico (QHO)
    # Lavoro GS e Varianza GS
    W_GS_avg = (N * K * (U_f - U_i)) / (4 * omega_i)
    W_GS_var = 2 * (W_GS_avg**2)
    
    # Scaling stato coerente
    ana_avg = W_GS_avg * (4 * alpha**2 + 1)
    ana_var = W_GS_var * (8 * alpha**2 + 1)
    
    res_avg_ana.append(ana_avg)
    res_var_ana.append(ana_var)

# --- Plotting ---
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Lavoro Medio
ax[0].loglog(U_range/K, np.abs(res_avg_num)/K, 'o', color='blue', label='Numerical (BJJ)', markersize=4)
ax[0].loglog(U_range/K, np.abs(res_avg_ana)/K, '--', color='red', label='Analytical (QHO)')
# Evidenziamo dove si rompe l'accordo
ax[0].axvline(1.0, color='gray', linestyle=':', alpha=0.5, label='Inizio Josephson')
ax[0].set_xlabel(r'$U_i/K$')
ax[0].set_ylabel(r'$|\langle W \rangle| / K$')
ax[0].set_title(f'Average Work (Alpha={alpha})')
ax[0].legend()
ax[0].grid(True, which="both", alpha=0.3)

# Varianza
ax[1].loglog(U_range/K, np.abs(res_var_num)/(K**2), 'o', color='green', label='Numerical (BJJ)', markersize=4)
ax[1].loglog(U_range/K, np.abs(res_var_ana)/(K**2), '--', color='orange', label='Analytical (QHO)')
ax[1].axvline(1.0, color='gray', linestyle=':', alpha=0.5, label='Inizio Josephson')
ax[1].set_xlabel(r'$U_i/K$')
ax[1].set_ylabel(r'$\mathrm{Var}(W) / K^2$')
ax[1].set_title(f'Work Variance (Alpha={alpha})')
ax[1].legend()
ax[1].grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.show()