import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt

# =============================================================================
#  FUNZIONE HAMILTONIANA CONDIVISA
# =============================================================================

def twomode_bose_hubbard_hamiltonian(N, K, U):
    """
    Costruisce l'Hamiltoniana di Bose-Hubbard a due modi.
    """
    dim = N + 1
    H = np.zeros((dim, dim))
    
    # Termine di interazione: U/4 * (n1 - n2)^2
    for n1 in range(N + 1):
        n2 = N - n1
        interaction_energy = (U / 4) * (n1 - n2)**2
        H[n1, n1] = interaction_energy
    
    # Termine di tunneling
    for n1 in range(1, N + 1):
        H[n1, n1 - 1] = H[n1 - 1, n1] = -K/2 * np.sqrt(n1 * (N - n1 + 1))
    
    return H

# =============================================================================
#  PARAMETRI GENERALI
# =============================================================================
N = 100
K = 1.0
spectrum_label_limit = 101 # Limite asse x per il plot

# Creazione della figura con due subplot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# =============================================================================
#  SUBPLOT 1: Primo Script (Interazione Debole)
# =============================================================================
U_f1 = 0.001 * K
H_f1 = twomode_bose_hubbard_hamiltonian(N, K, U_f1)
omega_f1 = K * np.sqrt(U_f1 * N / K + 1)

# Diagonalizzazione
evals1, evecs1 = eigh(H_f1)
indices1 = np.arange(0, evals1.shape[0], 1)

# Normalizzazione come da primo script: (E + K(N+1)/2) / omega_f
offset1 = K/2 * (N + 1)
y_plot1 = (evals1 + offset1) / omega_f1

ax1.plot(indices1, y_plot1, 'o', markersize=3, color='blue')
ax1.set_xlim(0, spectrum_label_limit)
ax1.set_ylim(bottom=0)
ax1.set_xlabel(r'Eigenvalue index $[i]$', fontsize=18)
ax1.set_ylabel(r'$E\left[i\right]/\omega_f$', fontsize=18)
ax1.tick_params(axis='both', labelsize=16)


# =============================================================================
#  SUBPLOT 2: Secondo Script (Interazione Forte + Teoria)
# =============================================================================
U_f2 = 0.1 * K
H_f2 = twomode_bose_hubbard_hamiltonian(N, K, U_f2)
omega_f2 = K * np.sqrt(U_f2 * N / K + 1.0)

# Diagonalizzazione
evals2, evecs2 = eigh(H_f2)
indices2 = np.arange(0, evals2.shape[0], 1)

# Offset e Normalizzazione come da secondo script: E / NK
offset2 = K/2 * (N + 1) # Circa E_j
y_numerical2 = (evals2 + offset2) / (K * N)

# Curve teoriche
# 1. Scaling Lineare (Plasma)
y_linear = indices2 * omega_f2 / (K * N)

# 2. Scaling Quadratico (Regime di Fock)
# Nota: La formula usa indices^2 come approssimazione
y_quadratic = ((U_f2 / 4.0) * indices2**2 + offset2) / (K * N)

# 3. Linea di transizione (Energia critica NK)
transition_energy_val = ((N + 1) * K / 2.0 + offset2) / (K * N)


# Plotting
ax2.plot(indices2, y_numerical2, 'o', markersize=3, color='blue', label='Numerical')
ax2.plot(indices2, y_linear, '--', color='green', linewidth=2, label=r'Linear Regime')
ax2.plot(indices2, y_quadratic, ':', color='red', linewidth=3, label=r'Quadratic Regime')
ax2.axhline(y=transition_energy_val, color='black', linestyle='-.', linewidth=1.5, label=r'$NK$')

ax2.set_xlim(0, spectrum_label_limit)
ax2.set_ylim(0, np.max(y_numerical2[:spectrum_label_limit])*1.1)
ax2.set_xlabel(r'Eigenvalue index $[i]$', fontsize=18)
ax2.set_ylabel(r'$E\left[i\right]/NK$', fontsize=18)
ax2.legend(loc='upper left', fontsize=14)
ax2.tick_params(axis='both', labelsize=16)


# =============================================================================
#  LAYOUT FINALE
# =============================================================================
plt.tight_layout()
plt.show()