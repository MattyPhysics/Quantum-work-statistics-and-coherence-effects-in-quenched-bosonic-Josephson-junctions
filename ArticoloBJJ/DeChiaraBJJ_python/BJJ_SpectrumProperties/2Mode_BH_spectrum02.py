import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt

def twomode_bose_hubbard_hamiltonian(N, K, U):
    dim = N + 1
    H = np.zeros((dim, dim))
    for n1 in range(N + 1):
        n2 = N - n1
        interaction_energy = (U / 4) * (n1 - n2)**2
        H[n1, n1] = interaction_energy
    for n1 in range(1, N + 1):
        H[n1, n1 - 1] = H[n1 - 1, n1] = -K/2 * np.sqrt(n1 * (N - n1 + 1))
    return H

# --- PARAMETRI ---
N = 100
K = 1.0  
U_f = 0.1* K
H_f = twomode_bose_hubbard_hamiltonian(N, K, U_f)
omega_f = K * np.sqrt(U_f * N / K + 1.0) 

# --- DIAGONALIZZAZIONE ---
final_eigenvalues, final_eigenvectors = eigh(H_f)

# --- PARAMETRI FISICI ---
E_c = 2.0 * U_f         
E_j = (N +1)* K/ 2.0     

# --- PLOT ---
spectrum_label = np.arange(0, final_eigenvalues.shape[0], 1)

# IL TUO OFFSET
# Offset = K/2 * (N+1) ≈ E_j. 
# Trasla il ground state (-E_j) a 0.
offset_user = K/2 * (N + 1) 

# Dati numerici rinormalizzati
y_numerical = (final_eigenvalues + offset_user) / K/N

plt.figure(figsize=(9, 7))

# 1. Dati Numerici
plt.plot(spectrum_label, y_numerical, 'o', markersize=3, color='blue', )

# 2. Scaling Lineare (Plasma Oscillations)
plt.plot(spectrum_label, spectrum_label*omega_f/K/N, '--', color='green', linewidth=2, label=r'Linear Regime')

# 3. Scaling Quadratico (Fock)
# CORREZIONE: L'energia assoluta di interazione è 0 + i^2*Ec/8.
# Aggiungiamo solo l'offset_user. Il vertice ora sarà a ~E_j (metà grafico).
y_quadratic = ((U_f / 4.0) * spectrum_label**2 + offset_user) / K/N

plt.plot(spectrum_label, y_quadratic, ':', color='red', linewidth=3, label=r'Quadratic Regime')

# 4. LINEA DI TRANSIZIONE
# La transizione avviene a +E_j (assoluto). 
# Nel grafico: +E_j + offset_user ≈ 2*E_j
transition_energy_plot = ((N +1)* K/ 2.0   + offset_user) / K/N

plt.axhline(y=transition_energy_plot, color='black', linestyle='-.', linewidth=1.5, 
            label=r'$NK$')

# Decorazioni
plt.xlim(0, 101)
plt.ylim(0, np.max(y_numerical[:101])*1.1)
plt.xlabel(r'Eigenvalue index $[i]$', fontsize=18)
plt.ylabel(r'E[i]/NK', fontsize=18)
plt.legend(loc='upper left', fontsize=15)
plt.tick_params(axis='both', labelsize=20)
plt.grid(True, alpha=0.2)

plt.show()