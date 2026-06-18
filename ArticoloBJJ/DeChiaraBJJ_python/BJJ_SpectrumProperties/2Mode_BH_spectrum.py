import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt

def twomode_bose_hubbard_hamiltonian(N, K, U):
    """
    Constructs the two-mode Bose-Hubbard Hamiltonian for a system of N bosons.
    
    Parameters:
    - N: Total number of bosons
    - K: Tunneling strength
    - U: Self-interaction strength
    
    Returns:
    - H: The Hamiltonian matrix
    """
    # Number of basis states (0 to N bosons in mode 1, rest in mode 2)
    dim = N + 1
    
    # Initialize the Hamiltonian matrix
    H = np.zeros((dim, dim))
    
    # Interaction term: U/2 * n(n-1) for both modes
    for n1 in range(N + 1):
        n2 = N - n1
        interaction_energy = (U / 4) * (n1 - n2)**2
        H[n1, n1] = interaction_energy
    
    # Tunneling term: -k/2 * (a1^dagger a2 + a2^dagger a1)
    for n1 in range(1, N + 1):
        # Off-diagonal tunneling elements
        H[n1, n1 - 1] = H[n1 - 1, n1] = -K/2 * np.sqrt(n1 * (N - n1 + 1))
    
    return H


def atomnumberdifference(N):
    dim = N + 1
    n = np.zeros((dim, dim))
    for i in range(dim):
        n[i, i] = i - N/2
    return n

def coherence(N):
    dim = N + 1
    alpha = np.zeros((dim, dim))
    for n1 in range(1, N + 1):
        alpha[n1, n1 - 1] = alpha[n1 - 1, n1] = 1/2 * np.sqrt(n1 * (N - n1 + 1))
    return alpha

# Parametri
N = 100
N_grid=1000

K = 1
# Generiamo U/K da 10^-5 a 10^5 per centrare i confini riscalati
U_range = np.geomspace(1e-5, 1e5, 100)
variance_n = []
average_coherence = []
U_f=0.1*K
# Construct the final Hamiltonian
H_f = twomode_bose_hubbard_hamiltonian(N, K, U_f)
# Final plasma frequency
omega_f=K*(U_f*N/K+1)**0.5


# In the following we study the energy spectrum of H_f
final_eigenvalues, final_eigenvectors = eigh(H_f)
spectrum_label=np.arange(0,final_eigenvalues.shape[0],1)
plt.plot(spectrum_label,(final_eigenvalues+K/2*(N+1))/omega_f,'o',markersize=2,color='blue')
plt.xlim(0,101)
plt.ylim(0)
plt.xlabel(r'Eigenvalue index', fontsize=14)
plt.ylabel(r'Energy', fontsize=14)

plt.show()
print(final_eigenvalues[0]+K*(N+1)/2,final_eigenvalues[2]-final_eigenvalues[1],omega_f)
n1=np.arange(N+1)
n2=vettore = np.arange(N, -1, -1)
q=(n1-n2)/np.sqrt(2*N)
dq = q[1] - q[0]
finalgroundstate=final_eigenvectors[:,0]
# Invece di x = np.linspace(-6, 6, N_grid)
# Usa i limiti calcolati dai dati numerici:
x = np.linspace(q.min(), q.max(), N_grid)
# Calcolo della funzione d'onda dello stato fondamentale psi_0
# Costante di normalizzazione
norm = (1/K* omega_f / (np.pi))**0.25
alpha = np.sqrt(omega_f / K) # Parametro di scala (alpha = sqrt(m*omega/hbar))
psi_1 = norm * (2 * alpha * x) * np.exp(-(alpha**2 * x**2) / 2)

# Esponenziale (la Gaussiana)
psi_0 = norm * np.exp(-(1/K * omega_f * x**2) / 2)
# Costante di normalizzazione per n=1: N1 = N0 / sqrt(2)
norm1 = norm/ np.sqrt(2)

# Funzione d'onda teorica psi_1: N1 * H1(alpha*x) * exp(...)
# Nota: H1(z) = 2*z
psi_1 = norm1 * (2 * alpha * x) * np.exp(-(alpha**2 * x**2) / 2)
# --- FIGURA 1: Ground State ---
plt.figure(figsize=(8, 5))
plt.plot(q, finalgroundstate / np.sqrt(dq), 'o', markersize=3, color='blue')
plt.plot(x, psi_0, '-', color='red')
plt.xlim(-np.sqrt(N/2), np.sqrt(N/2))
plt.xlabel(r'$q$', fontsize=12)
plt.ylabel(r'$\Phi_0(q)$', fontsize=12)
# plt.show() # Se vuoi vederle una alla volta metti show qui, altrimenti alla fine

# --- FIGURA 2: Primo Stato Eccitato ---
plt.figure(figsize=(8, 5))
plt.plot(q, final_eigenvectors[:,1] / np.sqrt(dq), 'o', markersize=3, color='green', label='1st Excited (Num)')
plt.plot(x, psi_1, '-', color='orange', label='$\Phi_1(q)$ (Teorico)')
plt.xlim(-np.sqrt(N/2), np.sqrt(N/2))
plt.xlabel(r'$q$', fontsize=12)
plt.ylabel(r'$\Phi_1(q)$', fontsize=12)
#plt.legend()

plt.show() # Mostra entrambe le finestre
# Creiamo una figura con 2 righe e 1 colonna
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

# --- PRIMO GRAFICO: Ground State ---
ax1.plot(q, finalgroundstate / np.sqrt(dq), 'o', markersize=3, color='blue', label='Ground State (Num)')
ax1.plot(x, psi_0, '-', color='red', label='$\Phi_0(q)$ (Teorico)')
ax1.set_ylabel(r'$\Phi_0(q)$', fontsize=12)
#ax1.legend()
ax1.grid(True, alpha=0.3)
#ax1.set_title("Stato Fondamentale")

# --- SECONDO GRAFICO: Primo Stato Eccitato ---
ax2.plot(q, final_eigenvectors[:,1] / np.sqrt(dq), 's', markersize=3, color='green', label='1st Excited (Num)')
ax2.plot(x, psi_1, '--', color='orange', label='$\Phi_1(q)$ (Teorico)')
ax2.set_ylabel(r'$\Phi_1(q)$', fontsize=12)
ax2.set_xlabel(r'$q$', fontsize=12)
#ax2.legend()
ax2.grid(True, alpha=0.3)
#ax2.set_title("Primo Stato Eccitato")

# Impostazioni comuni
plt.xlim(-np.sqrt(N/2), np.sqrt(N/2))
plt.tight_layout() # Sistema gli spazi tra i grafici automaticamente
plt.show()
for U in U_range:
    H = twomode_bose_hubbard_hamiltonian(N, K, U)
    n_op = atomnumberdifference(N)
    alpha_op = coherence(N)
    nsquared = n_op @ n_op
    eigenvalues, eigenvectors = eigh(H)
    groundstate = eigenvectors[:, 0]

    average_nsquared = np.dot(groundstate, nsquared.dot(groundstate))
    average_n = np.dot(groundstate, n_op.dot(groundstate))
    sigma_n2 = average_nsquared - np.power(average_n, 2)
    variance_n.append(sigma_n2)
    
    average_alpha = np.dot(groundstate, alpha_op.dot(groundstate))
    average_coherence.append(average_alpha)

# Valori asse X: U/K
x_vals = U_range / K

# Normalizzazione dati
y_data_variance = np.array(variance_n) / (N/4)
y_data_coherence = np.array(average_coherence) / (N/2)

# --- CREAZIONE GRAFICO ---
plt.figure(figsize=(8, 6))

# Calcolo dei confini riscalati per U/K (con N=100)
border_rabi_josephson = 1 / N  # 10^-2
border_josephson_fock = N      # 10^2

# 1. Colorazione delle Regioni (Versione più intensa)
# Rabi (Verde pastello più saturo)
plt.axvspan(1e-5, border_rabi_josephson, color='#c3e6cb', alpha=0.8, label='Rabi') 

# Josephson (Giallo/Sabbia più saturo)
plt.axvspan(border_rabi_josephson, border_josephson_fock, color='#ffeeba', alpha=0.8, label='Josephson')

# Fock (Grigio argento intenso)
plt.axvspan(border_josephson_fock, 1e5, color='#d6d8db', alpha=1.0, label='Fock')

# 2. Linee verticali divisorie
plt.axvline(x=border_rabi_josephson, color='grey', linestyle='--', linewidth=1, alpha=0.6)
plt.axvline(x=border_josephson_fock, color='grey', linestyle='--', linewidth=1, alpha=0.6)

# 3. Grafici dei dati
plt.plot(x_vals, y_data_variance, 'o', color='blue', markersize=3, 
         label=r'$\frac{\Delta n^2}{N/4}$', zorder=5)
plt.plot(x_vals, y_data_coherence, 'o', color='red', markersize=3, 
         label=r'$\alpha$', zorder=5)

# Impostazioni Assi
plt.xscale('log')
plt.xlim(1e-5, 1e5)
plt.ylim(0, 1.1)
plt.xlabel(r'$U/K$', fontsize=20)
#plt.ylabel('Ground state values', fontsize=12)
plt.grid(True, which="both", ls="-", alpha=0.1)
plt.tick_params(axis='both', labelsize=20)

# Legenda
plt.legend(loc='upper right', frameon=True, shadow=True, fontsize=15)
plt.tight_layout()
plt.show()