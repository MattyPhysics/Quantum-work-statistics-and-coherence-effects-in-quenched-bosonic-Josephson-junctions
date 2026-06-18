import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from scipy.special import lpmv, gammaln
import matplotlib.ticker as ticker

# =============================================================================
#  1. FUNZIONI FISICHE E MATEMATICHE
# =============================================================================

def twomode_bose_hubbard_hamiltonian(N, K, U):
    """Costruisce l'Hamiltoniana di Bose-Hubbard a due modi (BJJ)."""
    dim = N + 1
    H = np.zeros((dim, dim))
    # Termine di interazione (U)
    for n1 in range(N + 1):
        n2 = N - n1
        # Hamiltoniana coerente con Eq. 1.21 (H = U/4 * (nr - nl)^2 ...)
        H[n1, n1] = (U / 4) * (n1 - n2)**2
    # Termine di tunneling (K)
    for n1 in range(1, N + 1):
        H[n1, n1 - 1] = H[n1 - 1, n1] = -K/2 * np.sqrt(n1 * (N - n1 + 1))
    return H

def SqueezingMatrixTheory(m, n, x, index_sign):
    """
    Calcola l'elemento di matrice <m|S|n> per l'Operatore di Squeezing 
    usando i Polinomi di Legendre associati.
    """
    if (m + n) % 2 != 0: return 0
    l = (m + n) // 2
    k = (m - n) // 2
    
    # lpmv(m, v, x) corrisponde a P_v^m(x)
    legendrevalue = lpmv(abs(k), l, x)
    
    # Calcolo prefattore con logaritmi per stabilità numerica sui fattoriali
    log_pref = 0.5 * (gammaln(min(n, m) + 1) - gammaln(max(n, m) + 1))
    
    # Fase corretta basata sulla direzione del quench (index_sign)
    if m >= n:
        phase = (-index_sign)**abs(k)
    else:
        phase = (index_sign)**abs(k)
        
    return np.exp(log_pref) * np.sqrt(x) * legendrevalue * phase

def fix_phases(V):
    """
    CRUCIALE: Allinea la fase globale degli autovettori numerici.
    Impone che il componente con modulo massimo sia reale positivo.
    Senza questo, l'interferenza numerica avrà segno casuale rispetto alla teoria.
    """
    for i in range(V.shape[1]):
        idx = np.argmax(np.abs(V[:, i]))
        phase = np.sign(V[idx, i])
        if phase != 0:
            V[:, i] *= phase
    return V

# =============================================================================
#  2. PARAMETRI E DIAGONALIZZAZIONE
# =============================================================================

# Parametri fisici
N = 100
K = 1.0
U_i = 0.1* K  # Interazione iniziale
U_f = 0.0    # Interazione finale (Quench a non interagente)

# Diagonalizzazione Numerica BJJ
H_i = twomode_bose_hubbard_hamiltonian(N, K, U_i)
H_f = twomode_bose_hubbard_hamiltonian(N, K, U_f)
E_i, V_i = eigh(H_i)
E_f, V_f = eigh(H_f)

# --- ALLINEAMENTO FASI ---
V_i = fix_phases(V_i)
V_f = fix_phases(V_f)

# Stato Iniziale |Psi> = c0|0> + c2|2>
c_coeffs = np.zeros(N + 1, dtype=complex)
#c_coeffs[0] = 0.3158
#c_coeffs[2] = np.sqrt(1 - np.abs(c_coeffs[0])**2) # Normalizzazione esatta
c_coeffs[0] = 0.69
c_coeffs[2] = np.sqrt(1 - np.abs(c_coeffs[0])**2) # Normalizzazione esatta
rho_0 = np.outer(c_coeffs, np.conj(c_coeffs))

# Parametri Analitici (Mappatura a QHO)
omega_i = K * np.sqrt(U_i * N / K + 1) 
omega_f = K * np.sqrt(U_f * N / K + 1) 
index_sign = np.sign(omega_f - omega_i)

# Parametri Bogoliubov e Legendre per Quench Istantaneo
Q_param = (omega_i**2 + omega_f**2) / (2 * omega_i * omega_f)
u_bog = np.sqrt((Q_param + 1) / 2.0)
x_leg = 1.0 / u_bog

# =============================================================================
#  3. CALCOLO QUASIPROBABILITÀ (KDQ)
# =============================================================================

# Matrice di sovrapposizione numerica
SqueezingMatrixNum = V_f.T @ V_i 

q_num = np.zeros((N + 1, N + 1), dtype=complex)
q_theory = np.zeros((N + 1, N + 1), dtype=complex)

# Loop solo sugli stati iniziali popolati (n=0, 2) per efficienza
# Ma calcoliamo tutti gli m finali per la normalizzazione
for n in [0, 2]:
    for m in range(0, N + 1, 2): # Parità conservata, step 2
        # --- A. Numerico ---
        # Somma su s=0,2 (stati in sovrapposizione iniziale)
        interf_num = sum(rho_0[n, s] * np.conj(SqueezingMatrixNum[m, s]) for s in [0, 2])
        q_num[m, n] = SqueezingMatrixNum[m, n] * interf_num
        
        # --- B. Teorico ---
        S_mn_teo = SqueezingMatrixTheory(m, n, x_leg, index_sign)
        interf_theory = sum(rho_0[n, s] * SqueezingMatrixTheory(m, s, x_leg, index_sign) for s in [0, 2])
        q_theory[m, n] = S_mn_teo * interf_theory

# =============================================================================
#  4. VERIFICHE: NORMALIZZAZIONE E LAVORO MEDIO
# =============================================================================

print("\n" + "="*60)
print("VERIFICA NORMALIZZAZIONE (Somma su matrice NxN)")
print("="*60)
norm_num = np.sum(q_num)
norm_th = np.sum(q_theory)
print(f"{'Metodo':<15} | {'Reale (Target: 1.0)':<20} | {'Immaginaria (Target: 0.0)':<25}")
print("-" * 65)
print(f"{'Numerico':<15} | {norm_num.real:<20.8f} | {norm_num.imag:<25.8f}")
print(f"{'Teorico':<15} | {norm_th.real:<20.8f} | {norm_th.imag:<25.8f}")

print("\n" + "="*60)
print("CONFRONTO MEDIA DEL LAVORO <W>")
print("="*60)

# A. Numerico: Somma pesata su spettro BH
W_avg_num = 0
for n in [0, 2]:
    for m in range(N + 1):
        val_w = E_f[m] - E_i[n]
        W_avg_num += q_num[m, n].real * val_w

# B. Teorico: Formula Chiusa HO
prob_0 = np.abs(c_coeffs[0])**2
prob_2 = np.abs(c_coeffs[2])**2
# Termine diagonale (TPM)
term_tpm = prob_0 * (2*0 + 1) + prob_2 * (2*2 + 1)
# Termine di coerenza (Interferenza)
# Nota: 2*Re(c0*c2) * sqrt((n+1)(n+2)) per n=0 -> sqrt(1*2)
term_coh = 2 * np.real(np.conj(c_coeffs[0]) * c_coeffs[2]) * np.sqrt(2)

prefactor = (omega_f**2 - omega_i**2) / (4 * omega_i)
W_avg_theory = prefactor * (term_tpm + term_coh)

print(f"Media Numerica (BH):   {W_avg_num:.6f}")
print(f"Media Teorica (HO):    {W_avg_theory:.6f}")
print(f"Errore Relativo:       {abs(W_avg_num - W_avg_theory)/abs(W_avg_num)*100:.4f}%")
print("-" * 60)

# =============================================================================
#  5. PLOTTING
# =============================================================================
# =============================================================================
#  5. PLOTTING (VERSIONE CORRETTA)
# =============================================================================

# ... (I plot ax1 e ax2 precedenti vanno bene) ...

# Plot 2: Distribuzione Totale P(W)
all_W_num, all_q_num = [], []
all_W_th, all_q_th = [], []

m_limit = 24 # Mantieni lo stesso limite per coerenza visiva

for n in [0, 2]:
    for m in range(0, m_limit, 2):
        # Numerico: usiamo le proprie energie e i propri pesi
        val_q_num = np.real(q_num[m, n])
        if abs(val_q_num) > 1e-7:
            all_W_num.append(E_f[m] - E_i[n])
            all_q_num.append(val_q_num)
            
        # Teorico: usiamo le proprie energie e i propri pesi
        val_q_th = np.real(q_theory[m, n])
        if abs(val_q_th) > 1e-7:
            all_W_th.append(omega_f*(m+0.5) - omega_i*(n+0.5))
            all_q_th.append(val_q_th)

plt.figure(figsize=(10, 6), facecolor='white')

# Plot Numerico come barre (usando i dati numerici)
plt.bar(all_W_num, all_q_num, width=0.3, color='purple', alpha=0.4, 
        edgecolor='indigo', label='Numerical (BJJ)')

# Plot Teorico come stem (usando i dati teorici)
# NOTA: Ora all_W_th e all_q_th hanno la stessa lunghezza!
markerline, stemlines, baseline = plt.stem(all_W_th, all_q_th, linefmt='r-', 
                                           markerfmt='ro', basefmt=" ", label='Theory (QHO)')

plt.setp(markerline, markersize=5)
plt.setp(stemlines, linewidth=1.5)

plt.axhline(0, c='k', lw=1)
plt.xlabel(r'Work $W$', fontsize=14)
plt.ylabel(r'Quasiprobability $P(W)$', fontsize=14)
plt.title(rf'Total Work Distribution | Quench $U_i={U_i/K}K \to U_f={U_f/K}K$', fontsize=14)
plt.legend()
plt.grid(alpha=0.2, linestyle='--')
plt.tight_layout()
plt.show()