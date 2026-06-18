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
        # Hamiltoniana coerente con Eq. 1.21
        H[n1, n1] = (U / 4) * (n1 - n2)**2
    # Termine di tunneling (K)
    for n1 in range(1, N + 1):
        H[n1, n1 - 1] = H[n1 - 1, n1] = -K/2 * np.sqrt(n1 * (N - n1 + 1))
    return H

def SqueezingMatrixTheory(m, n, x, index_sign):
    """
    Calcola l'elemento <m|S|n> usato qui SOLO per calcolare la probabilità base p(m)=|S_m0|^2
    necessaria per le formule ricorsive.
    """
    if (m + n) % 2 != 0: return 0
    l = (m + n) // 2
    k = (m - n) // 2
    
    legendrevalue = lpmv(abs(k), l, x)
    log_pref = 0.5 * (gammaln(min(n, m) + 1) - gammaln(max(n, m) + 1))
    
    if m >= n:
        phase = (-index_sign)**abs(k)
    else:
        phase = (index_sign)**abs(k)
        
    return np.exp(log_pref) * np.sqrt(x) * legendrevalue * phase

def fix_phases(V):
    """Allinea la fase globale degli autovettori numerici."""
    for i in range(V.shape[1]):
        idx = np.argmax(np.abs(V[:, i]))
        phase = np.sign(V[idx, i])
        if phase != 0:
            V[:, i] *= phase
    return V

def analytical_KDQ_recursive(m, Q, alpha, beta, index_sign):
    """
    Implementazione esplicita delle Eq. 3.90 e 3.91 della tesi.
    Calcola q_m0 e q_m2 usando la relazione ricorsiva tra S_m2 e S_m0.
    """
    # Definizione di x come da Eq. 3.91 (testo sotto l'equazione)
    x = np.sqrt(2 / (Q + 1))
    
    # Calcoliamo S_m0 usando la teoria standard (Legendre) per ottenere p(m)
    # Nota: per il sudden quench, S_mn sono reali, quindi p(m) = (S_m0)^2
    # Serve x_leg per la funzione di Legendre, che è 1/u = sqrt(2/(Q+1)) = x
    # Quindi passiamo 'x' direttamente come argomento alla funzione SqueezingMatrixTheory
    S_m0 = SqueezingMatrixTheory(m, 0, x, index_sign)
    p_m = np.abs(S_m0)**2 

    # --- Eq. 3.90: Relazione ricorsiva ---
    # Definiamo il fattore moltiplicativo tra S_m2 e S_m0
    # R(m) = [1 - x^2(m+1)] / sqrt(2(1-x^2))
    denom = np.sqrt(2 * (1 - x**2))
    
    # Protezione numerica per Q=1 (nessun quench)
    if denom < 1e-10: 
        return (p_m * np.abs(alpha)**2, 0) # Ritorna valori limite o dummy

    R_m = (1 - (x**2) * (m + 1)) / denom
    
    # --- Eq. 3.91: Formule analitiche per KDQ ---
    
    # q_m0 = p(m) * [ |alpha|^2 + R(m)*alpha*conj(beta) ]
    # Nota: La tesi usa la notazione alpha * beta* (rho_02 = c0 c2*)
    term_interf_0 = R_m * alpha * np.conj(beta)
    q_m0 = p_m * (np.abs(alpha)**2 + term_interf_0)
    
    # q_m2 = p(m) * [ R(m)*|beta|^2 + alpha* * beta ] * R(m) ???
    # Guardando l'equazione 3.91 nello screenshot:
    # q_m2 = p(m) * [1-x^2(m+1)/sqrt(...)]^2 / [2(1-x^2)] ... NO, usiamo la forma compatta:
    # q_m2 = S_m2 * [ S_m2 * rho_22 + S_m0 * rho_02 ]
    # Sapendo che S_m2 = S_m0 * R_m:
    # q_m2 = (S_m0 * R_m) * [ (S_m0 * R_m) * |beta|^2 + S_m0 * alpha * conj(beta) ]
    # q_m2 = (S_m0^2) * R_m * [ R_m * |beta|^2 + alpha * conj(beta) ]
    # q_m2 = p(m) * R_m * [ R_m * |beta|^2 + alpha * conj(beta) ]
    
    # VERIFICA CON SCREENSHOT EQ 3.91:
    # Lo screenshot dice: q_m2 = p(m) * R_m^2 * [ |beta|^2 + (1/R_m)*conj(alpha)*beta ]
    # Che semplificando diventa: p(m) * R_m * [ R_m*|beta|^2 + conj(alpha)*beta ]
    # ATTENZIONE: Nello screenshot c'è alpha^* beta, nella mia derivazione sopra alpha beta^*.
    # Questo dipende dalla definizione di rho_jk. Solitamente rho = |psi><psi|, 
    # quindi rho_jk = c_j c_k*. Per q_m2 serve rho_02 (o rho_20?)
    # q_mn = <m|U|n><n|rho|m>... nel formalismo KDQ q_mn = Tr(Pi_m Pi_n rho).
    # Per stato puro iniziale somma di 0 e 2:
    # q_m2 = <m|U|2><2|psi><psi|0><0|U|m>* + <m|U|2><2|psi><psi|2><2|U|m>*
    # q_m2 = S_m2 [ c_2 c_0* S_m0 + c_2 c_2* S_m2 ] (assumendo S reali)
    # q_m2 = S_m2 [ beta alpha* S_m0 + |beta|^2 S_m2 ]
    # q_m2 = (S_m0 R_m) [ beta alpha* S_m0 + |beta|^2 S_m0 R_m ]
    # q_m2 = p(m) R_m [ alpha* beta + R_m |beta|^2 ]
    # Questa corrisponde esattamente alla formula dello screenshot (con alpha* beta).
    
    term_interf_2 = np.conj(alpha) * beta
    q_m2 = p_m * R_m * (R_m * np.abs(beta)**2 + term_interf_2)
    
    return q_m0, q_m2

# =============================================================================
#  2. PARAMETRI E DIAGONALIZZAZIONE
# =============================================================================

N = 100
K = 1.0
U_i = 0.1 * K  
U_f = 0.0    

# Diagonalizzazione Numerica
H_i = twomode_bose_hubbard_hamiltonian(N, K, U_i)
H_f = twomode_bose_hubbard_hamiltonian(N, K, U_f)
E_i, V_i = eigh(H_i)
E_f, V_f = eigh(H_f)
V_i = fix_phases(V_i)
V_f = fix_phases(V_f)

# Stato Iniziale
c_coeffs = np.zeros(N + 1, dtype=complex)
c_coeffs[0] = 0.3158
c_coeffs[2] = np.sqrt(1 - np.abs(c_coeffs[0])**2)
rho_0 = np.outer(c_coeffs, np.conj(c_coeffs))

# Parametri Analitici
omega_i = K * np.sqrt(U_i * N / K + 1) 
omega_f = K * np.sqrt(U_f * N / K + 1) 
index_sign = np.sign(omega_f - omega_i) # Per la fase di S_m0

# Parametri per formule 3.90/3.91
Q_param = (omega_i**2 + omega_f**2) / (2 * omega_i * omega_f)
# Coefficienti alpha e beta per le formule (c0 e c2)
alpha_val = c_coeffs[0]
beta_val = c_coeffs[2]

# =============================================================================
#  3. CALCOLO QUASIPROBABILITÀ (KDQ)
# =============================================================================

# Matrice numerica per confronto
SqueezingMatrixNum = V_f.T @ V_i 

q_num = np.zeros((N + 1, N + 1), dtype=complex)
q_theory = np.zeros((N + 1, N + 1), dtype=complex)

for m in range(0, N + 1, 2): # Parità conservata
    # --- A. Numerico ---
    # Calcolo standard tramite elementi di matrice numerici
    # q_mn = <m|U|n><n|rho|Udag|m> -> semplificato per stato puro
    # q_m0 = <m|U|0> * <0|psi><psi|m_U_evolved>
    # Ma usiamo la definizione base q_mn = Tr(Pi_m_f Pi_n_i rho)
    
    # Per n=0
    q_num[m, 0] = SqueezingMatrixNum[m, 0] * (rho_0[0, 0]*np.conj(SqueezingMatrixNum[m, 0]) + 
                                              rho_0[0, 2]*np.conj(SqueezingMatrixNum[m, 2]))
    # Per n=2
    q_num[m, 2] = SqueezingMatrixNum[m, 2] * (rho_0[2, 0]*np.conj(SqueezingMatrixNum[m, 0]) + 
                                              rho_0[2, 2]*np.conj(SqueezingMatrixNum[m, 2]))

    # --- B. Teorico (USANDO EQ 3.90 e 3.91) ---
    qm0_th, qm2_th = analytical_KDQ_recursive(m, Q_param, alpha_val, beta_val, index_sign)
    
    q_theory[m, 0] = qm0_th
    q_theory[m, 2] = qm2_th

# =============================================================================
#  4. VERIFICHE
# =============================================================================

print("\n" + "="*60)
print("VERIFICA NORMALIZZAZIONE (Somma su matrice)")
print("="*60)
norm_num = np.sum(q_num)
norm_th = np.sum(q_theory)
print(f"{'Metodo':<15} | {'Reale':<20} | {'Immaginaria':<25}")
print("-" * 65)
print(f"{'Numerico':<15} | {norm_num.real:<20.8f} | {norm_num.imag:<25.8f}")
print(f"{'Teorico':<15} | {norm_th.real:<20.8f} | {norm_th.imag:<25.8f}")

print("\n" + "="*60)
print("CONFRONTO MEDIA DEL LAVORO <W>")
print("="*60)

W_avg_num = 0
for n in [0, 2]:
    for m in range(N + 1):
        val_w = E_f[m] - E_i[n]
        W_avg_num += q_num[m, n].real * val_w

# Calcolo Teorico Media (Eq 3.92 della tesi)
# <W>_TPM = (delta_omega^2 / 2w_i) * (2|beta|^2 + 1/2)
# <W>_coh = (sqrt(2)/2) * (delta_omega^2 / w_i) * Re(alpha*beta*)
delta_sq = omega_f**2 - omega_i**2
W_TPM = (delta_sq / (2 * omega_i)) * (2 * np.abs(beta_val)**2 + 0.5)
W_coh = (np.sqrt(2) / 2) * (delta_sq / omega_i) * np.real(alpha_val * np.conj(beta_val))
W_avg_theory = W_TPM + W_coh

print(f"Media Numerica (BH):   {W_avg_num:.6f}")
print(f"Media Teorica (Eq.3.92): {W_avg_theory:.6f}")
print(f"Errore Relativo:       {abs(W_avg_num - W_avg_theory)/abs(W_avg_num)*100:.4f}%")

# =============================================================================
#  5. PLOTTING
# =============================================================================

W_m0 = E_f - E_i[0]
W_m2 = E_f - E_i[2]
m_indices = np.arange(N + 1)
# Energie teoriche HO
W_th_m0 = omega_f * (m_indices + 0.5) - omega_i * (0.5)
W_th_m2 = omega_f * (m_indices + 0.5) - omega_i * (2.5)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor='white')
m_limit = 24 
mask = (m_indices % 2 == 0) & (m_indices < m_limit)

# Plot q_m0
ax1.bar(W_m0[mask], np.real(q_num[mask, 0]), width=0.4, color='royalblue', alpha=0.5, 
        edgecolor='blue', label='Numerics')
markerline, stemlines, _ = ax1.stem(W_th_m0[mask], np.real(q_theory[mask, 0]), 
                                    linefmt='r-', markerfmt='ro', basefmt=" ", label='Theory (Eq 3.91)')
plt.setp(markerline, markersize=4); plt.setp(stemlines, linewidth=1.2)
ax1.set_xlabel(r'$W$', fontsize=12); ax1.set_ylabel(r'$q_{m0}$', fontsize=12)
ax1.legend(); ax1.axhline(0, c='k', lw=0.8)
ax1.set_title("KDQ component: Initial state |0>")

# Plot q_m2
ax2.bar(W_m2[mask], np.real(q_num[mask, 2]), width=0.4, color='forestgreen', alpha=0.5, 
        edgecolor='green', label='Numerics')
markerline, stemlines, _ = ax2.stem(W_th_m2[mask], np.real(q_theory[mask, 2]), 
                                    linefmt='r-', markerfmt='ro', basefmt=" ", label='Theory (Eq 3.91)')
plt.setp(markerline, markersize=4); plt.setp(stemlines, linewidth=1.2)
ax2.set_xlabel(r'$W$', fontsize=12); ax2.set_ylabel(r'$q_{m2}$', fontsize=12)
ax2.legend(); ax2.axhline(0, c='k', lw=0.8)
ax2.set_title("KDQ component: Initial state |2>")

plt.tight_layout()
plt.show()

# Plot Totale
all_W_num, all_q_num = [], []
all_W_th, all_q_th = [], []

for n in [0, 2]:
    for m in range(0, m_limit, 2):
        if abs(np.real(q_num[m, n])) > 1e-6:
            all_W_num.append(E_f[m] - E_i[n])
            all_q_num.append(np.real(q_num[m, n]))
        if abs(np.real(q_theory[m, n])) > 1e-6:
            # Ricostruzione energia teorica
            en_th_i = omega_i * (n + 0.5)
            en_th_f = omega_f * (m + 0.5)
            all_W_th.append(en_th_f - en_th_i)
            all_q_th.append(np.real(q_theory[m, n]))

plt.figure(figsize=(10, 6), facecolor='white')
plt.bar(all_W_num, all_q_num, width=0.4, color='purple', alpha=0.4, 
        edgecolor='indigo', label='Numerical Total P(W)')
markerline, stemlines, _ = plt.stem(all_W_num, all_q_th, linefmt='r-', 
                                    markerfmt='ro', basefmt=" ", label='Theory Total P(W)')
plt.setp(markerline, markersize=4); plt.setp(stemlines, linewidth=1)
plt.axhline(0, c='k', lw=1)
plt.xlabel(r'Work $W$', fontsize=14)
plt.ylabel(r'Quasiprobability $P(W)$', fontsize=14)
plt.title(f'Total Work Distribution using Eq 3.90/3.91', fontsize=14)
plt.legend()
plt.grid(alpha=0.2, linestyle='--')
plt.tight_layout()
plt.show()