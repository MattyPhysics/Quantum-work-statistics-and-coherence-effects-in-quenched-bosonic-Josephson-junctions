import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt

# --- PARAMETRI ---
N = 100
K = 1.0
omega_f = 3.0  # Frequenza finale del protocollo

# Calcolo di U corrispondente alla frequenza finale (dalla tesi)
# omega = K * sqrt(U*N/K + 1)  =>  U = (K/N) * ((omega/K)^2 - 1)
U_f = (K/N) * ((omega_f/K)**2 - 1)

# --- 1. COSTRUZIONE HAMILTONIANA BJJ (Exact) ---
def get_bjj_hamiltonian(N, K, U):
    dim = N + 1
    n_range = np.arange(dim)
    n_diff = n_range - N/2
    
    # Termine di interazione (U/4 * (n_L - n_R)^2)
    H_int = np.diag((U / 4.0) * (4 * n_diff**2)) # Nota: nella tesi H ~ U/4 * z^2
    # Correggiamo usando la forma standard H ~ U/2 * n^2 per confronto diretto o quella della tesi
    # Usiamo la forma della tesi Eq 1.23: Diag = U/4 * (N-2i)^2
    H_diag = np.diag( (U/4) * (N - 2*n_range)**2 )
    
    # Termine di tunneling
    H_off = np.zeros((dim, dim))
    for i in range(1, dim):
        val = -0.5 * K * np.sqrt(i * (N - i + 1))
        H_off[i, i-1] = H_off[i-1, i] = val
        
    return H_diag + H_off

H_bjj = get_bjj_hamiltonian(N, K, U_f)
E_bjj, _ = eigh(H_bjj)
# Shiftiamo lo zero dell'energia al ground state per confrontare le eccitazioni
E_bjj_excitations = E_bjj - E_bjj[0]

# --- 2. COSTRUZIONE SPETTRO HO (Approx) ---
# E_m = m * hbar * omega
m_levels = np.arange(N + 1)
E_ho_excitations = m_levels * omega_f

# --- 3. ANALISI DEVIAZIONE ---
limit = 20 # Guardiamo i primi 20 livelli
diff = E_ho_excitations[:limit] - E_bjj_excitations[:limit]

print(f"Indice m | E_HO (Analitico) | E_BJJ (Numerico) | Diff (Anarmonicità)")
print("-" * 65)
for m in range(limit):
    # Evidenziamo dove la differenza supera una soglia visibile (es. 0.1)
    marker = "<<" if abs(diff[m]) > 0.1 else ""
    if m % 2 == 0: # Mostriamo solo i pari perché sono quelli popolati nel tuo stato
        print(f"{m:8d} | {E_ho_excitations[m]:14.4f} | {E_bjj_excitations[m]:14.4f} | {diff[m]:14.4f} {marker}")

# --- 4. PLOT ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot Spettri
ax1.plot(m_levels[:limit], E_ho_excitations[:limit], 'r--o', label='Harmonic Oscillator (Linear)', alpha=0.6)
ax1.plot(m_levels[:limit], E_bjj_excitations[:limit], 'b-x', label='BJJ Exact (Anharmonic)')
ax1.set_xlabel('Livello di Eccitazione $m$')
ax1.set_ylabel('Energia $E_m - E_0$')
ax1.set_title('Confronto Spettri Energetici')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot Anarmonicità
ax2.plot(m_levels[:limit], diff, 'k-o', color='purple')
ax2.set_xlabel('Livello di Eccitazione $m$')
ax2.set_ylabel('$E_{HO} - E_{BJJ}$')
ax2.set_title('Deviazione Anarmonica (Errore Approssimazione)')
ax2.axhline(0.2, color='orange', linestyle='--', label='Soglia visibile graficamente')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.show()