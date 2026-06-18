import numpy as np
import matplotlib.pyplot as plt
from scipy.special import lpmv, gammaln

# =============================================================================
#  FUNZIONI DI BASE
# =============================================================================

def SqueezingMatrixElement(m, n, x, index_sign):
    if (m + n) % 2 != 0: return 0.0
    l = (m + n) // 2
    k = (m - n) // 2
    # Protezione per x > 1 (raro ma possibile numericamente)
    x_clip = np.clip(x, -1, 1)
    legendre_val = lpmv(abs(k), l, x_clip)
    log_pref = 0.5 * (gammaln(min(n, m) + 1) - gammaln(max(n, m) + 1))
    phase = (-index_sign)**abs(k) if m >= n else (index_sign)**abs(k)
    return np.exp(log_pref) * np.sqrt(x_clip) * legendre_val * phase

def E_harmonic(n, omega):
    return omega * (n + 0.5)

def calculate_metrics(Ui_val, Uf_val, N, K, c0_mod, phi):
    # Ricalcolo frequenze per ogni Ui
    omega_i = K * np.sqrt(Ui_val * N / K + 1)
    omega_f = K * np.sqrt(Uf_val * N / K + 1)

    Q_param = (omega_i**2 + omega_f**2) / (2 * omega_i * omega_f)
    u_bog = np.sqrt((Q_param + 1) / 2.0)
    x_leg = 1.0 / u_bog
    index_sign = np.sign(omega_f - omega_i)

    c2_mod = np.sqrt(1 - c0_mod**2)
    coeffs = {0: c0_mod, 2: c2_mod * np.exp(1j * phi)}
    
    W_real_total = 0.0
    Bound_total = 0.0
    m_cutoff = 100 # Aumentato per convergenza a Ui alti
    
    for n in [0, 2]:
        p_n_TPM = abs(coeffs[n])**2
        E_i = E_harmonic(n, omega_i)
        
        for m in range(0, m_cutoff, 2):
            E_f = E_harmonic(m, omega_f)
            # Definizione di "Work Extracted" (E_in - E_out)
            W_ext = E_i - E_f 
            S_mn = SqueezingMatrixElement(m, n, x_leg, index_sign)
            
            # KDQ Real Work
            interf_KDQ = 0.0
            for s in [0, 2]:
                rho_ns = coeffs[n] * np.conj(coeffs[s])
                interf_KDQ += rho_ns * SqueezingMatrixElement(m, s, x_leg, index_sign)
            
            q_real = S_mn * np.real(interf_KDQ)
            W_real_total += q_real * W_ext
            
            # Bound Classico (solo per estrazione positiva)
            if W_ext > 0:
                p_joint_TPM = p_n_TPM * (S_mn**2)
                amp_m = coeffs[0] * SqueezingMatrixElement(m, 0, x_leg, index_sign) + \
                        coeffs[2] * SqueezingMatrixElement(m, 2, x_leg, index_sign)
                p_m_coherent = abs(amp_m)**2
                Bound_total += W_ext * np.sqrt(p_joint_TPM * p_m_coherent)
                
    return W_real_total, Bound_total

# =============================================================================
#  PARAMETRI E SCAN
# =============================================================================

N = 100
K = 1.0
Uf = 0.0 * K  # Quench finale verso il regime Rabi (estrazione massima)

# Range di Ui/K (da Rabi a Josephson/Fock)
Ui_range = np.geomspace(0.05, 20, 60) 

# Stato iniziale: |Psi>_- = (|0> - |2>)/sqrt(2)
c0_init = 0.5
phi_init = 0

w_vals = []
b_vals = []

for ui in Ui_range:
    w, b = calculate_metrics(ui, Uf, N, K, c0_init, phi_init)
    w_vals.append(w)
    b_vals.append(b)

# =============================================================================
#  PLOTTING
# =============================================================================

plt.figure(figsize=(10, 6))

plt.plot(Ui_range, w_vals, 'b-', label='Real Work ($W_{KDQ}$)', linewidth=2)
plt.plot(Ui_range, b_vals, 'r--', label='Classical Bound', linewidth=2)

# Evidenzia l'area di vantaggio quantistico
plt.fill_between(Ui_range, w_vals, b_vals, where=(np.array(w_vals) > np.array(b_vals)), 
                 color='green', alpha=0.2, label='Quantum Advantage')

plt.xscale('log')
plt.yscale('linear')
plt.xlabel(r'Initial Interaction $U_i / K$')
plt.ylabel('Extracted Work / Energy [K]')
plt.title(r'Work vs Bound as a function of $U_i$ (State: $|\Psi\rangle_-$)')
plt.grid(True, which="both", ls="-", alpha=0.3)
plt.legend()

plt.show()