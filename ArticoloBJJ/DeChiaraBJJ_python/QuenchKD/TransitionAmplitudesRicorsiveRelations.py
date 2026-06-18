import numpy as np
from scipy.special import lpmv, gammaln

def TransitionAmplitude(m, n, x, index_sign):
    # La funzione lavora su array x, quindi gestiamo i casi impossibili
    if (m + n) % 2 != 0: return np.zeros_like(x)
    l = (m + n) // 2
    k = (m - n) // 2
    
    legendrevalue = lpmv(abs(k), l, x)
    log_pref = 0.5 * (gammaln(min(n, m) + 1) - gammaln(max(n, m) + 1))
    
    # Fase come da definizione del paper
    phase = np.where(m >= n, (index_sign)**abs(k), (-index_sign)**abs(k))
        
    return np.exp(log_pref) * np.sqrt(x) * legendrevalue * phase

def paper_relation_35(x, s, index_sign):
    # Relazione 35 (con la correzione del denominatore discussa)
    return index_sign * (1 - x**2 * (1 + 2*s)) / np.sqrt((2 * (1 - x**2)))

# Parametri
x = np.linspace(0.1, 0.9, 50) # Range positivo per evitare radici complesse
omega_f = 1.0
omega_i = 2.0
index_sign = np.sign(omega_i - omega_f)

print(f"{'m':<5} | {'s':<5} | {'Max Diff':<20}")
print("-" * 35)

# Verifichiamo per m = 2, 4, 6, ..., 12
for m in range(2, 14, 2):
    s = m // 2
    
    # Rapporto numerico (Eq 35: Lambda_{2s,2} / Lambda_{2s,0})
    # Nota: m=2s, quindi Lambda_{2s,2} / Lambda_{2s,0}
    num = TransitionAmplitude(m, 2, x, index_sign)
    den = TransitionAmplitude(m, 0, x, index_sign)
    
    # Evitiamo divisioni per zero se den è molto piccolo
    ratio_num = np.divide(num, den, out=np.zeros_like(num), where=den!=0)
    
    # Rapporto teorico (35)
    ratio_th = paper_relation_35(x, s, index_sign)
    
    # Calcolo differenza massima
    diff = np.max(np.abs(ratio_num - ratio_th))
    
    print(f"{m:<5} | {s:<5} | {diff:<20.2e}")

    # Verifica soglia di precisione
    if diff > 1e-12:
        print(f"   ! Discrepanza rilevata per m={m}")