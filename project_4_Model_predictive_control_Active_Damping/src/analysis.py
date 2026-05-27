# src/analysis.py
"""
Метрики и теоретические оценки для строгого доказательства сходимости.

Что считаем:
    - Энергетическая функция Ляпунова V(x) = x^T P x, где P — решение
      дискретного уравнения Риккати для пары (Ad, Bd, Q, R). По теории
      бесконечно-горизонтного LQR замкнутая система x_{k+1} = (Ad - Bd K) x_k
      удовлетворяет V(x_{k+1}) - V(x_k) = -x_k^T (Q + K^T R K) x_k ≤ 0,
      т.е. V — строгая функция Ляпунова. Если численно V(t) монотонно
      убывает — это и есть свидетельство сходимости.
    - Спектр замкнутой матрицы Acl = Ad - Bd K. Все |λ_i(Acl)| < 1 ⇔ устойчиво.
    - Теоретический показатель экспоненциальной устойчивости:
          ρ = max_i |λ_i(Acl)|       (в дискретном времени)
          α = -ln(ρ) / dt_control    (непрерывная экспонента ||x|| ≤ C e^{-α t})
    - Интегральные метрики IAE/ISE/ITAE, RMS управления, энергия.
    - Время установления по 2%-полосе от пика.
"""
import numpy as np
from scipy.linalg import solve_discrete_are


def lqr_gain_and_P(Ad, Bd, Q, R):
    P = solve_discrete_are(Ad, Bd, Q, R)
    BtPB_R = Bd.T @ P @ Bd + R
    K = np.linalg.solve(BtPB_R, Bd.T @ P @ Ad)
    return K, P


def closed_loop_spectrum(Ad, Bd, K):
    Acl = Ad - Bd @ K
    eig = np.linalg.eigvals(Acl)
    rho = float(np.max(np.abs(eig)))
    return eig, rho


def lyapunov_series(states, P):
    """V(x_k) = x_k^T P x_k для всей траектории."""
    return np.einsum('ki,ij,kj->k', states, P, states)


def state_norm(states):
    return np.linalg.norm(states, axis=1)


def settling_time(t, signal, ref=0.0, tol_frac=0.02):
    """
    Время установления: момент, после которого |signal - ref| < tol_frac * peak
    и больше не выходит за полосу.
    Возвращает t_settle или np.nan если не успокоилось до конца.
    """
    err = np.abs(signal - ref)
    peak = float(np.max(err))
    if peak <= 0:
        return float(t[0])
    band = tol_frac * peak
    # Идём с конца: ищем последний выход за полосу
    outside = err > band
    if not np.any(outside):
        return float(t[0])
    last_out = int(np.max(np.where(outside)[0]))
    if last_out >= len(t) - 1:
        return float('nan')
    return float(t[last_out + 1])


def integral_metrics(t, signal, ref=0.0):
    """IAE, ISE, ITAE."""
    err = signal - ref
    abs_err = np.abs(err)
    iae = float(np.trapezoid(abs_err, t))
    ise = float(np.trapezoid(err ** 2, t))
    itae = float(np.trapezoid(t * abs_err, t))
    return iae, ise, itae


def control_metrics(t, controls):
    rms = float(np.sqrt(np.mean(controls ** 2)))
    energy = float(np.trapezoid(controls ** 2, t))
    peak = float(np.max(np.abs(controls)))
    return rms, energy, peak


def fit_exponential_envelope(t, signal, n_peaks=None):
    """
    Подгонка экспоненциального огибающего по локальным максимумам |signal|.
    Возвращает (alpha, c0, peaks_t, peaks_y) где огибающая ≈ c0 * exp(-alpha * t).
    """
    s = np.abs(signal)
    # Локальные максимумы — простой O(n) сканер
    peaks_idx = []
    for i in range(1, len(s) - 1):
        if s[i] > s[i - 1] and s[i] >= s[i + 1] and s[i] > 1e-10:
            peaks_idx.append(i)
    peaks_idx = np.array(peaks_idx, dtype=int)
    if len(peaks_idx) < 2:
        return None
    if n_peaks is not None:
        peaks_idx = peaks_idx[:n_peaks]
    tp = t[peaks_idx]
    yp = s[peaks_idx]
    # log(yp) = log(c0) - alpha * tp  →  линейная регрессия
    log_y = np.log(yp)
    A = np.vstack([tp, np.ones_like(tp)]).T
    sol, *_ = np.linalg.lstsq(A, log_y, rcond=None)
    slope, intercept = sol
    alpha = -float(slope)
    c0 = float(np.exp(intercept))
    return {'alpha': alpha, 'c0': c0, 'peaks_t': tp, 'peaks_y': yp}


def all_metrics(data, label='ctrl'):
    t = data['t']
    x1 = data['states'][:, 0]
    x2 = data['states'][:, 1]
    u = data['controls']
    ref = data['ref']

    iae1, ise1, itae1 = integral_metrics(t, x1, ref[0])
    iae2, ise2, itae2 = integral_metrics(t, x2, ref[1])
    rms_u, energy_u, peak_u = control_metrics(t, u)
    ts1 = settling_time(t, x1, ref[0])
    overshoot = float(np.max(np.abs(x1 - ref[0])))

    return {
        'label': label,
        'IAE_x1': iae1, 'ISE_x1': ise1, 'ITAE_x1': itae1,
        'IAE_x2': iae2, 'ISE_x2': ise2, 'ITAE_x2': itae2,
        'RMS_u': rms_u, 'energy_u': energy_u, 'peak_u': peak_u,
        'settling_time_x1': ts1,
        'peak_x1': overshoot,
    }
