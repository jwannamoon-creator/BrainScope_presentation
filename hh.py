import numpy as np

def run_hh(I_amp=10.0):
    dt, T = 0.01, 100
    time = np.arange(0, T, dt)

    C_m = 1.0
    g_Na, g_K, g_L = 120.0, 36.0, 0.3
    E_Na, E_K, E_L = 50.0, -77.0, -54.387

    I_ext = np.zeros(len(time))
    I_ext[(time >= 10) & (time <= 40)] = I_amp

    def alpha_m(V):
        return 1.0 if abs(V + 40) < 1e-7 else 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))

    def beta_m(V): return 4.0 * np.exp(-(V + 65) / 18)
    def alpha_h(V): return 0.07 * np.exp(-(V + 65) / 20)
    def beta_h(V): return 1 / (1 + np.exp(-(V + 35) / 10))

    def alpha_n(V):
        return 0.1 if abs(V + 55) < 1e-7 else 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))

    def beta_n(V): return 0.125 * np.exp(-(V + 65) / 80)

    V = np.zeros(len(time))
    m = np.zeros(len(time))
    h = np.zeros(len(time))
    n = np.zeros(len(time))

    V[0] = -65.0
    m[0] = alpha_m(V[0]) / (alpha_m(V[0]) + beta_m(V[0]))
    h[0] = alpha_h(V[0]) / (alpha_h(V[0]) + beta_h(V[0]))
    n[0] = alpha_n(V[0]) / (alpha_n(V[0]) + beta_n(V[0]))

    for i in range(1, len(time)):
        I_Na = g_Na * (m[i-1] ** 3) * h[i-1] * (V[i-1] - E_Na)
        I_K = g_K * (n[i-1] ** 4) * (V[i-1] - E_K)
        I_L = g_L * (V[i-1] - E_L)

        V[i] = V[i-1] + dt * (I_ext[i-1] - I_Na - I_K - I_L) / C_m
        m[i] = m[i-1] + dt * (alpha_m(V[i-1]) * (1 - m[i-1]) - beta_m(V[i-1]) * m[i-1])
        h[i] = h[i-1] + dt * (alpha_h(V[i-1]) * (1 - h[i-1]) - beta_h(V[i-1]) * h[i-1])
        n[i] = n[i-1] + dt * (alpha_n(V[i-1]) * (1 - n[i-1]) - beta_n(V[i-1]) * n[i-1])

    return time, V, I_ext