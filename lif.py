import numpy as np

def run_lif(noise=0.8, N=30):
    dt, T = 0.1, 500
    time = np.arange(0, T, dt)

    V_rest, V_reset, V_th = -65, -65, -50
    tau_m = 20

    V = np.ones((N, len(time))) * V_rest
    spikes = np.zeros((N, len(time)))

    I_random = np.random.normal(16, 5 * noise, size=(N, len(time)))
    I_random[I_random < 0] = 0

    for t in range(1, len(time)):
        dV = (-(V[:, t-1] - V_rest) + I_random[:, t]) / tau_m
        V[:, t] = V[:, t-1] + dt * dV

        fired = V[:, t] >= V_th
        spikes[fired, t] = 1
        V[fired, t] = V_reset

    return time, V, spikes