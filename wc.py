import numpy as np


def run_wc_network(regions, state="Normal", W_external=None):
    dt, T = 0.1, 100
    time = np.arange(0, T, dt)

    n_regions = len(regions)
    E = np.zeros((n_regions, len(time)))
    I = np.zeros((n_regions, len(time)))

    E[:, 0] = 0.2
    I[:, 0] = 0.1

    tau_E, tau_I = 10, 20

    if state == "Normal":
        w_EE, w_EI, w_IE, w_II = 12, 10, 10, 2
        P, Q = 1.5, 0.5
    elif state == "Depression-like":
        w_EE, w_EI, w_IE, w_II = 8, 13, 8, 3
        P, Q = 0.7, 0.8
    else:
        w_EE, w_EI, w_IE, w_II = 15, 7, 12, 1
        P, Q = 2.0, 0.3

    if W_external is None:
        W = np.ones((n_regions, n_regions)) * 0.15
        np.fill_diagonal(W, 0)
    else:
        W = W_external

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    for t in range(1, len(time)):
        network_input = W @ E[:, t - 1]

        dE = (
            -E[:, t - 1]
            + sigmoid(w_EE * E[:, t - 1] - w_EI * I[:, t - 1] + P + network_input)
        ) / tau_E

        dI = (
            -I[:, t - 1]
            + sigmoid(w_IE * E[:, t - 1] - w_II * I[:, t - 1] + Q)
        ) / tau_I

        E[:, t] = E[:, t - 1] + dt * dE
        I[:, t] = I[:, t - 1] + dt * dI

    return time, E, I