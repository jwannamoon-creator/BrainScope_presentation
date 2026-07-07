import numpy as np

def make_pseudo_eeg(E, I, noise=0.05):
    signal = E.mean(axis=0) - I.mean(axis=0)
    signal = signal + np.random.normal(0, noise, len(signal))

    freqs = np.fft.rfftfreq(len(signal), d=0.1)
    power = np.abs(np.fft.rfft(signal)) ** 2

    return signal, freqs, power