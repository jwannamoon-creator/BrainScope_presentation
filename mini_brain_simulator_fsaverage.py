import streamlit as st
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from nilearn import datasets
from nilearn.surface import load_surf_mesh


# =========================
# Page setting
# =========================

st.set_page_config(
    page_title="Mini Brain Simulator",
    layout="wide"
)

st.title("Mini Brain Simulator")
st.write("FreeSurfer fsaverage 표준 뇌 기반 감정 회로·기억 회로 시각화 및 신경동역학 시뮬레이션")


# =========================
# Load fsaverage brain mesh
# =========================

@st.cache_data
def load_fsaverage_mesh():
    fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage5")

    left_coords, left_faces = load_surf_mesh(fsaverage.pial_left)
    right_coords, right_faces = load_surf_mesh(fsaverage.pial_right)

    return left_coords, left_faces, right_coords, right_faces


def add_fsaverage_brain(fig):
    left_coords, left_faces, right_coords, right_faces = load_fsaverage_mesh()

    fig.add_trace(go.Mesh3d(
        x=left_coords[:, 0],
        y=left_coords[:, 1],
        z=left_coords[:, 2],
        i=left_faces[:, 0],
        j=left_faces[:, 1],
        k=left_faces[:, 2],
        opacity=0.18,
        color="lightblue",
        name="Left hemisphere",
        hoverinfo="skip"
    ))

    fig.add_trace(go.Mesh3d(
        x=right_coords[:, 0],
        y=right_coords[:, 1],
        z=right_coords[:, 2],
        i=right_faces[:, 0],
        j=right_faces[:, 1],
        k=right_faces[:, 2],
        opacity=0.18,
        color="lightblue",
        name="Right hemisphere",
        hoverinfo="skip"
    ))

    return fig


# =========================
# Brain regions
# =========================

regions = {
    "PFC": {
        "pos": (-30, 45, 25),
        "desc": "전전두엽(PFC): 감정 조절, 의사결정, 작업기억"
    },
    "ACC": {
        "pos": (-5, 25, 30),
        "desc": "전대상피질(ACC): 갈등 감지, 감정 조절"
    },
    "Amygdala": {
        "pos": (-22, -5, -18),
        "desc": "편도체(Amygdala): 공포, 불안 등 감정 처리"
    },
    "Insula": {
        "pos": (-38, 0, 5),
        "desc": "섬엽(Insula): 내부 감각, 정서 인식"
    },
    "Hippocampus": {
        "pos": (-28, -25, -12),
        "desc": "해마(Hippocampus): 기억 형성, 맥락 처리"
    },
    "DG": {
        "pos": (-30, -30, -10),
        "desc": "치아이랑(DG): 패턴 분리"
    },
    "CA3": {
        "pos": (-27, -26, -12),
        "desc": "CA3: 패턴 완성"
    },
    "CA1": {
        "pos": (-24, -22, -14),
        "desc": "CA1: 기억 출력"
    }
}

emotion_regions = ["PFC", "ACC", "Amygdala", "Insula", "Hippocampus"]
memory_regions = ["DG", "CA3", "CA1", "PFC", "Hippocampus"]

emotion_edges = [
    ("PFC", "ACC"),
    ("ACC", "Amygdala"),
    ("Amygdala", "Insula"),
    ("Amygdala", "Hippocampus"),
    ("Hippocampus", "PFC")
]

memory_edges = [
    ("DG", "CA3"),
    ("CA3", "CA1"),
    ("CA1", "Hippocampus"),
    ("Hippocampus", "PFC")
]


# =========================
# Sidebar
# =========================

st.sidebar.header("설정")

selected_circuit = st.sidebar.selectbox(
    "표시할 회로",
    ["Emotion Circuit", "Memory Circuit"]
)

brain_state = st.sidebar.selectbox(
    "Brain State",
    ["Normal", "Depression-like", "Hyperexcited"]
)

input_current = st.sidebar.slider(
    "HH Input Current",
    min_value=0.0,
    max_value=20.0,
    value=10.0,
    step=0.5
)

noise_level = st.sidebar.slider(
    "LIF Noise Level",
    min_value=0.1,
    max_value=2.0,
    value=0.8,
    step=0.1
)


# =========================
# 3D Brain Viewer
# =========================

st.subheader("1. FreeSurfer fsaverage 기반 3D Brain Viewer")

if selected_circuit == "Emotion Circuit":
    active_regions = emotion_regions
    active_edges = emotion_edges
else:
    active_regions = memory_regions
    active_edges = memory_edges

fig = go.Figure()
fig = add_fsaverage_brain(fig)

# Add brain region nodes
for name in active_regions:
    x, y, z = regions[name]["pos"]

    fig.add_trace(go.Scatter3d(
        x=[x],
        y=[y],
        z=[z],
        mode="markers+text",
        marker=dict(size=8),
        text=[name],
        textposition="top center",
        name=name,
        hovertext=regions[name]["desc"],
        hoverinfo="text"
    ))

# Add circuit edges
for start, end in active_edges:
    x0, y0, z0 = regions[start]["pos"]
    x1, y1, z1 = regions[end]["pos"]

    fig.add_trace(go.Scatter3d(
        x=[x0, x1],
        y=[y0, y1],
        z=[z0, z1],
        mode="lines",
        line=dict(width=6),
        showlegend=False,
        hoverinfo="skip"
    ))

fig.update_layout(
    height=650,
    margin=dict(l=0, r=0, b=0, t=30),
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        bgcolor="rgba(0,0,0,0)",
        aspectmode="data",
        camera=dict(
            eye=dict(x=1.7, y=-2.2, z=1.1)
        )
    ),
    showlegend=True
)

st.plotly_chart(fig, use_container_width=True)

st.info(
    "현재 버전은 FreeSurfer fsaverage 표준 뇌 표면 위에 감정 회로와 기억 회로의 핵심 영역을 배치한 프로토타입입니다. "
    "좌표는 발표용 시각화를 위한 근사 위치이며, 이후 실제 atlas 좌표로 보정할 수 있습니다."
)


# =========================
# Hodgkin-Huxley model
# =========================

def run_hh(I_amp=10.0):
    dt = 0.01
    T = 100
    time = np.arange(0, T, dt)

    C_m = 1.0
    g_Na = 120.0
    g_K = 36.0
    g_L = 0.3

    E_Na = 50.0
    E_K = -77.0
    E_L = -54.387

    I_ext = np.zeros(len(time))
    I_ext[(time >= 10) & (time <= 40)] = I_amp

    def alpha_m(V):
        if abs(V + 40) < 1e-7:
            return 1.0
        return 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))

    def beta_m(V):
        return 4.0 * np.exp(-(V + 65) / 18)

    def alpha_h(V):
        return 0.07 * np.exp(-(V + 65) / 20)

    def beta_h(V):
        return 1 / (1 + np.exp(-(V + 35) / 10))

    def alpha_n(V):
        if abs(V + 55) < 1e-7:
            return 0.1
        return 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))

    def beta_n(V):
        return 0.125 * np.exp(-(V + 65) / 80)

    V = np.zeros(len(time))
    m = np.zeros(len(time))
    h = np.zeros(len(time))
    n = np.zeros(len(time))

    V[0] = -65.0
    m[0] = alpha_m(V[0]) / (alpha_m(V[0]) + beta_m(V[0]))
    h[0] = alpha_h(V[0]) / (alpha_h(V[0]) + beta_h(V[0]))
    n[0] = alpha_n(V[0]) / (alpha_n(V[0]) + beta_n(V[0]))

    for i in range(1, len(time)):
        I_Na = g_Na * (m[i - 1] ** 3) * h[i - 1] * (V[i - 1] - E_Na)
        I_K = g_K * (n[i - 1] ** 4) * (V[i - 1] - E_K)
        I_L = g_L * (V[i - 1] - E_L)

        dVdt = (I_ext[i - 1] - I_Na - I_K - I_L) / C_m
        dmdt = alpha_m(V[i - 1]) * (1 - m[i - 1]) - beta_m(V[i - 1]) * m[i - 1]
        dhdt = alpha_h(V[i - 1]) * (1 - h[i - 1]) - beta_h(V[i - 1]) * h[i - 1]
        dndt = alpha_n(V[i - 1]) * (1 - n[i - 1]) - beta_n(V[i - 1]) * n[i - 1]

        V[i] = V[i - 1] + dt * dVdt
        m[i] = m[i - 1] + dt * dmdt
        h[i] = h[i - 1] + dt * dhdt
        n[i] = n[i - 1] + dt * dndt

    return time, V, I_ext, m, h, n


# =========================
# LIF population model
# =========================

def run_lif(noise=0.8):
    dt = 0.1
    T = 500
    time = np.arange(0, T, dt)

    N = 30
    V_rest = -65
    V_reset = -65
    V_th = -50
    tau_m = 20

    V = np.ones((N, len(time))) * V_rest
    spikes = np.zeros((N, len(time)))

    I_random = np.random.normal(16, 5 * noise, size=(N, len(time)))
    I_random[I_random < 0] = 0

    for t in range(1, len(time)):
        for neuron in range(N):
            dV = (-(V[neuron, t - 1] - V_rest) + I_random[neuron, t]) / tau_m
            V[neuron, t] = V[neuron, t - 1] + dt * dV

            if V[neuron, t] >= V_th:
                spikes[neuron, t] = 1
                V[neuron, t] = V_reset

    return time, V, spikes


# =========================
# Wilson-Cowan model
# =========================

def run_wc(state="Normal"):
    dt = 0.1
    T = 100
    time = np.arange(0, T, dt)

    E = np.zeros(len(time))
    I = np.zeros(len(time))

    E[0] = 0.2
    I[0] = 0.1

    tau_E = 10
    tau_I = 20

    if state == "Normal":
        w_EE, w_EI, w_IE, w_II = 12, 10, 10, 2
        P, Q = 1.5, 0.5
    elif state == "Depression-like":
        w_EE, w_EI, w_IE, w_II = 8, 13, 8, 3
        P, Q = 0.7, 0.8
    else:
        w_EE, w_EI, w_IE, w_II = 15, 7, 12, 1
        P, Q = 2.0, 0.3

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    for t in range(1, len(time)):
        dE = (-E[t - 1] + sigmoid(w_EE * E[t - 1] - w_EI * I[t - 1] + P)) / tau_E
        dI = (-I[t - 1] + sigmoid(w_IE * E[t - 1] - w_II * I[t - 1] + Q)) / tau_I

        E[t] = E[t - 1] + dt * dE
        I[t] = I[t - 1] + dt * dI

    return time, E, I


# =========================
# Simulation panels
# =========================

st.subheader("2. 신경동역학 시뮬레이션")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Hodgkin–Huxley 단일 뉴런")
    t_hh, V_hh, I_hh, m, h, n = run_hh(input_current)

    fig_hh, ax = plt.subplots(figsize=(7, 3))
    ax.plot(t_hh, V_hh)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Membrane potential (mV)")
    ax.set_title("HH Single Neuron Spike")
    ax.grid(True)
    st.pyplot(fig_hh)

with col2:
    st.markdown("### LIF 뉴런 집단")
    t_lif, V_lif, spikes = run_lif(noise_level)

    fig_lif, ax = plt.subplots(figsize=(7, 3))
    for neuron in range(spikes.shape[0]):
        spike_times = t_lif[spikes[neuron] == 1]
        ax.scatter(spike_times, np.ones_like(spike_times) * neuron, s=5)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Neuron index")
    ax.set_title("LIF Raster Plot")
    st.pyplot(fig_lif)


col3, col4 = st.columns(2)

with col3:
    st.markdown("### Wilson–Cowan 신경집단 동역학")
    t_wc, E_wc, I_wc = run_wc(brain_state)

    fig_wc, ax = plt.subplots(figsize=(7, 3))
    ax.plot(t_wc, E_wc, label="Excitatory E")
    ax.plot(t_wc, I_wc, label="Inhibitory I")
    ax.set_xlabel("Time")
    ax.set_ylabel("Activity")
    ax.set_title(f"Wilson-Cowan Model: {brain_state}")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig_wc)

with col4:
    st.markdown("### Pseudo EEG + FFT")

    pseudo_eeg = E_wc - I_wc + np.random.normal(0, 0.05, len(E_wc))
    freqs = np.fft.rfftfreq(len(pseudo_eeg), d=0.1)
    fft_power = np.abs(np.fft.rfft(pseudo_eeg)) ** 2

    fig_eeg, ax = plt.subplots(figsize=(7, 3))
    ax.plot(t_wc, pseudo_eeg)
    ax.set_xlabel("Time")
    ax.set_ylabel("Signal")
    ax.set_title("Pseudo EEG")
    ax.grid(True)
    st.pyplot(fig_eeg)

    fig_fft, ax = plt.subplots(figsize=(7, 3))
    ax.plot(freqs, fft_power)
    ax.set_xlim(0, 40)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Power")
    ax.set_title("FFT Power Spectrum")
    ax.grid(True)
    st.pyplot(fig_fft)


# =========================
# Explanation
# =========================

st.subheader("3. 현재 구현 범위와 확장 계획")

st.write(
    """
    현재 버전은 FreeSurfer fsaverage 표준 뇌 표면 위에 감정 회로와 기억 회로를 배치하고,
    단일 뉴런 수준의 Hodgkin–Huxley 모델, 뉴런 집단 수준의 LIF 모델,
    신경집단 수준의 Wilson–Cowan 모델, 그리고 pseudo EEG 분석을 연결한 프로토타입이다.

    앞으로는 각 뇌 영역마다 Wilson–Cowan 모델을 따로 부여하고,
    감정 회로와 기억 회로의 연결 강도를 조절하여 정상 상태, 우울 유사 상태,
    기억 회로 손상 상태 등을 비교할 수 있도록 확장할 계획이다.
    """
)