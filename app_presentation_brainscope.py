import time

import pandas as pd
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from scipy.signal import welch, butter, sosfiltfilt

from streamlit_autorefresh import st_autorefresh
from nilearn import datasets
from nilearn.surface import load_surf_mesh

from regions import REGIONS, CIRCUITS
from connectivity import (
    load_connectivity_matrix,
    load_group_difference_matrix,
    scale_matrix_for_wc,
)
from wc import run_wc_network
from hh import run_hh
from lif import run_lif


# ============================================================
# BrainScope: Presentation Edition
# 발표용 버전: 시각화, 직관성, 안정성 우선
# ============================================================

st.set_page_config(
    page_title="BrainScope Presentation Edition",
    layout="wide"
)

st.title("BrainScope")
st.write("실제 resting-state fMRI 기반 감정·기억 회로 Brain Activity Simulator")


# ============================================================
# Load FreeSurfer fsaverage brain mesh
# ============================================================

@st.cache_data
def load_fsaverage_mesh():
    fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
    left_coords, left_faces = load_surf_mesh(fsaverage.pial_left)
    right_coords, right_faces = load_surf_mesh(fsaverage.pial_right)
    return left_coords, left_faces, right_coords, right_faces


def add_fsaverage_brain(fig):
    left_coords, left_faces, right_coords, right_faces = load_fsaverage_mesh()

    for coords, faces in [
        (left_coords, left_faces),
        (right_coords, right_faces)
    ]:
        fig.add_trace(go.Mesh3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            opacity=0.42,
            color="lightgray",
            hoverinfo="skip",
            showlegend=False
        ))

    return fig


# ============================================================
# Utility functions
# ============================================================

def get_dataset_id(label):
    if label == "Healthy":
        return "control_mean"
    if label == "Depression":
        return "depression_mean"
    return "control_mean"


def get_top_connection_changes(diff_sub, top_n=5):
    changes = []
    regions = list(diff_sub.index)

    for i in range(len(regions)):
        for j in range(i + 1, len(regions)):
            r1 = regions[i]
            r2 = regions[j]
            diff = float(diff_sub.loc[r1, r2])
            changes.append({
                "Connection": f"{r1} - {r2}",
                "Difference": diff,
                "Direction": "Depression ↑" if diff > 0 else "Depression ↓",
                "AbsDifference": abs(diff),
            })

    changes = sorted(changes, key=lambda x: x["AbsDifference"], reverse=True)
    return changes[:top_n]


def build_connection_submatrix(dataset_id, active_regions):
    conn_df = load_connectivity_matrix(dataset_id, "correlation")
    available_regions = [r for r in active_regions if r in conn_df.index and r in conn_df.columns]

    if len(available_regions) < 2:
        raise ValueError("회로 ROI와 connectivity matrix의 영역 이름이 충분히 일치하지 않습니다.")

    conn_sub = conn_df.loc[available_regions, available_regions]
    W_external, wc_regions = scale_matrix_for_wc(conn_sub, scale=0.35)

    return conn_sub, W_external, wc_regions


def draw_brain(
    active_regions,
    active_edges,
    selected_region,
    region_activity,
    activity_time,
    mode,
    conn_sub=None,
    diff_sub=None,
):
    fig = go.Figure()
    fig = add_fsaverage_brain(fig)

    # Edges
    for start, end in active_edges:
        if start not in REGIONS or end not in REGIONS:
            continue

        x0, y0, z0 = REGIONS[start]["pos"]
        x1, y1, z1 = REGIONS[end]["pos"]

        selected_edge = start == selected_region or end == selected_region

        if mode == "Difference Network" and diff_sub is not None:
            diff_value = 0.0
            if start in diff_sub.index and end in diff_sub.columns:
                diff_value = float(diff_sub.loc[start, end])

            line_width = 4 + 90 * abs(diff_value)
            if diff_value > 0:
                line_color = "crimson"
            elif diff_value < 0:
                line_color = "royalblue"
            else:
                line_color = "lightgray"

            hover = (
                f"{start} - {end}<br>"
                f"Depression - Healthy: {diff_value:.3f}<br>"
                f"Red = stronger in depression<br>"
                f"Blue = weaker in depression"
            )

        else:
            weight = 0.25
            if conn_sub is not None and start in conn_sub.index and end in conn_sub.columns:
                weight = float(conn_sub.loc[start, end])

            line_width = 3 + 10 * weight
            line_color = "crimson" if selected_edge else "gray"
            hover = f"{start} - {end}<br>Connectivity: {weight:.3f}"

        fig.add_trace(go.Scatter3d(
            x=[x0, x1],
            y=[y0, y1],
            z=[z0, z1],
            mode="lines",
            line=dict(width=line_width, color=line_color),
            hovertext=hover,
            hoverinfo="text",
            showlegend=False
        ))

    # Nodes
    for idx, name in enumerate(active_regions):
        if name not in REGIONS:
            continue

        x, y, z = REGIONS[name]["pos"]
        activity = float(region_activity[idx])
        is_selected = name == selected_region

        if mode == "Difference Network":
            node_color = "crimson" if is_selected else "royalblue"
            node_size = 20 if is_selected else 12
            colorbar = None
        else:
            node_color = activity
            node_size = (16 + 18 * activity) if is_selected else (9 + 14 * activity)
            colorbar = dict(title="E activity", thickness=12, len=0.55, x=0.93) if idx == 0 else None

        marker = dict(
            size=node_size,
            color=node_color,
            symbol="diamond" if is_selected else "circle",
            opacity=0.96,
            line=dict(width=2, color="white")
        )

        if mode != "Difference Network":
            marker.update(
                colorscale="Turbo",
                cmin=0,
                cmax=1,
                colorbar=colorbar
            )

        fig.add_trace(go.Scatter3d(
            x=[x],
            y=[y],
            z=[z],
            mode="markers+text",
            marker=marker,
            text=[name],
            textposition="top center",
            name=name,
            hovertext=(
                f"<b>{name}</b><br>"
                f"{REGIONS[name]['desc']}<br>"
                f"E activity at t={activity_time:.1f}: {activity:.3f}"
            ),
            hoverinfo="text",
            showlegend=False
        ))

    fig.update_layout(
        height=670,
        margin=dict(l=0, r=0, b=0, t=30),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor="white",
            aspectmode="data",
            camera=dict(eye=dict(x=1.7, y=-2.2, z=1.1))
        ),
        showlegend=False
    )

    return fig


# ============================================================
# Pseudo EEG v2
# 영역별 E/I 활동 + 회로별 가중치 + 상태별 스펙트럼 보정
# ============================================================

PSEUDO_EEG_BANDS = {
    "Delta": (1.0, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 13.0),
    "Beta": (13.0, 30.0),
    "Gamma": (30.0, 40.0),
}

# 실제 EEG 두 사례에서 얻은 상대 대역 파워를 모델 보정값으로 사용함.
# 따라서 v2의 결과는 독립적 임상 검증이 아니라 calibration 결과로 해석해야 함.
STATE_BAND_PROFILE = {
    "Healthy": {
        "Delta": 0.248595,
        "Theta": 0.103795,
        "Alpha": 0.258479,
        "Beta": 0.249581,
        "Gamma": 0.114941,
    },
    "Depression": {
        "Delta": 0.283521,
        "Theta": 0.101867,
        "Alpha": 0.444172,
        "Beta": 0.126937,
        "Gamma": 0.020432,
    },
}

# 상태 차이를 유지하면서 감정 회로와 기억 회로가 완전히 같은
# 스펙트럼을 생성하지 않도록 작은 회로별 변조를 적용함.
CIRCUIT_BAND_MULTIPLIER = {
    "Emotion Circuit": {
        "Delta": 1.06,
        "Theta": 1.10,
        "Alpha": 0.90,
        "Beta": 1.06,
        "Gamma": 1.03,
    },
    "Memory Circuit": {
        "Delta": 0.94,
        "Theta": 1.08,
        "Alpha": 1.10,
        "Beta": 0.96,
        "Gamma": 0.94,
    },
}

# 회로마다 기여도가 높은 영역을 다르게 가중함.
REGION_KEYWORD_WEIGHTS = {
    "Emotion Circuit": {
        "amygdala": 1.45,
        "insula": 1.25,
        "acc": 1.25,
        "pfc": 1.15,
        "ofc": 1.15,
        "hippocampus": 0.90,
        "pcc": 0.85,
    },
    "Memory Circuit": {
        "hippocampus": 1.50,
        "dg": 1.35,
        "ca3": 1.35,
        "ca1": 1.35,
        "parahippocampal": 1.25,
        "pcc": 1.05,
        "angular": 1.00,
        "pfc": 0.90,
        "amygdala": 0.80,
    },
}


def _zscore_signal(signal):
    signal = np.asarray(signal, dtype=float)
    signal = signal - np.mean(signal)

    standard_deviation = float(np.std(signal))
    if standard_deviation > 1e-12:
        signal = signal / standard_deviation

    return signal


def _get_region_weights(region_names, circuit):
    keyword_weights = REGION_KEYWORD_WEIGHTS[circuit]
    weights = np.ones(len(region_names), dtype=float)

    for index, region_name in enumerate(region_names):
        normalized_name = str(region_name).lower().replace(" ", "")
        matched_weights = []

        for keyword, weight in keyword_weights.items():
            normalized_keyword = keyword.lower().replace(" ", "")
            if normalized_keyword in normalized_name:
                matched_weights.append(weight)

        if matched_weights:
            weights[index] = max(matched_weights)

    weights = weights / np.sum(weights)
    return weights


def _make_band_limited_noise(
    n_samples,
    sfreq,
    low_frequency,
    high_frequency,
    rng,
):
    white_noise = rng.normal(0.0, 1.0, n_samples)

    nyquist = sfreq / 2.0
    low_normalized = max(low_frequency / nyquist, 1e-5)
    high_normalized = min(high_frequency / nyquist, 0.999)

    if not 0 < low_normalized < high_normalized < 1:
        raise ValueError(
            f"{low_frequency}-{high_frequency} Hz 대역을 "
            f"{sfreq} Hz 샘플링 주파수에서 생성할 수 없습니다."
        )

    sos = butter(
        4,
        [low_normalized, high_normalized],
        btype="bandpass",
        output="sos",
    )

    filtered_noise = sosfiltfilt(sos, white_noise)
    return _zscore_signal(filtered_noise)


def _get_target_band_profile(brain_state, circuit):
    profile = {}

    for band_name in PSEUDO_EEG_BANDS:
        profile[band_name] = (
            STATE_BAND_PROFILE[brain_state][band_name]
            * CIRCUIT_BAND_MULTIPLIER[circuit][band_name]
        )

    total_power = sum(profile.values())

    return {
        band_name: band_power / total_power
        for band_name, band_power in profile.items()
    }


def _get_condition_seed(brain_state, circuit, random_state):
    seed_offset = {
        ("Healthy", "Emotion Circuit"): 101,
        ("Healthy", "Memory Circuit"): 211,
        ("Depression", "Emotion Circuit"): 307,
        ("Depression", "Memory Circuit"): 401,
    }

    return int(
        random_state
        + seed_offset[(brain_state, circuit)]
    )


def make_pseudo_eeg_100hz(
    t,
    E,
    I,
    region_names,
    circuit,
    brain_state,
    sfreq=100.0,
    noise=0.025,
    random_state=42,
    wc_mix=0.18,
):
    """
    BrainScope pseudo EEG v2.

    Wilson-Cowan의 영역별 E/I 활동을 회로별로 가중합하고,
    Healthy/Depression 상태별 대역 파워와 감정/기억 회로별
    스펙트럼 변조를 반영하여 100 Hz 모의 EEG를 생성한다.

    반환값
    -------
    pseudo_time
    pseudo_eeg
    freqs
    power
    target_profile
    """

    t = np.asarray(t, dtype=float)
    E = np.asarray(E, dtype=float)
    I = np.asarray(I, dtype=float)
    region_names = list(region_names)

    if t.ndim != 1 or len(t) < 3:
        raise ValueError("t는 길이 3 이상인 1차원 시간 배열이어야 합니다.")

    if E.ndim != 2 or I.ndim != 2:
        raise ValueError(
            "E와 I는 (영역 수, 시간점 수) 형태의 2차원 배열이어야 합니다."
        )

    if E.shape != I.shape:
        raise ValueError("E와 I의 배열 크기가 서로 같아야 합니다.")

    if E.shape[0] != len(region_names):
        raise ValueError(
            "region_names의 길이는 E/I의 영역 수와 같아야 합니다."
        )

    if E.shape[1] != len(t):
        raise ValueError(
            "E/I의 시간축 길이와 t의 길이가 일치해야 합니다."
        )

    if circuit not in CIRCUIT_BAND_MULTIPLIER:
        raise ValueError(
            f"지원하지 않는 회로입니다: {circuit}"
        )

    if brain_state not in STATE_BAND_PROFILE:
        raise ValueError(
            f"지원하지 않는 뇌 상태입니다: {brain_state}"
        )

    if sfreq <= 80:
        raise ValueError(
            "40 Hz까지 분석하려면 sfreq는 80 Hz보다 커야 합니다."
        )

    duration = float(t[-1] - t[0])
    if duration <= 0:
        raise ValueError("기록 길이는 0초보다 커야 합니다.")

    pseudo_time = np.arange(
        float(t[0]),
        float(t[-1]) + 0.5 / sfreq,
        1.0 / sfreq,
    )

    # --------------------------------------------------------
    # 1. 회로별 영역 가중 Wilson-Cowan source
    # --------------------------------------------------------

    region_weights = _get_region_weights(
        region_names,
        circuit,
    )

    weighted_ei = np.sum(
        region_weights[:, None]
        * (E - 0.72 * I),
        axis=0,
    )

    weighted_e = np.sum(
        region_weights[:, None] * E,
        axis=0,
    )

    wc_source = (
        0.78 * weighted_ei
        + 0.22 * weighted_e
    )

    interpolated_wc = np.interp(
        pseudo_time,
        t,
        wc_source,
    )
    interpolated_wc = _zscore_signal(interpolated_wc)

    # Wilson-Cowan 흥분성 활동을 느린 진폭 envelope로 반영
    interpolated_e = np.interp(
        pseudo_time,
        t,
        weighted_e,
    )
    interpolated_e = _zscore_signal(interpolated_e)

    envelope = (
        1.0
        + 0.14 * np.tanh(interpolated_e)
    )

    # --------------------------------------------------------
    # 2. 상태·회로별 목표 대역 파워
    # --------------------------------------------------------

    target_profile = _get_target_band_profile(
        brain_state,
        circuit,
    )

    condition_seed = _get_condition_seed(
        brain_state,
        circuit,
        random_state,
    )
    rng = np.random.default_rng(condition_seed)

    spectral_signal = np.zeros(
        len(pseudo_time),
        dtype=float,
    )

    for band_name, (
        low_frequency,
        high_frequency,
    ) in PSEUDO_EEG_BANDS.items():

        band_component = _make_band_limited_noise(
            n_samples=len(pseudo_time),
            sfreq=sfreq,
            low_frequency=low_frequency,
            high_frequency=high_frequency,
            rng=rng,
        )

        # 파워는 진폭의 제곱에 비례하므로 sqrt를 적용함.
        amplitude = np.sqrt(
            target_profile[band_name]
        )

        spectral_signal += (
            amplitude * band_component
        )

    spectral_signal = (
        envelope * spectral_signal
    )
    spectral_signal = _zscore_signal(
        spectral_signal
    )

    # --------------------------------------------------------
    # 3. 회로별 위상 및 상태별 중심주파수 차이
    # --------------------------------------------------------

    phase_shift = {
        "Emotion Circuit": 0.35,
        "Memory Circuit": 0.95,
    }[circuit]

    alpha_frequency = (
        10.0
        if brain_state == "Healthy"
        else 9.5
    )

    rhythmic_component = (
        0.10
        * np.sin(
            2.0
            * np.pi
            * alpha_frequency
            * pseudo_time
            + phase_shift
        )
        + 0.045
        * np.sin(
            2.0
            * np.pi
            * 6.0
            * pseudo_time
            + 0.5 * phase_shift
        )
    )

    # --------------------------------------------------------
    # 4. 최종 pseudo EEG
    # --------------------------------------------------------

    pseudo_eeg = (
        np.sqrt(max(0.0, 1.0 - wc_mix ** 2))
        * spectral_signal
        + wc_mix * interpolated_wc
        + rhythmic_component
        + rng.normal(
            0.0,
            noise,
            len(pseudo_time),
        )
    )

    pseudo_eeg = _zscore_signal(pseudo_eeg)

    # 4초 Welch 창, 50% 중첩
    nperseg = min(
        int(round(4.0 * sfreq)),
        len(pseudo_eeg),
    )
    noverlap = nperseg // 2

    freqs, power = welch(
        pseudo_eeg,
        fs=sfreq,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        scaling="density",
    )

    frequency_mask = (
        (freqs >= 0.0)
        & (freqs <= 40.0)
    )

    return (
        pseudo_time,
        pseudo_eeg,
        freqs[frequency_mask],
        power[frequency_mask],
        target_profile,
    )


# ============================================================
# Sidebar: Presentation UI
# ============================================================

st.sidebar.header("Presentation Controls")

selected_circuit = st.sidebar.radio(
    "Circuit",
    ["Emotion Circuit", "Memory Circuit"]
)

brain_dataset_label = st.sidebar.radio(
    "Brain State",
    ["Healthy", "Depression"]
)

view_mode = st.sidebar.radio(
    "View",
    ["Brain Activity Map", "Difference Network"]
)

active_regions_original = CIRCUITS[selected_circuit]["regions"]
active_edges_original = CIRCUITS[selected_circuit]["edges"]

selected_region = st.sidebar.selectbox(
    "Selected Region",
    active_regions_original
)

st.sidebar.divider()

activity_time_index = st.sidebar.slider(
    "Brain Activity Time",
    min_value=0,
    max_value=999,
    value=999,
    step=10
)

auto_play = st.sidebar.checkbox("▶ Auto Play", value=False)
play_speed = st.sidebar.selectbox(
    "Animation Speed",
    ["Slow", "Normal", "Fast"],
    index=1
)

speed_map = {
    "Slow": 400,
    "Normal": 200,
    "Fast": 100
}

st.sidebar.caption("발표용 버전은 시각화와 설명 안정성을 위해 옵션을 단순화했습니다.")


# ============================================================
# Load connectivity and run Wilson-Cowan
# ============================================================

try:
    active_regions = active_regions_original.copy()
    active_edges = active_edges_original.copy()

    if view_mode == "Difference Network":
        diff_df, control_df, depression_df = load_group_difference_matrix("correlation")
        available_regions = [r for r in active_regions if r in diff_df.index and r in diff_df.columns]

        if len(available_regions) < 2:
            raise ValueError("Difference matrix와 현재 회로 ROI가 충분히 일치하지 않습니다.")

        diff_sub = diff_df.loc[available_regions, available_regions]
        depression_sub = depression_df.loc[available_regions, available_regions]
        W_external, wc_regions = scale_matrix_for_wc(depression_sub, scale=0.35)

        conn_sub = depression_sub
        active_regions = wc_regions
        active_edges = [
            edge for edge in active_edges
            if edge[0] in active_regions and edge[1] in active_regions
        ]

        t_wc, E_wc, I_wc = run_wc_network(
            active_regions,
            "Normal",
            W_external=W_external
        )

        status_message = "Difference Network: Depression Mean - Healthy Mean을 표시합니다."

    else:
        dataset_id = get_dataset_id(brain_dataset_label)
        conn_sub, W_external, wc_regions = build_connection_submatrix(dataset_id, active_regions)

        diff_sub = None
        active_regions = wc_regions
        active_edges = [
            edge for edge in active_edges
            if edge[0] in active_regions and edge[1] in active_regions
        ]

        t_wc, E_wc, I_wc = run_wc_network(
            active_regions,
            "Normal",
            W_external=W_external
        )

        status_message = f"{brain_dataset_label} Mean connectivity를 Wilson–Cowan 모델에 적용했습니다."

except Exception as e:
    st.warning(f"Connectivity matrix를 불러오지 못해 기본 연결 행렬을 사용합니다: {e}")

    active_regions = active_regions_original.copy()
    active_edges = active_edges_original.copy()
    diff_sub = None
    conn_sub = None

    t_wc, E_wc, I_wc = run_wc_network(active_regions, "Normal")
    status_message = "기본 연결 행렬을 사용합니다."


if selected_region not in active_regions:
    selected_region = active_regions[0]

selected_index = active_regions.index(selected_region)
activity_time_index = min(activity_time_index, E_wc.shape[1] - 1)
activity_time = t_wc[activity_time_index]

region_activity = E_wc[:, activity_time_index]
selected_E = E_wc[selected_index]
selected_I = I_wc[selected_index]


# Auto Play: Streamlit rerun-based simple animation
if "play_time_index" not in st.session_state:
    st.session_state.play_time_index = activity_time_index

if auto_play:
    st_autorefresh(
        interval=speed_map[play_speed],
        key="brain_activity_autoplay"
    )

    st.session_state.play_time_index += 10

    if st.session_state.play_time_index >= E_wc.shape[1]:
        st.session_state.play_time_index = 0

    activity_time_index = st.session_state.play_time_index
else:
    st.session_state.play_time_index = activity_time_index

activity_time_index = min(activity_time_index, E_wc.shape[1] - 1)
activity_time = t_wc[activity_time_index]

region_activity = E_wc[:, activity_time_index]
selected_E = E_wc[selected_index]
selected_I = I_wc[selected_index]

# ============================================================
# Main presentation layout
# ============================================================

st.success(status_message)

top_col1, top_col2, top_col3 = st.columns(3)
with top_col1:
    st.metric("Circuit", selected_circuit.replace(" Circuit", ""))
with top_col2:
    st.metric("Brain State", brain_dataset_label if view_mode != "Difference Network" else "Difference")
with top_col3:
    st.metric("Time", f"{activity_time:.1f}")

st.subheader("1. 3D Brain Activity Map")

brain_fig = draw_brain(
    active_regions=active_regions,
    active_edges=active_edges,
    selected_region=selected_region,
    region_activity=region_activity,
    activity_time=activity_time,
    mode=view_mode,
    conn_sub=conn_sub,
    diff_sub=diff_sub
)

st.plotly_chart(brain_fig, use_container_width=True)

if view_mode == "Brain Activity Map":
    st.caption(
        "노드의 색과 크기는 Wilson–Cowan 모델에서 계산된 각 뇌 영역의 흥분성 activity E(t)를 의미합니다."
    )
else:
    st.caption(
        "빨간 연결은 우울증군에서 더 강한 연결, 파란 연결은 우울증군에서 더 약한 연결을 의미합니다."
    )


# ============================================================
# Selected region + Dynamics
# ============================================================

st.subheader("2. Selected Region Dynamics")

info_col, graph_col = st.columns([1, 2])

with info_col:
    st.markdown(f"### {selected_region}")
    st.write(REGIONS[selected_region]["desc"])

    st.metric(
        f"E activity at t={activity_time:.1f}",
        f"{selected_E[activity_time_index]:.3f}"
    )
    st.metric(
        f"I activity at t={activity_time:.1f}",
        f"{selected_I[activity_time_index]:.3f}"
    )

    connected_regions = []
    for start, end in active_edges:
        if start == selected_region:
            connected_regions.append(end)
        elif end == selected_region:
            connected_regions.append(start)

    st.write("**Connected regions:**")
    st.write(", ".join(connected_regions) if connected_regions else "없음")

    if view_mode == "Difference Network" and diff_sub is not None:
        st.write("**Depression - Healthy changes:**")
        changes = []
        for other in active_regions:
            if other == selected_region:
                continue
            if selected_region in diff_sub.index and other in diff_sub.columns:
                changes.append((other, float(diff_sub.loc[selected_region, other])))

        changes = sorted(changes, key=lambda x: abs(x[1]), reverse=True)
        for other, value in changes:
            if value > 0:
                st.write(f"▲ {other}: +{value:.3f}")
            elif value < 0:
                st.write(f"▼ {other}: {value:.3f}")
            else:
                st.write(f"- {other}: 0.000")

with graph_col:
    fig_region, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(t_wc, selected_E, label=f"{selected_region} E")
    ax.plot(t_wc, selected_I, label=f"{selected_region} I")
    ax.axvline(activity_time, linestyle="--", alpha=0.8)
    ax.set_xlabel("Time")
    ax.set_ylabel("Activity")
    ax.set_title(f"{selected_region} Wilson–Cowan E/I Dynamics")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig_region)


# ============================================================
# Difference analysis for presentation
# ============================================================

if view_mode == "Difference Network" and diff_sub is not None:
    st.subheader("3. Healthy vs Depression Difference")

    top_changes = get_top_connection_changes(diff_sub, top_n=5)

    table_col, heatmap_col = st.columns([1, 1.2])

    with table_col:
        st.markdown("### Top 5 Changed Connections")
        st.dataframe(
            [
                {
                    "Connection": c["Connection"],
                    "Difference": round(c["Difference"], 3),
                    "Direction": c["Direction"],
                }
                for c in top_changes
            ],
            use_container_width=True
        )

    with heatmap_col:
        scaled_diff = diff_sub.values * 5.0
        vmax = np.max(np.abs(scaled_diff))
        if vmax == 0:
            vmax = 1.0

        fig_diff, ax = plt.subplots(figsize=(6.5, 5.5))
        im = ax.imshow(
            scaled_diff,
            cmap="bwr",
            vmin=-vmax,
            vmax=vmax
        )
        ax.set_xticks(range(len(diff_sub.columns)))
        ax.set_xticklabels(diff_sub.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(diff_sub.index)))
        ax.set_yticklabels(diff_sub.index)
        ax.set_title("Depression - Healthy Difference Matrix")
        plt.colorbar(im, ax=ax, label="Scaled difference")
        plt.tight_layout()
        st.pyplot(fig_diff)

else:
    st.subheader("3. Circuit Activity Overview")

    fig_network, ax = plt.subplots(figsize=(9, 3.2))
    for idx, name in enumerate(active_regions):
        if name == selected_region:
            ax.plot(t_wc, E_wc[idx], linewidth=3, label=f"{name} selected")
        else:
            ax.plot(t_wc, E_wc[idx], alpha=0.65, label=name)

    ax.axvline(activity_time, linestyle="--", alpha=0.8)
    ax.set_xlabel("Time")
    ax.set_ylabel("Excitatory activity")
    ax.set_title(f"{selected_circuit} - {brain_dataset_label}")
    ax.legend(fontsize=8)
    ax.grid(True)
    st.pyplot(fig_network)


# ============================================================
# Pseudo EEG
# ============================================================

st.subheader("4. Pseudo EEG")

PSEUDO_EEG_SFREQ = 100.0

(
    pseudo_eeg_time,
    pseudo_eeg,
    freqs,
    power,
    target_profile,
) = make_pseudo_eeg_100hz(
    t=t_wc,
    E=E_wc,
    I=I_wc,
    region_names=active_regions,
    circuit=selected_circuit,
    brain_state=brain_dataset_label,
    sfreq=PSEUDO_EEG_SFREQ,
    noise=0.025,
    random_state=42,
    wc_mix=0.18,
)

eeg_col1, eeg_col2 = st.columns(2)

with eeg_col1:
    fig_eeg, ax = plt.subplots(figsize=(7, 3))

    ax.plot(
        pseudo_eeg_time,
        pseudo_eeg,
        linewidth=1.0,
    )

    ax.axvline(
        activity_time,
        linestyle="--",
        alpha=0.7,
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Signal")
    ax.set_title(
        f"Pseudo EEG Raw Signal ({PSEUDO_EEG_SFREQ:.0f} Hz)"
    )
    ax.grid(True)

    st.pyplot(fig_eeg)
    plt.close(fig_eeg)

with eeg_col2:
    fig_fft, ax = plt.subplots(figsize=(7, 3))

    ax.plot(
        freqs,
        power,
        linewidth=1.0,
    )

    ax.set_xlim(0, 40)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power spectral density")
    ax.set_title("Welch Power Spectrum")
    ax.grid(True)

    st.pyplot(fig_fft)
    plt.close(fig_fft)

st.caption(
    "Pseudo EEG v2는 Wilson–Cowan의 영역별 E/I 활동, 회로별 영역 가중치, "
    "상태별 대역 파워 보정을 결합한 모의 신호입니다. 상태별 보정값은 현재 "
    "실제 EEG 사례에서 추정했으므로 독립적 임상 검증이 아닌 calibration "
    "결과로 해석해야 합니다."
)

with st.expander("Pseudo EEG v2 목표 상대 대역 파워"):
    st.dataframe(
        pd.DataFrame(
            {
                "Band": list(target_profile.keys()),
                "Target relative power": list(target_profile.values()),
            }
        ),
        use_container_width=True,
    )

# ============================================================
# CSV 다운로드
# ============================================================

raw_eeg_df = pd.DataFrame({
    "time_sec": pseudo_eeg_time,
    "pseudo_eeg": pseudo_eeg,
})

fft_df = pd.DataFrame({
    "frequency_hz": freqs,
    "power": power,
})

safe_circuit = selected_circuit.replace(" ", "_")
safe_state = brain_dataset_label.replace(" ", "_")

raw_eeg_csv = (
    raw_eeg_df
    .to_csv(index=False)
    .encode("utf-8-sig")
)

fft_csv = (
    fft_df
    .to_csv(index=False)
    .encode("utf-8-sig")
)

download_col1, download_col2 = st.columns(2)

with download_col1:
    st.download_button(
        label="📥 Pseudo EEG Raw CSV 저장",
        data=raw_eeg_csv,
        file_name=(
            f"{safe_state}_{safe_circuit}"
            "_pseudo_eeg_v2_raw.csv"
        ),
        mime="text/csv",
    )

with download_col2:
    st.download_button(
        label="📥 Welch PSD CSV 저장",
        data=fft_csv,
        file_name=(
            f"{safe_state}_{safe_circuit}"
            "_pseudo_eeg_v2_welch_psd.csv"
        ),
        mime="text/csv",
    )

# ============================================================
# Simple neuron-level explanation
# ============================================================

with st.expander("원리 설명용: 단일 뉴런과 뉴런 집단 모델 보기"):
    col_hh, col_lif = st.columns(2)

    with col_hh:
        st.markdown("### Hodgkin–Huxley 단일 뉴런")
        t_hh, V_hh, I_hh = run_hh(10.0)

        fig_hh, ax = plt.subplots(figsize=(7, 3))
        ax.plot(t_hh, V_hh)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Membrane potential (mV)")
        ax.set_title("HH Single Neuron Spike")
        ax.grid(True)
        st.pyplot(fig_hh)

    with col_lif:
        st.markdown("### LIF 뉴런 집단")
        t_lif, V_lif, spikes = run_lif(noise=0.8, N=30)

        fig_lif, ax = plt.subplots(figsize=(7, 3))
        for neuron in range(spikes.shape[0]):
            spike_times = t_lif[spikes[neuron] == 1]
            ax.scatter(spike_times, np.ones_like(spike_times) * neuron, s=5)

        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Neuron index")
        ax.set_title("LIF Population Raster Plot")
        st.pyplot(fig_lif)
