import time

import pandas as pd
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from pathlib import Path

from scipy.stats import pearsonr
from scipy.spatial.distance import cosine

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

def prepare_laplacian_adjacency(conn_sub):
    """
    상관 기반 연결행렬을 Graph Laplacian 계산용
    비음수 대칭 인접행렬로 변환한다.

    - NaN과 무한값 제거
    - 대칭화
    - 상관계수 절댓값 사용
    - 자기 연결 제거
    """

    adjacency = np.asarray(
        conn_sub.values,
        dtype=float,
    )

    adjacency = np.nan_to_num(
        adjacency,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    # 수치 오차로 인한 비대칭 제거
    adjacency = (
        adjacency + adjacency.T
    ) / 2.0

    # 일반적인 Laplacian의 λ₂ 해석을 위해 비음수 가중치 사용
    adjacency = np.abs(adjacency)

    # 자기 자신과의 상관계수 1 제거
    np.fill_diagonal(adjacency, 0.0)

    return adjacency


def calculate_graph_laplacian(conn_sub):
    """
    비정규화 Graph Laplacian과 정규화 Graph Laplacian을 계산한다.

    L = D - A
    L_norm = I - D^(-1/2) A D^(-1/2)
    """

    adjacency = prepare_laplacian_adjacency(
        conn_sub
    )

    # --------------------------------------------------------
    # Degree matrix
    # --------------------------------------------------------

    weighted_degree = np.sum(
        adjacency,
        axis=1,
    )

    degree_matrix = np.diag(
        weighted_degree
    )

    # --------------------------------------------------------
    # Unnormalized Laplacian
    # --------------------------------------------------------

    laplacian = (
        degree_matrix - adjacency
    )

    laplacian_eigenvalues = np.linalg.eigvalsh(
        laplacian
    )

    laplacian_eigenvalues[
        np.abs(laplacian_eigenvalues) < 1e-10
    ] = 0.0

    laplacian_eigenvalues = np.sort(
        laplacian_eigenvalues
    )

    # --------------------------------------------------------
    # Normalized Laplacian
    # --------------------------------------------------------

    inverse_sqrt_degree = np.zeros(
        len(weighted_degree),
        dtype=float,
    )

    nonzero_degree_mask = (
        weighted_degree > 1e-12
    )

    inverse_sqrt_degree[
        nonzero_degree_mask
    ] = (
        1.0
        / np.sqrt(
            weighted_degree[
                nonzero_degree_mask
            ]
        )
    )

    degree_inverse_sqrt_matrix = np.diag(
        inverse_sqrt_degree
    )

    normalized_laplacian = (
        np.eye(
            len(adjacency),
            dtype=float,
        )
        - degree_inverse_sqrt_matrix
        @ adjacency
        @ degree_inverse_sqrt_matrix
    )

    # 고립 노드가 있을 때 대각값을 0으로 설정
    isolated_node_mask = (
        weighted_degree <= 1e-12
    )

    normalized_laplacian[
        isolated_node_mask,
        isolated_node_mask,
    ] = 0.0

    normalized_eigenvalues = np.linalg.eigvalsh(
        normalized_laplacian
    )

    normalized_eigenvalues[
        np.abs(normalized_eigenvalues) < 1e-10
    ] = 0.0

    normalized_eigenvalues = np.sort(
        normalized_eigenvalues
    )

    # --------------------------------------------------------
    # Adjacency eigenvalues
    # --------------------------------------------------------

    adjacency_eigenvalues = np.linalg.eigvalsh(
        adjacency
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    if len(laplacian_eigenvalues) >= 2:
        algebraic_connectivity = float(
            laplacian_eigenvalues[1]
        )
    else:
        algebraic_connectivity = 0.0

    if len(normalized_eigenvalues) >= 2:
        normalized_lambda2 = float(
            normalized_eigenvalues[1]
        )
    else:
        normalized_lambda2 = 0.0

    metrics = {
        "algebraic_connectivity": (
            algebraic_connectivity
        ),
        "normalized_lambda2": (
            normalized_lambda2
        ),
        "largest_laplacian_eigenvalue": float(
            laplacian_eigenvalues[-1]
        ),
        "normalized_lambda_max": float(
            normalized_eigenvalues[-1]
        ),
        "average_weighted_degree": float(
            np.mean(weighted_degree)
        ),
        "spectral_radius": float(
            np.max(
                np.abs(
                    adjacency_eigenvalues
                )
            )
        ),
        "total_edge_weight": float(
            np.sum(adjacency) / 2.0
        ),
    }

    return {
        "adjacency": adjacency,
        "degree_matrix": degree_matrix,
        "laplacian": laplacian,
        "normalized_laplacian": (
            normalized_laplacian
        ),
        "eigenvalues": (
            laplacian_eigenvalues
        ),
        "normalized_eigenvalues": (
            normalized_eigenvalues
        ),
        "weighted_degree": weighted_degree,
        "metrics": metrics,
    }


normalized_eigenvalues = np.linalg.eigvalsh(
    normalized_laplacian
)

normalized_eigenvalues[
    np.abs(normalized_eigenvalues) < 1e-10
] = 0

def build_laplacian_comparison(active_regions):
    """
    현재 회로에 대해 Healthy와 Depression 연결행렬을
    같은 영역 순서로 정렬하여 Laplacian을 비교한다.
    """

    healthy_df = load_connectivity_matrix(
        "control_mean",
        "correlation",
    )

    depression_df = load_connectivity_matrix(
        "depression_mean",
        "correlation",
    )

    common_regions = [
        region
        for region in active_regions
        if (
            region in healthy_df.index
            and region in healthy_df.columns
            and region in depression_df.index
            and region in depression_df.columns
        )
    ]

    if len(common_regions) < 2:
        raise ValueError(
            "Graph Laplacian 계산에 필요한 공통 영역이 "
            "2개 미만입니다."
        )

    healthy_sub = healthy_df.loc[
        common_regions,
        common_regions,
    ]

    depression_sub = depression_df.loc[
        common_regions,
        common_regions,
    ]

    healthy_result = calculate_graph_laplacian(
        healthy_sub
    )

    depression_result = calculate_graph_laplacian(
        depression_sub
    )

    return (
        common_regions,
        healthy_result,
        depression_result,
    )

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
# Real EEG vs Pseudo EEG validation
# ============================================================

REAL_EEG_PSD_FILES = {
    "Healthy": Path(
        "real_eeg_reference/real_healthy_psd.csv"
    ),
    "Depression": Path(
        "real_eeg_reference/real_depression_psd.csv"
    ),
}

VALIDATION_BANDS = {
    "Delta": (1.0, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 13.0),
    "Beta": (13.0, 30.0),
    "Gamma": (30.0, 40.0),
}


def _find_column(
    dataframe,
    candidate_names,
):
    """
    대소문자와 일부 열 이름 차이를 허용해
    적절한 CSV 열을 찾는다.
    """

    normalized_columns = {
        str(column).strip().lower(): column
        for column in dataframe.columns
    }

    for candidate in candidate_names:
        candidate_normalized = (
            candidate.strip().lower()
        )

        if candidate_normalized in normalized_columns:
            return normalized_columns[
                candidate_normalized
            ]

    # 정확히 일치하지 않을 경우 부분 문자열 탐색
    for column in dataframe.columns:
        normalized_column = (
            str(column).strip().lower()
        )

        for candidate in candidate_names:
            if candidate.lower() in normalized_column:
                return column

    raise ValueError(
        "필요한 열을 찾지 못했습니다. "
        f"후보 열 이름: {candidate_names}"
    )


@st.cache_data
def load_real_eeg_psd(
    brain_state,
):
    """
    실제 EEG PSD CSV를 불러오고 1–40 Hz 구간만 반환한다.
    """

    file_path = REAL_EEG_PSD_FILES[
        brain_state
    ]

    if not file_path.exists():
        raise FileNotFoundError(
            f"실제 EEG PSD 파일을 찾지 못했습니다: "
            f"{file_path}"
        )

    dataframe = pd.read_csv(
        file_path
    )

    frequency_column = _find_column(
        dataframe,
        [
            "frequency_hz",
            "frequency",
            "freq",
            "freqs",
            "hz",
        ],
    )

    power_column = _find_column(
    dataframe,
    [
        "mean_psd_uV2_per_Hz",
        "mean_psd_V2_per_Hz",
        "normalized_power",
        "normalized_psd",
        "power",
        "psd",
        "density",
    ],
)

    frequencies = pd.to_numeric(
        dataframe[frequency_column],
        errors="coerce",
    ).to_numpy(dtype=float)

    power = pd.to_numeric(
        dataframe[power_column],
        errors="coerce",
    ).to_numpy(dtype=float)

    valid_mask = (
        np.isfinite(frequencies)
        & np.isfinite(power)
        & (frequencies >= 1.0)
        & (frequencies <= 40.0)
        & (power >= 0.0)
    )

    frequencies = frequencies[
        valid_mask
    ]
    power = power[
        valid_mask
    ]

    sort_indices = np.argsort(
        frequencies
    )

    frequencies = frequencies[
        sort_indices
    ]
    power = power[
        sort_indices
    ]

    if len(frequencies) < 3:
        raise ValueError(
            f"{brain_state} 실제 EEG PSD에 "
            "유효한 주파수점이 충분하지 않습니다."
        )

    return frequencies, power


def normalize_psd_area(
    frequencies,
    power,
    low_frequency=1.0,
    high_frequency=40.0,
):
    """
    지정 주파수 범위의 PSD 면적이 1이 되도록 정규화한다.
    """

    frequencies = np.asarray(
        frequencies,
        dtype=float,
    )

    power = np.asarray(
        power,
        dtype=float,
    )

    mask = (
        (frequencies >= low_frequency)
        & (frequencies <= high_frequency)
        & np.isfinite(power)
        & (power >= 0.0)
    )

    selected_frequencies = frequencies[
        mask
    ]

    selected_power = power[
        mask
    ]

    if len(selected_frequencies) < 3:
        raise ValueError(
            "PSD 정규화에 필요한 주파수점이 부족합니다."
        )

    total_area = np.trapz(
        selected_power,
        selected_frequencies,
    )

    if total_area <= 1e-15:
        raise ValueError(
            "PSD 전체 면적이 0에 가깝습니다."
        )

    normalized_power = (
        selected_power / total_area
    )

    return (
        selected_frequencies,
        normalized_power,
    )


def calculate_relative_band_power(
    frequencies,
    normalized_power,
):
    """
    정규화 PSD에서 각 EEG 대역의 상대 파워를 계산한다.
    """

    frequencies = np.asarray(
        frequencies,
        dtype=float,
    )

    normalized_power = np.asarray(
        normalized_power,
        dtype=float,
    )

    band_power = {}

    for band_name, (
        low_frequency,
        high_frequency,
    ) in VALIDATION_BANDS.items():

        # 경계 중복을 최소화한다.
        if band_name == "Gamma":
            mask = (
                (frequencies >= low_frequency)
                & (frequencies <= high_frequency)
            )
        else:
            mask = (
                (frequencies >= low_frequency)
                & (frequencies < high_frequency)
            )

        if np.sum(mask) < 2:
            band_power[band_name] = np.nan
            continue

        band_power[band_name] = float(
            np.trapz(
                normalized_power[mask],
                frequencies[mask],
            )
        )

    valid_total = np.nansum(
        list(band_power.values())
    )

    if valid_total > 1e-15:
        band_power = {
            band_name: (
                value / valid_total
                if np.isfinite(value)
                else np.nan
            )
            for band_name, value in band_power.items()
        }

    return band_power


def compare_real_and_pseudo_psd(
    real_frequencies,
    real_power,
    pseudo_frequencies,
    pseudo_power,
):
    """
    실제 PSD와 pseudo PSD를 같은 주파수축으로 맞추고
    Pearson, cosine, RMSE, NRMSE를 계산한다.
    """

    (
        real_frequencies,
        real_normalized_power,
    ) = normalize_psd_area(
        real_frequencies,
        real_power,
    )

    (
        pseudo_frequencies,
        pseudo_normalized_power,
    ) = normalize_psd_area(
        pseudo_frequencies,
        pseudo_power,
    )

    # 실제 EEG 주파수축에 pseudo PSD를 보간
    pseudo_interpolated = np.interp(
        real_frequencies,
        pseudo_frequencies,
        pseudo_normalized_power,
    )

    # 보간 뒤 다시 면적 정규화
    pseudo_area = np.trapz(
        pseudo_interpolated,
        real_frequencies,
    )

    if pseudo_area > 1e-15:
        pseudo_interpolated = (
            pseudo_interpolated
            / pseudo_area
        )

    pearson_r, pearson_p = pearsonr(
        real_normalized_power,
        pseudo_interpolated,
    )

    cosine_similarity = (
        1.0
        - cosine(
            real_normalized_power,
            pseudo_interpolated,
        )
    )

    rmse = float(
        np.sqrt(
            np.mean(
                (
                    real_normalized_power
                    - pseudo_interpolated
                )
                ** 2
            )
        )
    )

    real_range = float(
        np.max(real_normalized_power)
        - np.min(real_normalized_power)
    )

    nrmse = (
        rmse / real_range
        if real_range > 1e-15
        else np.nan
    )

    real_band_power = (
        calculate_relative_band_power(
            real_frequencies,
            real_normalized_power,
        )
    )

    pseudo_band_power = (
        calculate_relative_band_power(
            real_frequencies,
            pseudo_interpolated,
        )
    )

    return {
        "frequencies": real_frequencies,
        "real_normalized_power": (
            real_normalized_power
        ),
        "pseudo_normalized_power": (
            pseudo_interpolated
        ),
        "pearson_r": float(
            pearson_r
        ),
        "pearson_p": float(
            pearson_p
        ),
        "cosine_similarity": float(
            cosine_similarity
        ),
        "rmse": rmse,
        "nrmse": float(
            nrmse
        ),
        "real_band_power": (
            real_band_power
        ),
        "pseudo_band_power": (
            pseudo_band_power
        ),
    }


def generate_condition_pseudo_eeg(
    circuit,
    brain_state,
    sfreq=100.0,
):
    """
    Healthy/Depression 방향 비교를 위해 특정 상태의
    Wilson–Cowan 활동과 pseudo EEG를 독립적으로 생성한다.
    """

    circuit_regions = (
        CIRCUITS[circuit]["regions"].copy()
    )

    dataset_id = get_dataset_id(
        brain_state
    )

    (
        condition_conn_sub,
        condition_W_external,
        condition_regions,
    ) = build_connection_submatrix(
        dataset_id,
        circuit_regions,
    )

    (
        condition_t,
        condition_E,
        condition_I,
    ) = run_wc_network(
        condition_regions,
        "Normal",
        W_external=condition_W_external,
    )

    (
        condition_pseudo_time,
        condition_pseudo_eeg,
        condition_frequencies,
        condition_power,
        condition_target_profile,
    ) = make_pseudo_eeg_100hz(
        t=condition_t,
        E=condition_E,
        I=condition_I,
        region_names=condition_regions,
        circuit=circuit,
        brain_state=brain_state,
        sfreq=sfreq,
        noise=0.025,
        random_state=42,
        wc_mix=0.18,
    )

    return {
        "time": condition_pseudo_time,
        "signal": condition_pseudo_eeg,
        "frequencies": condition_frequencies,
        "power": condition_power,
        "target_profile": (
            condition_target_profile
        ),
    }


def get_change_direction(
    healthy_value,
    depression_value,
    tolerance=1e-6,
):
    difference = (
        depression_value
        - healthy_value
    )

    if difference > tolerance:
        return "Increase"

    if difference < -tolerance:
        return "Decrease"

    return "No change"


def describe_similarity(
    pearson_r,
    cosine_similarity,
):
    """
    앱 표시용 보수적인 정성 평가.
    """

    if (
        pearson_r >= 0.70
        and cosine_similarity >= 0.80
    ):
        return "High"

    if (
        pearson_r >= 0.50
        and cosine_similarity >= 0.70
    ):
        return "Moderate to high"

    if (
        pearson_r >= 0.30
        and cosine_similarity >= 0.50
    ):
        return "Moderate"

    return "Low"

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

# ============================================================
# Graph Laplacian analysis
# ============================================================

try:
    (
        laplacian_regions,
        healthy_laplacian_result,
        depression_laplacian_result,
    ) = build_laplacian_comparison(
        active_regions
    )

    laplacian_available = True
    laplacian_error_message = None

except Exception as e:
    laplacian_available = False
    laplacian_error_message = str(e)

    laplacian_regions = []
    healthy_laplacian_result = None
    depression_laplacian_result = None

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
# Graph Laplacian Network Analysis
# ============================================================

st.subheader("4. Graph Laplacian Network Analysis")

st.caption(
    "연결행렬의 절댓값을 비음수 가중 인접행렬 A로 사용하고, "
    "차수행렬 D를 계산하여 비정규화 라플라시안 L = D - A를 "
    "구했습니다. λ₂는 회로의 전체적 연결성과 통합 정도를 "
    "나타내는 algebraic connectivity입니다."
)

if laplacian_available:

    healthy_metrics = (
        healthy_laplacian_result["metrics"]
    )

    depression_metrics = (
        depression_laplacian_result["metrics"]
    )

    lap_col1, lap_col2, lap_col3, lap_col4 = st.columns(4)

    with lap_col1:
        st.metric(
            "Healthy λ₂",
            (
                f"{healthy_metrics['algebraic_connectivity']:.4f}"
            ),
            help=(
                "두 번째로 작은 Laplacian 고유값. "
                "값이 클수록 네트워크가 전체적으로 강하게 "
                "연결되어 있음을 의미합니다."
            ),
        )

    with lap_col2:
        lambda2_difference = (
            depression_metrics[
                "algebraic_connectivity"
            ]
            - healthy_metrics[
                "algebraic_connectivity"
            ]
        )

        st.metric(
            "Depression λ₂",
            (
                f"{depression_metrics['algebraic_connectivity']:.4f}"
            ),
            delta=f"{lambda2_difference:+.4f}",
            delta_color="normal",
            help=(
                "Depression λ₂에서 Healthy λ₂를 뺀 값입니다."
            ),
        )

    with lap_col3:
        degree_difference = (
            depression_metrics[
                "average_weighted_degree"
            ]
            - healthy_metrics[
                "average_weighted_degree"
            ]
        )

        st.metric(
            "Average weighted degree",
            (
                f"{depression_metrics['average_weighted_degree']:.4f}"
            ),
            delta=f"{degree_difference:+.4f}",
            help=(
                "Depression 값과 Healthy 대비 변화량입니다."
            ),
        )

with lap_col4:
    normalized_lambda2_difference = (
        depression_metrics[
            "normalized_lambda2"
        ]
        - healthy_metrics[
            "normalized_lambda2"
        ]
    )

    st.metric(
        "Normalized λ₂",
        (
            f"{depression_metrics['normalized_lambda2']:.4f}"
        ),
        delta=(
            f"{normalized_lambda2_difference:+.4f}"
        ),
        help=(
            "노드별 연결 강도 차이를 보정한 "
            "정규화 Laplacian의 두 번째 고유값입니다."
        ),
    )

    spectrum_col, matrix_col = st.columns(
        [1.1, 1]
    )

    with spectrum_col:
        fig_laplacian, ax = plt.subplots(
            figsize=(7.5, 4.0)
        )

        healthy_eigenvalues = (
            healthy_laplacian_result["eigenvalues"]
        )

        depression_eigenvalues = (
            depression_laplacian_result["eigenvalues"]
        )

        healthy_indices = np.arange(
            1,
            len(healthy_eigenvalues) + 1,
        )

        depression_indices = np.arange(
            1,
            len(depression_eigenvalues) + 1,
        )

        ax.plot(
            healthy_indices,
            healthy_eigenvalues,
            marker="o",
            linewidth=2,
            label="Healthy",
        )

        ax.plot(
            depression_indices,
            depression_eigenvalues,
            marker="s",
            linewidth=2,
            label="Depression",
        )

        ax.axhline(
            0.0,
            linestyle="--",
            alpha=0.5,
        )

        ax.set_xlabel("Eigenvalue index")
        ax.set_ylabel("Laplacian eigenvalue")
        ax.set_title(
            f"{selected_circuit} Laplacian Spectrum"
        )
        ax.grid(True, alpha=0.3)
        ax.legend()

        st.pyplot(fig_laplacian)
        plt.close(fig_laplacian)

    with matrix_col:
        selected_laplacian_result = (
            healthy_laplacian_result
            if brain_dataset_label == "Healthy"
            else depression_laplacian_result
        )

        fig_laplacian_matrix, ax = plt.subplots(
            figsize=(6.5, 5.2)
        )

        laplacian_matrix = (
            selected_laplacian_result["laplacian"]
        )

        vmax = np.max(
            np.abs(laplacian_matrix)
        )

        if vmax == 0:
            vmax = 1.0

        image = ax.imshow(
            laplacian_matrix,
            cmap="bwr",
            vmin=-vmax,
            vmax=vmax,
        )

        ax.set_xticks(
            range(len(laplacian_regions))
        )
        ax.set_xticklabels(
            laplacian_regions,
            rotation=45,
            ha="right",
            fontsize=8,
        )

        ax.set_yticks(
            range(len(laplacian_regions))
        )
        ax.set_yticklabels(
            laplacian_regions,
            fontsize=8,
        )

        ax.set_title(
            f"{brain_dataset_label} Graph Laplacian"
        )

        plt.colorbar(
            image,
            ax=ax,
            label="Laplacian value",
        )
        plt.tight_layout()

        st.pyplot(fig_laplacian_matrix)
        plt.close(fig_laplacian_matrix)

    # 수치 비교 표
    laplacian_metrics_df = pd.DataFrame(
        {
            "Metric": [
                "Algebraic connectivity λ₂",
                "Normalized algebraic connectivity λ₂",
                "Largest Laplacian eigenvalue",
                "Largest normalized Laplacian eigenvalue",
                "Average weighted degree",
                "Adjacency spectral radius",
                "Total edge weight",
            ],
            "Healthy": [
                healthy_metrics[
                    "algebraic_connectivity"
                ],
                healthy_metrics[
                    "largest_laplacian_eigenvalue"
                ],
                healthy_metrics[
                    "average_weighted_degree"
                ],
                healthy_metrics[
                    "spectral_radius"
                ],
                healthy_metrics[
                    "total_edge_weight"
                ],
            ],
            "Depression": [
                depression_metrics[
                    "algebraic_connectivity"
                ],
                depression_metrics[
                    "largest_laplacian_eigenvalue"
                ],
                depression_metrics[
                    "average_weighted_degree"
                ],
                depression_metrics[
                    "spectral_radius"
                ],
                depression_metrics[
                    "total_edge_weight"
                ],
            ],
        }
    )

    laplacian_metrics_df["Depression - Healthy"] = (
        laplacian_metrics_df["Depression"]
        - laplacian_metrics_df["Healthy"]
    )

    st.dataframe(
        laplacian_metrics_df.style.format(
            {
                "Healthy": "{:.5f}",
                "Depression": "{:.5f}",
                "Depression - Healthy": "{:+.5f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    # λ₂ 해석 자동 출력
    healthy_lambda2 = healthy_metrics[
        "algebraic_connectivity"
    ]
    depression_lambda2 = depression_metrics[
        "algebraic_connectivity"
    ]

    if depression_lambda2 < healthy_lambda2:
        st.info(
            "이 회로에서는 Depression의 λ₂가 Healthy보다 낮게 "
            "나타났습니다. 이는 사용한 가중 네트워크에서 우울증군의 "
            "전반적 네트워크 통합성과 연결 강건성이 상대적으로 "
            "낮아진 것으로 해석할 수 있습니다."
        )

    elif depression_lambda2 > healthy_lambda2:
        st.info(
            "이 회로에서는 Depression의 λ₂가 Healthy보다 높게 "
            "나타났습니다. 이는 사용한 가중 네트워크에서 우울증군의 "
            "전반적 결합이 더 강하거나 일부 연결이 과도하게 강화된 "
            "상태일 가능성을 보여줍니다."
        )

    else:
        st.info(
            "Healthy와 Depression의 λ₂가 동일하게 나타났습니다."
        )

else:
    st.warning(
        "Graph Laplacian 분석을 수행하지 못했습니다: "
        f"{laplacian_error_message}"
    )

# ============================================================
# Pseudo EEG
# ============================================================

st.subheader("5. Pseudo EEG")

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

# STEP 9에서 Healthy→Depression 대역 변화 방향을 비교하기 위해
# 현재 회로의 두 상태 pseudo EEG를 모두 생성한다.
try:
    healthy_condition_pseudo = (
        generate_condition_pseudo_eeg(
            circuit=selected_circuit,
            brain_state="Healthy",
            sfreq=PSEUDO_EEG_SFREQ,
        )
    )

    depression_condition_pseudo = (
        generate_condition_pseudo_eeg(
            circuit=selected_circuit,
            brain_state="Depression",
            sfreq=PSEUDO_EEG_SFREQ,
        )
    )

    condition_pseudo_available = True
    condition_pseudo_error = None

except Exception as e:
    condition_pseudo_available = False
    condition_pseudo_error = str(e)

    healthy_condition_pseudo = None
    depression_condition_pseudo = None

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
# STEP 9. Real EEG vs Pseudo EEG validation
# ============================================================

st.subheader("6. Real EEG vs Pseudo EEG Validation")

st.caption(
    "실제 EEG와 BrainScope pseudo EEG의 1–40 Hz Welch PSD를 "
    "면적 정규화한 뒤 동일 주파수축에서 비교합니다. "
    "Pearson은 PSD 형태의 선형 유사도, cosine similarity는 "
    "스펙트럼 방향 유사도, RMSE는 주파수별 차이를 나타냅니다."
)

try:
    (
        real_frequencies,
        real_power,
    ) = load_real_eeg_psd(
        brain_dataset_label
    )

    validation_result = (
        compare_real_and_pseudo_psd(
            real_frequencies=real_frequencies,
            real_power=real_power,
            pseudo_frequencies=freqs,
            pseudo_power=power,
        )
    )

    validation_available = True
    validation_error_message = None

except Exception as e:
    validation_available = False
    validation_error_message = str(e)
    validation_result = None


if validation_available:

    similarity_level = describe_similarity(
        validation_result["pearson_r"],
        validation_result[
            "cosine_similarity"
        ],
    )

    metric_col1, metric_col2, metric_col3, metric_col4 = (
        st.columns(4)
    )

    with metric_col1:
        st.metric(
            "Pearson r",
            (
                f"{validation_result['pearson_r']:.3f}"
            ),
            help=(
                "실제 PSD와 pseudo PSD의 주파수별 "
                "선형 상관계수입니다."
            ),
        )

    with metric_col2:
        st.metric(
            "Cosine similarity",
            (
                f"{validation_result['cosine_similarity']:.3f}"
            ),
            help=(
                "두 스펙트럼 벡터의 방향 유사도입니다. "
                "1에 가까울수록 유사합니다."
            ),
        )

    with metric_col3:
        st.metric(
            "RMSE",
            (
                f"{validation_result['rmse']:.5f}"
            ),
            help=(
                "주파수별 정규화 PSD 차이의 "
                "제곱평균제곱근입니다."
            ),
        )

    with metric_col4:
        st.metric(
            "Similarity",
            similarity_level,
        )

    # --------------------------------------------------------
    # PSD 중첩 비교
    # --------------------------------------------------------

    fig_validation, ax = plt.subplots(
        figsize=(10, 4.2)
    )

    ax.plot(
        validation_result["frequencies"],
        validation_result[
            "real_normalized_power"
        ],
        linewidth=2.2,
        label=f"Real {brain_dataset_label}",
    )

    ax.plot(
        validation_result["frequencies"],
        validation_result[
            "pseudo_normalized_power"
        ],
        linewidth=1.8,
        alpha=0.85,
        label=(
            f"Pseudo {brain_dataset_label} "
            f"{selected_circuit.replace(' Circuit', '')}"
        ),
    )

    ax.set_xlim(1, 40)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized PSD")
    ax.set_title(
        f"Real vs Pseudo EEG Spectrum — "
        f"{brain_dataset_label}, {selected_circuit}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()

    st.pyplot(fig_validation)
    plt.close(fig_validation)

    # --------------------------------------------------------
    # 상대 대역 파워 비교
    # --------------------------------------------------------

    real_band_power = validation_result[
        "real_band_power"
    ]

    pseudo_band_power = validation_result[
        "pseudo_band_power"
    ]

    band_power_df = pd.DataFrame(
        {
            "Band": list(
                VALIDATION_BANDS.keys()
            ),
            "Real": [
                real_band_power[band]
                for band in VALIDATION_BANDS
            ],
            "Pseudo": [
                pseudo_band_power[band]
                for band in VALIDATION_BANDS
            ],
        }
    )

    band_power_df["Pseudo - Real"] = (
        band_power_df["Pseudo"]
        - band_power_df["Real"]
    )

    band_table_col, band_graph_col = (
        st.columns([1, 1.25])
    )

    with band_table_col:
        st.markdown(
            "### Relative band power"
        )

        st.dataframe(
            band_power_df.style.format(
                {
                    "Real": "{:.4f}",
                    "Pseudo": "{:.4f}",
                    "Pseudo - Real": "{:+.4f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    with band_graph_col:
        fig_band, ax = plt.subplots(
            figsize=(7, 4)
        )

        band_indices = np.arange(
            len(band_power_df)
        )

        bar_width = 0.36

        ax.bar(
            band_indices - bar_width / 2,
            band_power_df["Real"],
            width=bar_width,
            label="Real",
        )

        ax.bar(
            band_indices + bar_width / 2,
            band_power_df["Pseudo"],
            width=bar_width,
            label="Pseudo",
        )

        ax.set_xticks(
            band_indices
        )
        ax.set_xticklabels(
            band_power_df["Band"]
        )
        ax.set_ylabel(
            "Relative power"
        )
        ax.set_title(
            "EEG Band Power Comparison"
        )
        ax.grid(
            True,
            axis="y",
            alpha=0.3,
        )
        ax.legend()

        st.pyplot(fig_band)
        plt.close(fig_band)

    st.caption(
        f"Pearson p-value: "
        f"{validation_result['pearson_p']:.3e} · "
        f"NRMSE: {validation_result['nrmse']:.4f}"
    )

else:
    st.warning(
        "실제 EEG 검증을 수행하지 못했습니다: "
        f"{validation_error_message}"
    )

    st.code(
        "real_eeg_reference/\n"
        "├─ real_healthy_psd.csv\n"
        "└─ real_depression_psd.csv"
    )


# ------------------------------------------------------------
# Healthy → Depression direction match
# ------------------------------------------------------------

st.markdown(
    "### Healthy → Depression band-direction validation"
)

try:
    if not condition_pseudo_available:
        raise RuntimeError(
            condition_pseudo_error
        )

    (
        real_healthy_frequencies,
        real_healthy_power,
    ) = load_real_eeg_psd(
        "Healthy"
    )

    (
        real_depression_frequencies,
        real_depression_power,
    ) = load_real_eeg_psd(
        "Depression"
    )

    (
        real_healthy_frequencies,
        real_healthy_normalized,
    ) = normalize_psd_area(
        real_healthy_frequencies,
        real_healthy_power,
    )

    (
        real_depression_frequencies,
        real_depression_normalized,
    ) = normalize_psd_area(
        real_depression_frequencies,
        real_depression_power,
    )

    real_healthy_bands = (
        calculate_relative_band_power(
            real_healthy_frequencies,
            real_healthy_normalized,
        )
    )

    real_depression_bands = (
        calculate_relative_band_power(
            real_depression_frequencies,
            real_depression_normalized,
        )
    )

    (
        pseudo_healthy_frequencies,
        pseudo_healthy_normalized,
    ) = normalize_psd_area(
        healthy_condition_pseudo[
            "frequencies"
        ],
        healthy_condition_pseudo[
            "power"
        ],
    )

    (
        pseudo_depression_frequencies,
        pseudo_depression_normalized,
    ) = normalize_psd_area(
        depression_condition_pseudo[
            "frequencies"
        ],
        depression_condition_pseudo[
            "power"
        ],
    )

    pseudo_healthy_bands = (
        calculate_relative_band_power(
            pseudo_healthy_frequencies,
            pseudo_healthy_normalized,
        )
    )

    pseudo_depression_bands = (
        calculate_relative_band_power(
            pseudo_depression_frequencies,
            pseudo_depression_normalized,
        )
    )

    direction_rows = []

    for band_name in VALIDATION_BANDS:

        real_direction = get_change_direction(
            real_healthy_bands[band_name],
            real_depression_bands[band_name],
        )

        pseudo_direction = get_change_direction(
            pseudo_healthy_bands[band_name],
            pseudo_depression_bands[band_name],
        )

        direction_match = (
            real_direction
            == pseudo_direction
        )

        direction_rows.append(
            {
                "Band": band_name,
                "Real direction": real_direction,
                "Pseudo direction": pseudo_direction,
                "Match": (
                    "✓" if direction_match else "✗"
                ),
            }
        )

    direction_df = pd.DataFrame(
        direction_rows
    )

    number_of_matches = int(
        np.sum(
            direction_df["Match"] == "✓"
        )
    )

    direction_col1, direction_col2 = (
        st.columns([2, 1])
    )

    with direction_col1:
        st.dataframe(
            direction_df,
            use_container_width=True,
            hide_index=True,
        )

    with direction_col2:
        st.metric(
            "Direction match",
            (
                f"{number_of_matches}/"
                f"{len(VALIDATION_BANDS)}"
            ),
        )

        match_percentage = (
            100.0
            * number_of_matches
            / len(VALIDATION_BANDS)
        )

        st.metric(
            "Match rate",
            f"{match_percentage:.0f}%",
        )

    if number_of_matches == len(
        VALIDATION_BANDS
    ):
        st.success(
            "Pseudo EEG가 모든 주파수 대역에서 실제 EEG의 "
            "Healthy→Depression 변화 방향을 재현했습니다."
        )

    elif number_of_matches >= 4:
        st.info(
            "Pseudo EEG가 대부분의 대역에서 실제 EEG의 "
            "Healthy→Depression 변화 방향을 재현했습니다. "
            "불일치 대역은 실제 변화량, 회로 변조 및 확률적 "
            "신호 생성의 영향을 함께 고려해 해석해야 합니다."
        )

    else:
        st.warning(
            "Healthy→Depression 변화 방향의 일치도가 제한적입니다. "
            "상태별 대역 프로필과 Wilson–Cowan 혼합 비율을 "
            "추가로 점검할 필요가 있습니다."
        )

except Exception as e:
    st.warning(
        "대역 변화 방향을 계산하지 못했습니다: "
        f"{e}"
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
