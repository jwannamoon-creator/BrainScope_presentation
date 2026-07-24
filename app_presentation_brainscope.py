import time

import pandas as pd
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

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
from eeg import make_pseudo_eeg
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

# 실행할 때마다 동일한 모의 EEG가 생성되도록 난수 시드를 고정함.
# Healthy와 Depression 결과를 같은 조건에서 비교하기 위한 설정임.
np.random.seed(42)

pseudo_eeg, freqs, power = make_pseudo_eeg(E_wc, I_wc, noise=0.04)

eeg_col1, eeg_col2 = st.columns(2)

with eeg_col1:
    fig_eeg, ax = plt.subplots(figsize=(7, 3))
    ax.plot(t_wc, pseudo_eeg)
    ax.axvline(activity_time, linestyle="--", alpha=0.7)
    ax.set_xlabel("Time")
    ax.set_ylabel("Signal")
    ax.set_title("Pseudo EEG Raw Signal")
    ax.grid(True)
    st.pyplot(fig_eeg)

with eeg_col2:
    fig_fft, ax = plt.subplots(figsize=(7, 3))
    ax.plot(freqs, power)
    ax.set_xlim(0, 40)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Power")
    ax.set_title("FFT Power Spectrum")
    ax.grid(True)
    st.pyplot(fig_fft)

# ============================================================
# CSV 다운로드
# ============================================================

# Raw EEG 데이터
raw_eeg_df = pd.DataFrame({
    "time": t_wc,
    "pseudo_eeg": pseudo_eeg
})

# FFT 데이터
fft_df = pd.DataFrame({
    "frequency_hz": freqs,
    "power": power
})

# 파일명에 사용할 안전한 문자열
# 사이드바에서 실제로 사용 중인 변수명은 selected_circuit과
# brain_dataset_label이므로 이 두 변수를 사용함.
safe_circuit = selected_circuit.replace(" ", "_")
safe_state = brain_dataset_label.replace(" ", "_")

raw_eeg_csv = raw_eeg_df.to_csv(index=False).encode("utf-8-sig")
fft_csv = fft_df.to_csv(index=False).encode("utf-8-sig")

download_col1, download_col2 = st.columns(2)

with download_col1:
    st.download_button(
        label="📥 Pseudo EEG Raw CSV 저장",
        data=raw_eeg_csv,
        file_name=f"{safe_state}_{safe_circuit}_pseudo_eeg_raw.csv",
        mime="text/csv"
    )

with download_col2:
    st.download_button(
        label="📥 FFT Spectrum CSV 저장",
        data=fft_csv,
        file_name=f"{safe_state}_{safe_circuit}_pseudo_eeg_fft.csv",
        mime="text/csv"
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
