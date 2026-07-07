import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from nilearn import datasets
from nilearn.surface import load_surf_mesh

from brain_models.connectivity import (
    load_connectivity_matrix,
    make_network_from_matrix,
    compute_centrality,
    scale_matrix_for_wc,
    load_group_difference_matrix,
    make_difference_network,
)
from brain_models.regions import REGIONS, CIRCUITS
from brain_models.hh import run_hh
from brain_models.lif import run_lif
from brain_models.wc import run_wc_network
from brain_models.eeg import make_pseudo_eeg


# =========================
# Page setting
# =========================

st.set_page_config(page_title="Mini Brain Simulator", layout="wide")

st.title("Mini Brain Simulator")
st.write("FreeSurfer fsaverage 표준 뇌 기반 감정·기억 회로 신경동역학 시뮬레이터")


# =========================
# Brain mesh
# =========================

@st.cache_data
def load_fsaverage_mesh():
    fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
    left_coords, left_faces = load_surf_mesh(fsaverage.pial_left)
    right_coords, right_faces = load_surf_mesh(fsaverage.pial_right)
    return left_coords, left_faces, right_coords, right_faces


def add_fsaverage_brain(fig):
    left_coords, left_faces, right_coords, right_faces = load_fsaverage_mesh()

    for coords, faces, name in [
        (left_coords, left_faces, "Left hemisphere"),
        (right_coords, right_faces, "Right hemisphere"),
    ]:
        fig.add_trace(
            go.Mesh3d(
                x=coords[:, 0],
                y=coords[:, 1],
                z=coords[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                opacity=0.5,
                color="lightgray",
                name=name,
                hoverinfo="skip",
                showlegend=False,
            )
        )

    return fig


# =========================
# Helper functions
# =========================

def get_top_connection_changes(diff_sub, top_n=5):
    changes = []
    regions = list(diff_sub.index)

    for i in range(len(regions)):
        for j in range(i + 1, len(regions)):
            r1 = regions[i]
            r2 = regions[j]
            diff = float(diff_sub.loc[r1, r2])

            changes.append(
                {
                    "Connection": f"{r1} - {r2}",
                    "Difference": diff,
                    "Direction": "Depression ↑" if diff > 0 else "Depression ↓",
                    "AbsDifference": abs(diff),
                }
            )

    changes = sorted(changes, key=lambda x: x["AbsDifference"], reverse=True)
    return changes[:top_n]


def safe_selected_region(selected_region, active_regions):
    if selected_region in active_regions:
        return selected_region
    return active_regions[0]


# =========================
# Sidebar
# =========================

st.sidebar.header("시뮬레이션 설정")

brain_dataset_label = st.sidebar.selectbox(
    "Brain Dataset",
    ["Subject 01", "Healthy Mean", "Depression Mean"],
)

dataset_map = {
    "Subject 01": "sub-01",
    "Healthy Mean": "control_mean",
    "Depression Mean": "depression_mean",
}
brain_dataset = dataset_map[brain_dataset_label]

view_mode = st.sidebar.selectbox(
    "View Mode",
    ["Single Dataset", "Depression - Healthy Difference"],
)

selected_circuit = st.sidebar.selectbox(
    "회로 선택",
    list(CIRCUITS.keys()),
)

active_regions = CIRCUITS[selected_circuit]["regions"]
active_edges = CIRCUITS[selected_circuit]["edges"]

selected_region = st.sidebar.selectbox(
    "분석할 뇌 영역 선택",
    active_regions,
)

brain_state = st.sidebar.selectbox(
    "Brain State",
    ["Normal", "Depression-like", "Hyperexcited"],
)

matrix_type = st.sidebar.selectbox(
    "Connectivity Matrix",
    ["correlation", "partial"],
)

input_current = st.sidebar.slider("HH Input Current", 0.0, 20.0, 10.0, 0.5)
lif_noise = st.sidebar.slider("LIF Noise Level", 0.1, 2.0, 0.8, 0.1)
eeg_noise = st.sidebar.slider("Pseudo EEG Noise", 0.0, 0.2, 0.05, 0.01)

activity_time_index_raw = st.sidebar.slider(
    "Brain Activity Time",
    min_value=0,
    max_value=999,
    value=999,
    step=10,
)


# =========================
# Run Wilson-Cowan
# =========================

try:
    if view_mode == "Single Dataset":
        conn_df = load_connectivity_matrix(brain_dataset, matrix_type)

        available_regions = [r for r in active_regions if r in conn_df.index]

        if len(available_regions) < 2:
            raise ValueError("현재 회로와 connectivity matrix의 ROI 이름이 충분히 일치하지 않습니다.")

        conn_sub = conn_df.loc[available_regions, available_regions]
        W_external, wc_regions = scale_matrix_for_wc(conn_sub, scale=0.35)

        active_regions = wc_regions
        active_edges = [
            edge for edge in active_edges
            if edge[0] in active_regions and edge[1] in active_regions
        ]
        selected_region = safe_selected_region(selected_region, active_regions)

        t_wc, E_wc, I_wc = run_wc_network(
            active_regions,
            brain_state,
            W_external=W_external,
        )

        G = make_network_from_matrix(conn_sub, threshold=0.45)
        centrality = compute_centrality(G)
        diff_sub = None

        st.success(
            f"{brain_dataset_label}의 실제 fMRI 기반 {matrix_type} connectivity matrix를 "
            "Wilson–Cowan 모델에 적용했습니다."
        )

    else:
        diff_df, control_df, depression_df = load_group_difference_matrix(matrix_type)

        available_regions = [r for r in active_regions if r in diff_df.index]

        if len(available_regions) < 2:
            raise ValueError("현재 회로와 difference matrix의 ROI 이름이 충분히 일치하지 않습니다.")

        diff_sub = diff_df.loc[available_regions, available_regions]

        # Difference view에서는 Wilson-Cowan을 Depression Mean matrix로 구동함.
        depression_sub = depression_df.loc[available_regions, available_regions]
        W_external, wc_regions = scale_matrix_for_wc(depression_sub, scale=0.35)

        active_regions = wc_regions
        active_edges = [
            edge for edge in active_edges
            if edge[0] in active_regions and edge[1] in active_regions
        ]
        selected_region = safe_selected_region(selected_region, active_regions)

        t_wc, E_wc, I_wc = run_wc_network(
            active_regions,
            brain_state,
            W_external=W_external,
        )

        conn_sub = depression_sub
        G = make_difference_network(diff_sub, threshold=0.10)
        centrality = compute_centrality(G)

        st.success(
            "Depression Mean - Healthy Mean difference network를 표시합니다. "
            "Wilson–Cowan은 Depression Mean matrix로 구동됩니다."
        )

except Exception as e:
    t_wc, E_wc, I_wc = run_wc_network(active_regions, brain_state)
    conn_sub = None
    diff_sub = None
    G = None
    centrality = {}
    selected_region = safe_selected_region(selected_region, active_regions)
    st.warning(f"Connectivity matrix를 불러오지 못해 기본 연결 행렬을 사용합니다: {e}")


selected_index = active_regions.index(selected_region)
selected_E = E_wc[selected_index]
selected_I = I_wc[selected_index]

activity_time_index = min(activity_time_index_raw, E_wc.shape[1] - 1)
activity_time = t_wc[activity_time_index]
region_activity = E_wc[:, activity_time_index]


# =========================
# 3D Brain Viewer + Activity Map
# =========================

st.subheader("1. 3D Brain Activity Map")

fig = go.Figure()
fig = add_fsaverage_brain(fig)

# edges
for start, end in active_edges:
    x0, y0, z0 = REGIONS[start]["pos"]
    x1, y1, z1 = REGIONS[end]["pos"]

    selected_edge = start == selected_region or end == selected_region

    if view_mode == "Depression - Healthy Difference" and diff_sub is not None:
        diff_value = 0.0

        if start in diff_sub.index and end in diff_sub.columns:
            diff_value = float(diff_sub.loc[start, end])

        line_width = 3 + 80 * abs(diff_value)

        if diff_value > 0:
            line_color = "crimson"
        elif diff_value < 0:
            line_color = "royalblue"
        else:
            line_color = "lightgray"

        hover = (
            f"{start} - {end}<br>"
            f"Depression - Healthy: {diff_value:.3f}<br>"
            "Red = stronger in depression<br>"
            "Blue = weaker in depression"
        )

    else:
        weight = 0.3
        if conn_sub is not None and start in conn_sub.index and end in conn_sub.columns:
            weight = float(conn_sub.loc[start, end])

        line_width = 3 + 8 * weight
        line_color = "crimson" if selected_edge else "gray"
        hover = f"{start} - {end}<br>Connectivity: {weight:.3f}"

    fig.add_trace(
        go.Scatter3d(
            x=[x0, x1],
            y=[y0, y1],
            z=[z0, z1],
            mode="lines",
            line=dict(width=line_width, color=line_color),
            showlegend=False,
            hovertext=hover,
            hoverinfo="text",
        )
    )

# nodes: Wilson-Cowan E(t)를 색과 크기로 표시
for idx, name in enumerate(active_regions):
    x, y, z = REGIONS[name]["pos"]
    activity = float(region_activity[idx])
    is_selected = name == selected_region

    fig.add_trace(
        go.Scatter3d(
            x=[x],
            y=[y],
            z=[z],
            mode="markers+text",
            marker=dict(
                size=(14 + 16 * activity) if is_selected else (9 + 12 * activity),
                color=[activity],
                colorscale="Turbo",
                cmin=0,
                cmax=1,
                symbol="diamond" if is_selected else "circle",
                opacity=0.95,
                line=dict(width=2, color="white"),
                colorbar=dict(title="E activity") if idx == 0 else None,
            ),
            text=[name],
            textposition="top center",
            name=name,
            hovertext=(
                f"<b>{name}</b><br>"
                f"{REGIONS[name]['desc']}<br>"
                f"E activity at t={activity_time:.1f}: {activity:.3f}"
            ),
            hoverinfo="text",
        )
    )

fig.update_layout(
    height=650,
    margin=dict(l=0, r=0, b=0, t=30),
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        bgcolor="white",
        aspectmode="data",
        camera=dict(eye=dict(x=1.7, y=-2.2, z=1.1)),
    ),
    showlegend=False,
)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    f"현재 Brain Activity Time: {activity_time:.1f}. "
    "노드 색과 크기는 해당 시점의 Wilson–Cowan 흥분성 activity E(t)를 의미한다. "
    "선택한 영역은 다이아몬드로 표시된다."
)


# =========================
# Difference Analysis
# =========================

if view_mode == "Depression - Healthy Difference" and diff_sub is not None:
    st.subheader("Depression - Healthy Difference 분석")

    diff_display_scale = st.slider(
        "Difference 시각화 확대 배율",
        min_value=1.0,
        max_value=10.0,
        value=5.0,
        step=0.5,
    )

    scaled_diff = diff_sub.values * diff_display_scale
    max_abs = max(float(np.max(np.abs(scaled_diff))), 1e-6)

    fig_diff, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(
        scaled_diff,
        cmap="bwr",
        vmin=-max_abs,
        vmax=max_abs,
    )

    ax.set_xticks(range(len(diff_sub.columns)))
    ax.set_xticklabels(diff_sub.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(diff_sub.index)))
    ax.set_yticklabels(diff_sub.index)
    ax.set_title("Depression - Healthy Difference Matrix")

    plt.colorbar(im, ax=ax, label="Scaled difference")
    plt.tight_layout()
    st.pyplot(fig_diff)

    st.caption(
        "빨간색은 우울증군에서 연결이 더 강한 경우, 파란색은 우울증군에서 연결이 더 약한 경우를 의미한다. "
        "시각화를 위해 차이값에 확대 배율을 적용했다."
    )

    top_changes = get_top_connection_changes(diff_sub, top_n=5)

    st.markdown("### 연결 변화 Top 5")
    top_df = pd.DataFrame(top_changes)
    top_df = top_df[["Connection", "Difference", "Direction"]]
    st.dataframe(top_df, use_container_width=True)


# =========================
# Selected Region Info
# =========================

st.subheader("2. 선택한 뇌 영역 정보")

info_col1, info_col2, info_col3 = st.columns([2, 1, 1])

with info_col1:
    st.markdown(f"## {selected_region}")
    st.write(REGIONS[selected_region]["desc"])

    connected_regions = []
    for start, end in active_edges:
        if start == selected_region:
            connected_regions.append(end)
        elif end == selected_region:
            connected_regions.append(start)

    st.write("**연결된 영역:**", ", ".join(connected_regions) if connected_regions else "없음")

    if view_mode == "Depression - Healthy Difference" and diff_sub is not None:
        st.write("**우울증군-정상군 연결 변화:**")

        changes = []

        for other in active_regions:
            if other == selected_region:
                continue

            if selected_region in diff_sub.index and other in diff_sub.columns:
                diff_value = float(diff_sub.loc[selected_region, other])
                changes.append((other, diff_value))

        changes = sorted(changes, key=lambda x: abs(x[1]), reverse=True)

        for other, diff_value in changes:
            if diff_value > 0:
                st.write(f"▲ {other}: +{diff_value:.3f} 증가")
            elif diff_value < 0:
                st.write(f"▼ {other}: {diff_value:.3f} 감소")
            else:
                st.write(f"- {other}: 변화 없음")

with info_col2:
    st.metric(
        f"E activity at t={activity_time:.1f}",
        f"{selected_E[activity_time_index]:.3f}",
    )

with info_col3:
    st.metric(
        f"I activity at t={activity_time:.1f}",
        f"{selected_I[activity_time_index]:.3f}",
    )

    if centrality and selected_region in centrality:
        st.metric(
            "Degree centrality",
            f"{centrality[selected_region]['degree']:.3f}",
        )
        st.metric(
            "Eigenvector centrality",
            f"{centrality[selected_region]['eigenvector']:.3f}",
        )


# =========================
# Wilson-Cowan graphs
# =========================

st.subheader("3. 선택 영역 중심 신경동역학 분석")

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"### {selected_region} Wilson–Cowan E/I Activity")

    fig_region, ax = plt.subplots(figsize=(7, 3))
    ax.plot(t_wc, selected_E, label=f"{selected_region} E")
    ax.plot(t_wc, selected_I, label=f"{selected_region} I")
    ax.axvline(activity_time, linestyle="--", alpha=0.7)
    ax.set_xlabel("Time")
    ax.set_ylabel("Activity")
    ax.set_title(f"{selected_region} E/I Dynamics")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig_region)

with col2:
    st.markdown(f"### {selected_circuit} 전체 영역 Activity")

    fig_network, ax = plt.subplots(figsize=(7, 3))

    for idx, name in enumerate(active_regions):
        if name == selected_region:
            ax.plot(t_wc, E_wc[idx], linewidth=3, label=f"{name} selected")
        else:
            ax.plot(t_wc, E_wc[idx], alpha=0.6, label=name)

    ax.axvline(activity_time, linestyle="--", alpha=0.7)
    ax.set_xlabel("Time")
    ax.set_ylabel("Excitatory activity")
    ax.set_title(f"{selected_circuit} - {brain_state}")
    ax.legend(fontsize=8)
    ax.grid(True)
    st.pyplot(fig_network)


# =========================
# HH + LIF
# =========================

st.subheader("4. 선택 영역의 뉴런 수준 시뮬레이션")

col3, col4 = st.columns(2)

with col3:
    st.markdown(f"### {selected_region} 대표 뉴런: Hodgkin–Huxley")

    t_hh, V_hh, I_hh = run_hh(input_current)

    fig_hh, ax = plt.subplots(figsize=(7, 3))
    ax.plot(t_hh, V_hh)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Membrane potential (mV)")
    ax.set_title(f"{selected_region} HH Single Neuron Spike")
    ax.grid(True)
    st.pyplot(fig_hh)

with col4:
    st.markdown(f"### {selected_region} 뉴런 집단: LIF Raster Plot")

    t_lif, V_lif, spikes = run_lif(noise=lif_noise, N=30)

    fig_lif, ax = plt.subplots(figsize=(7, 3))
    for neuron in range(spikes.shape[0]):
        spike_times = t_lif[spikes[neuron] == 1]
        ax.scatter(spike_times, np.ones_like(spike_times) * neuron, s=5)

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Neuron index")
    ax.set_title(f"{selected_region} LIF Population Raster")
    st.pyplot(fig_lif)


# =========================
# Pseudo EEG
# =========================

st.subheader("5. 회로 전체 Activity 기반 Pseudo EEG")

pseudo_eeg, freqs, power = make_pseudo_eeg(E_wc, I_wc, noise=eeg_noise)

col5, col6 = st.columns(2)

with col5:
    st.markdown("### Pseudo EEG Raw Signal")

    fig_eeg, ax = plt.subplots(figsize=(7, 3))
    ax.plot(t_wc, pseudo_eeg)
    ax.axvline(activity_time, linestyle="--", alpha=0.7)
    ax.set_xlabel("Time")
    ax.set_ylabel("Signal")
    ax.set_title(f"Pseudo EEG from {selected_circuit}")
    ax.grid(True)
    st.pyplot(fig_eeg)

with col6:
    st.markdown("### FFT Power Spectrum")

    fig_fft, ax = plt.subplots(figsize=(7, 3))
    ax.plot(freqs, power)
    ax.set_xlim(0, 40)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Power")
    ax.set_title("Pseudo EEG Frequency Spectrum")
    ax.grid(True)
    st.pyplot(fig_fft)


# =========================
# Summary
# =========================

st.subheader("6. 구현 의미")

st.write(
    f"""
    현재 버전은 **FreeSurfer fsaverage 표준 뇌 표면** 위에 **{selected_circuit}**를 배치하고,
    왼쪽 사이드바에서 선택한 영역인 **{selected_region}**을 중심으로 신경동역학 분석을 수행한다.

    구현된 스케일:
    - 표준 뇌 3D 메쉬
    - 감정 회로 / 기억 회로
    - 실제 fMRI 기반 기능적 연결성 행렬
    - Correlation / Partial correlation 연결성 비교
    - 영역별 Wilson–Cowan E/I 동역학
    - 선택 영역의 Hodgkin–Huxley 단일 뉴런
    - 선택 영역의 LIF 뉴런 집단
    - 회로 전체 activity 기반 pseudo EEG와 FFT 분석
    - Brain Activity Time 슬라이더를 이용한 시간별 활동 지도

    현재 선택된 데이터셋은 **{brain_dataset_label}**이며,
    해당 데이터셋의 AAL3 기반 functional connectivity matrix가 Wilson–Cowan 모델의 연결 행렬로 사용되었다.
    """
)

if view_mode == "Depression - Healthy Difference":
    st.write(
        """
        Difference View에서는 우울증군 평균 연결성에서 정상군 평균 연결성을 뺀 값을 표시하였다.
        빨간색 연결은 우울증군에서 더 강한 연결을, 파란색 연결은 우울증군에서 더 약한 연결을 의미한다.
        """
    )
