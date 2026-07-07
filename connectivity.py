from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx


def load_connectivity_matrix(dataset="sub-01", matrix_type="correlation"):
    """
    Load AAL3 connectivity matrix.

    dataset:
    - "sub-01"
    - "control_mean"
    - "depression_mean"

    matrix_type:
    - "correlation"
    - "partial"
    """

    base_dir = Path(__file__).resolve().parent.parent
    output_dir = base_dir / "connectivity_outputs"

    if dataset == "sub-01":
        prefix = "sub-01_aal3"
    elif dataset == "control_mean":
        prefix = "control_mean_aal3"
    elif dataset == "depression_mean":
        prefix = "depression_mean_aal3"
    else:
        raise ValueError("dataset must be 'sub-01', 'control_mean', or 'depression_mean'")

    if matrix_type == "correlation":
        file_name = f"{prefix}_wc_weight_matrix.csv"
    elif matrix_type == "partial":
        file_name = f"{prefix}_partial_wc_weight_matrix.csv"
    else:
        raise ValueError("matrix_type must be 'correlation' or 'partial'")

    file_path = output_dir / file_name

    if not file_path.exists():
        raise FileNotFoundError(f"Connectivity matrix not found: {file_path}")

    df = pd.read_csv(file_path, index_col=0)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

    return df


def make_network_from_matrix(df, threshold=0.25):
    """
    Convert connectivity matrix to NetworkX graph.
    Only edges with weight >= threshold are included.
    """

    G = nx.Graph()

    for region in df.index:
        G.add_node(region)

    for i, region_i in enumerate(df.index):
        for j, region_j in enumerate(df.columns):
            if j <= i:
                continue

            weight = float(df.iloc[i, j])

            if weight >= threshold:
                G.add_edge(region_i, region_j, weight=weight)

    return G


def compute_centrality(G):
    """
    Compute degree and eigenvector centrality.
    """

    if G is None or len(G.nodes) == 0:
        return {}

    degree = nx.degree_centrality(G)

    try:
        eigen = nx.eigenvector_centrality_numpy(G)
    except Exception:
        eigen = {node: 0 for node in G.nodes}

    centrality = {}

    for node in G.nodes:
        centrality[node] = {
            "degree": degree.get(node, 0),
            "eigenvector": eigen.get(node, 0),
        }

    return centrality


def scale_matrix_for_wc(df, scale=0.35):
    """
    Convert connectivity dataframe to Wilson-Cowan weight matrix.

    Steps:
    1. Convert to numpy array
    2. Remove NaN
    3. Remove negative values
    4. Set diagonal to 0
    5. Normalize 0~1
    6. Scale to Wilson-Cowan input range
    """

    W = df.values.astype(float)

    W = np.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)

    W[W < 0] = 0
    np.fill_diagonal(W, 0)

    if W.max() > 0:
        W = W / W.max()

    W = W * scale

    return W, list(df.index)


def get_available_subjects():
    """
    Return subject IDs that have connectivity output files.
    """

    base_dir = Path(__file__).resolve().parent.parent
    output_dir = base_dir / "connectivity_outputs"

    if not output_dir.exists():
        return []

    subjects = []

    for file in output_dir.glob("*_aal3_wc_weight_matrix.csv"):
        name = file.name
        subject_id = name.split("_aal3_wc_weight_matrix.csv")[0]
        subjects.append(subject_id)

    return sorted(subjects)


def subset_connectivity(df, regions):
    """
    Extract only selected regions from connectivity dataframe.
    """

    available = [r for r in regions if r in df.index and r in df.columns]

    if len(available) < 2:
        return None, available

    return df.loc[available, available], available

def load_group_difference_matrix(matrix_type="correlation"):
    """
    Load control mean and depression mean matrices,
    then compute difference matrix:

    difference = depression - control

    Positive value: stronger in depression group
    Negative value: weaker in depression group
    """

    control_df = load_connectivity_matrix("control_mean", matrix_type)
    depression_df = load_connectivity_matrix("depression_mean", matrix_type)

    common_regions = [
        r for r in control_df.index
        if r in depression_df.index and r in depression_df.columns
    ]

    control_sub = control_df.loc[common_regions, common_regions]
    depression_sub = depression_df.loc[common_regions, common_regions]

    diff_df = depression_sub - control_sub

    return diff_df, control_sub, depression_sub


def make_difference_network(diff_df, threshold=0.10):
    """
    Make NetworkX graph from difference matrix.
    Edges are included if absolute difference >= threshold.
    """

    G = nx.Graph()

    for region in diff_df.index:
        G.add_node(region)

    for i, region_i in enumerate(diff_df.index):
        for j, region_j in enumerate(diff_df.columns):
            if j <= i:
                continue

            diff = float(diff_df.iloc[i, j])

            if abs(diff) >= threshold:
                G.add_edge(region_i, region_j, difference=diff, weight=abs(diff))

    return G

def load_group_difference_matrix(matrix_type="correlation"):
    """
    difference = depression_mean - control_mean
    Positive: stronger in depression
    Negative: weaker in depression
    """
    control_df = load_connectivity_matrix("control_mean", matrix_type)
    depression_df = load_connectivity_matrix("depression_mean", matrix_type)

    common_regions = [
        r for r in control_df.index
        if r in depression_df.index and r in depression_df.columns
    ]

    control_sub = control_df.loc[common_regions, common_regions]
    depression_sub = depression_df.loc[common_regions, common_regions]

    diff_df = depression_sub - control_sub

    return diff_df, control_sub, depression_sub