from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

PARTICIPANTS_PATH = PROJECT_ROOT / "participants.tsv"
OUTPUT_DIR = SCRIPT_DIR / "connectivity_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

print("PROJECT_ROOT:", PROJECT_ROOT)
print("participants.tsv exists:", PARTICIPANTS_PATH.exists())

participants = pd.read_csv(PARTICIPANTS_PATH, sep="\t")

print(participants[["participant_id", "group"]].head())
print(participants["group"].value_counts())


aal = datasets.fetch_atlas_aal(version="3v2")
atlas_img = aal.maps
labels = list(aal.labels)
indices = [int(x) for x in aal.indices]


target_keywords = {
    "vmPFC_OFC": [
        "Frontal_Med_Orb",
        "Frontal_Sup_Orb",
        "Frontal_Mid_Orb",
        "Frontal_Inf_Orb",
    ],
    "ACC": [
        "ACC_sub",
        "ACC_pre",
        "ACC_sup",
        "Cingulate_Ant",
        "Cingulum_Ant",
        "Cingulate",
    ],
    "Amygdala": ["Amygdala"],
    "Insula": ["Insula"],
    "Hippocampus": ["Hippocampus"],
    "dlPFC": ["Frontal_Mid_2", "Frontal_Sup_2"],
    "Parahippocampal": ["ParaHippocampal"],
    "PCC": ["Cingulate_Post", "Cingulum_Post", "PCC"],
    "Angular": ["Angular"],
}


def find_label_indices(keyword_list):
    selected = []
    for label, index in zip(labels, indices):
        for keyword in keyword_list:
            if keyword in label:
                selected.append((label, index))
                break
    return selected


roi_map = {
    region_name: find_label_indices(keywords)
    for region_name, keywords in target_keywords.items()
}


print("\nSelected labels:")
for region, matched in roi_map.items():
    print(f"\n[{region}]")
    for label, index in matched:
        print(label, index)


masker = NiftiLabelsMasker(
    labels_img=atlas_img,
    standardize=True,
    detrend=True,
    low_pass=0.1,
    high_pass=0.01,
    t_r=2.0,
    verbose=0,
)


def get_columns_for_region(matched_labels, all_time_series):
    cols = []
    used_labels = []

    for label, index in matched_labels:
        if label in labels:
            col = labels.index(label)
            if col < all_time_series.shape[1]:
                cols.append(col)
                used_labels.append(label)

    return cols, used_labels


def make_wc_weight_matrix(matrix):
    W = matrix.copy()
    np.fill_diagonal(W, 0)
    W[W < 0] = 0

    if W.max() > 0:
        W = W / W.max()

    return W


def process_subject(subject_id):
    bold_path = PROJECT_ROOT / subject_id / "func" / f"{subject_id}_task-rest_bold.nii.gz"

    if not bold_path.exists():
        print(f"[SKIP] Missing BOLD: {subject_id}")
        return None

    print(f"[PROCESS] {subject_id}")

    try:
        all_time_series = masker.fit_transform(str(bold_path))

        reduced_ts = {}

        for region_name, matched in roi_map.items():
            cols, used = get_columns_for_region(matched, all_time_series)

            if len(cols) == 0:
                print(f"  WARNING: no ROI columns for {region_name}")
                continue

            reduced_ts[region_name] = all_time_series[:, cols].mean(axis=1)

        region_names = list(reduced_ts.keys())
        X = np.column_stack([reduced_ts[name] for name in region_names])

        corr_measure = ConnectivityMeasure(kind="correlation")
        corr_matrix = corr_measure.fit_transform([X])[0]
        np.fill_diagonal(corr_matrix, 0)

        partial_measure = ConnectivityMeasure(kind="partial correlation")
        partial_matrix = partial_measure.fit_transform([X])[0]
        np.fill_diagonal(partial_matrix, 0)

        corr_W = make_wc_weight_matrix(corr_matrix)
        partial_W = make_wc_weight_matrix(partial_matrix)

        return {
            "subject_id": subject_id,
            "region_names": region_names,
            "corr": corr_matrix,
            "partial": partial_matrix,
            "corr_W": corr_W,
            "partial_W": partial_W,
        }

    except Exception as e:
        print(f"[ERROR] {subject_id}: {e}")
        return None


def save_matrix(matrix, region_names, filename):
    path = OUTPUT_DIR / filename
    pd.DataFrame(matrix, index=region_names, columns=region_names).to_csv(
        path, encoding="utf-8-sig"
    )
    print("Saved:", path)


def save_heatmap(matrix, region_names, filename, title):
    path = OUTPUT_DIR / filename

    plt.figure(figsize=(7, 6))
    plt.imshow(matrix, cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(label="Normalized connectivity")
    plt.xticks(range(len(region_names)), region_names, rotation=45, ha="right")
    plt.yticks(range(len(region_names)), region_names)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

    print("Saved:", path)


def average_group(results, group_name):
    group_results = [r for r in results if r is not None]

    if len(group_results) == 0:
        print(f"No valid subjects for {group_name}")
        return

    region_names = group_results[0]["region_names"]

    corr_stack = np.stack([r["corr"] for r in group_results], axis=0)
    partial_stack = np.stack([r["partial"] for r in group_results], axis=0)
    corr_W_stack = np.stack([r["corr_W"] for r in group_results], axis=0)
    partial_W_stack = np.stack([r["partial_W"] for r in group_results], axis=0)

    corr_mean = corr_stack.mean(axis=0)
    partial_mean = partial_stack.mean(axis=0)
    corr_W_mean = corr_W_stack.mean(axis=0)
    partial_W_mean = partial_W_stack.mean(axis=0)

    save_matrix(corr_mean, region_names, f"{group_name}_mean_aal3_correlation_matrix.csv")
    save_matrix(corr_W_mean, region_names, f"{group_name}_mean_aal3_wc_weight_matrix.csv")
    save_matrix(partial_mean, region_names, f"{group_name}_mean_aal3_partial_correlation_matrix.csv")
    save_matrix(partial_W_mean, region_names, f"{group_name}_mean_aal3_partial_wc_weight_matrix.csv")

    save_heatmap(
        corr_W_mean,
        region_names,
        f"{group_name}_mean_aal3_wc_weight_matrix.png",
        f"{group_name} mean correlation-based WC weight matrix",
    )

    save_heatmap(
        partial_W_mean,
        region_names,
        f"{group_name}_mean_aal3_partial_wc_weight_matrix.png",
        f"{group_name} mean partial-correlation-based WC weight matrix",
    )


depression_subjects = participants.loc[
    participants["group"] == "depr", "participant_id"
].tolist()

control_subjects = participants.loc[
    participants["group"] == "control", "participant_id"
].tolist()

print("\nDepression subjects:", len(depression_subjects))
print("Control subjects:", len(control_subjects))

depression_results = [process_subject(s) for s in depression_subjects]
control_results = [process_subject(s) for s in control_subjects]

average_group(depression_results, "depression")
average_group(control_results, "control")

print("\nDone.")