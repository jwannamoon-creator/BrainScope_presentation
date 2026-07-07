from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure


# =========================
# 1. Auto path setting
# =========================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

SUBJECT_ID = "sub-01"

BOLD_PATH = PROJECT_ROOT / SUBJECT_ID / "func" / f"{SUBJECT_ID}_task-rest_bold.nii.gz"

OUTPUT_DIR = SCRIPT_DIR / "connectivity_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_PREFIX = OUTPUT_DIR / f"{SUBJECT_ID}_aal3"

print("=" * 60)
print("PROJECT_ROOT:", PROJECT_ROOT)
print("BOLD_PATH:", BOLD_PATH)
print("BOLD file exists:", BOLD_PATH.exists())
print("=" * 60)

if not BOLD_PATH.exists():
    raise FileNotFoundError(f"BOLD file not found: {BOLD_PATH}")


# =========================
# 2. Load AAL3 atlas
# =========================

aal = datasets.fetch_atlas_aal(version="3v2")

atlas_img = aal.maps
labels = list(aal.labels)
indices = [int(x) for x in aal.indices]

print("\nAAL3 atlas loaded")
print("Number of labels:", len(labels))
print("Number of indices:", len(indices))


# =========================
# 3. Define refined target regions
# =========================

target_keywords = {
    # Emotion circuit
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
    "Amygdala": [
        "Amygdala",
    ],
    "Insula": [
        "Insula",
    ],
    "Hippocampus": [
        "Hippocampus",
    ],

    # Memory circuit
    "dlPFC": [
        "Frontal_Mid_2",
        "Frontal_Sup_2",
    ],
    "Parahippocampal": [
        "ParaHippocampal",
    ],
    "PCC": [
        "Cingulate_Post",
        "Cingulum_Post",
        "PCC",
    ],
    "Angular": [
        "Angular",
    ],
}


def find_label_indices(keyword_list):
    selected = []

    for label, index in zip(labels, indices):
        for keyword in keyword_list:
            if keyword in label:
                selected.append((label, index))
                break

    return selected


roi_map = {}

print("\nSelected AAL3 labels:")
for region_name, keywords in target_keywords.items():
    matched = find_label_indices(keywords)
    roi_map[region_name] = matched

    print(f"\n[{region_name}]")
    if not matched:
        print("  No labels matched.")
    for label, index in matched:
        print(f"  {label} {index}")


# =========================
# 4. Extract all AAL3 ROI time series
# =========================

masker = NiftiLabelsMasker(
    labels_img=atlas_img,
    standardize=True,
    detrend=True,
    low_pass=0.1,
    high_pass=0.01,
    t_r=2.0,
    verbose=1,
)

print("\nExtracting AAL3 ROI time series...")
all_time_series = masker.fit_transform(str(BOLD_PATH))

print("\nTime series shape:", all_time_series.shape)
print("Meaning: timepoints x atlas regions")


# =========================
# 5. Convert atlas labels to columns
# =========================

def get_columns_for_region(matched_labels):
    cols = []
    used_labels = []

    for label, index in matched_labels:
        if label in labels:
            col = labels.index(label)
            if col < all_time_series.shape[1]:
                cols.append(col)
                used_labels.append(label)

    return cols, used_labels


# =========================
# 6. Build reduced regional time series
# =========================

reduced_ts = {}
used_label_record = {}

for region_name, matched in roi_map.items():
    cols, used = get_columns_for_region(matched)

    if len(cols) == 0:
        print(f"WARNING: No valid columns for {region_name}. Skipping.")
        continue

    reduced_ts[region_name] = all_time_series[:, cols].mean(axis=1)
    used_label_record[region_name] = used

region_names = list(reduced_ts.keys())
X = np.column_stack([reduced_ts[name] for name in region_names])

print("\nReduced time series shape:", X.shape)
print("Regions:", region_names)


# =========================
# 7. Connectivity calculation
# =========================

def make_wc_weight_matrix(matrix):
    W = matrix.copy()

    np.fill_diagonal(W, 0)

    # 음수 연결은 Wilson-Cowan의 흥분성 network input으로 직접 쓰기 어렵기 때문에 0으로 절단
    W[W < 0] = 0

    # 0~1 정규화
    if W.max() > 0:
        W = W / W.max()

    return W


# 7-1. Correlation
corr_measure = ConnectivityMeasure(kind="correlation")
corr_matrix = corr_measure.fit_transform([X])[0]
np.fill_diagonal(corr_matrix, 0)
corr_W = make_wc_weight_matrix(corr_matrix)

# 7-2. Partial correlation
partial_measure = ConnectivityMeasure(kind="partial correlation")
partial_matrix = partial_measure.fit_transform([X])[0]
np.fill_diagonal(partial_matrix, 0)
partial_W = make_wc_weight_matrix(partial_matrix)


# =========================
# 8. Save results
# =========================

def save_matrix(matrix, filename):
    path = Path(str(OUTPUT_PREFIX) + filename)
    pd.DataFrame(
        matrix,
        index=region_names,
        columns=region_names,
    ).to_csv(path, encoding="utf-8-sig")
    print(path)
    return path


def save_heatmap(matrix, filename, title):
    path = Path(str(OUTPUT_PREFIX) + filename)

    plt.figure(figsize=(7, 6))
    plt.imshow(matrix, cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(label="Normalized connectivity")
    plt.xticks(range(len(region_names)), region_names, rotation=45, ha="right")
    plt.yticks(range(len(region_names)), region_names)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

    print(path)
    return path


print("\nSaved files:")

# Correlation outputs
save_matrix(corr_matrix, "_correlation_matrix.csv")
save_matrix(corr_W, "_wc_weight_matrix.csv")
save_heatmap(
    corr_W,
    "_wc_weight_matrix.png",
    f"{SUBJECT_ID} AAL3 Correlation-based WC Weight Matrix"
)

# Partial correlation outputs
save_matrix(partial_matrix, "_partial_correlation_matrix.csv")
save_matrix(partial_W, "_partial_wc_weight_matrix.csv")
save_heatmap(
    partial_W,
    "_partial_wc_weight_matrix.png",
    f"{SUBJECT_ID} AAL3 Partial-correlation-based WC Weight Matrix"
)

# ROI time series
ts_path = Path(str(OUTPUT_PREFIX) + "_roi_timeseries.csv")
pd.DataFrame(
    X,
    columns=region_names,
).to_csv(ts_path, index=False, encoding="utf-8-sig")
print(ts_path)

# Used labels
labels_path = Path(str(OUTPUT_PREFIX) + "_used_labels.txt")
with open(labels_path, "w", encoding="utf-8") as f:
    for region_name, used in used_label_record.items():
        f.write(f"[{region_name}]\n")
        for label in used:
            f.write(f"{label}\n")
        f.write("\n")
print(labels_path)

print("\nDone.")