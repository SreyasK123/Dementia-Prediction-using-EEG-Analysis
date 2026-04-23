"""
EEG Classifier: FTD vs Normal (CN)
Dataset: OpenNeuro ds004504
  - 23 FTD subjects
  - 29 CN (healthy control) subjects
  - 19-channel, 500 Hz, resting-state eyes-closed EEG (.set / BIDS format)

Pipeline:
  1. Download dataset via openneuro-py (or AWS S3)
  2. Load .set files with MNE
  3. Preprocessing: bandpass filter, epoch, artifact rejection
  4. Feature extraction: band power (delta/theta/alpha/beta/gamma), 
     spectral entropy, alpha/theta ratio, connectivity (PLV)
  5. Classification: SVM, Random Forest, XGBoost with cross-validation
  6. Evaluation: accuracy, AUC-ROC, confusion matrix

Requirements:
  pip install mne openneuro-py scipy scikit-learn xgboost matplotlib seaborn pandas numpy tqdm
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

import mne
from mne.preprocessing import ICA
from scipy.signal import welch
from scipy.stats import entropy as scipy_entropy

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, RocCurveDisplay)
from sklearn.inspection import permutation_importance
import xgboost as xgb

warnings.filterwarnings("ignore")
mne.set_log_level("WARNING")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATASET_ID   = "ds004504"
DATA_DIR     = Path("./data/ds004504")          # local cache
OUTPUT_DIR   = Path("./outputs")
CLASSES      = ["FTD", "CN"]                    # binary: FTD vs Normal
SFREQ_TARGET = 256                              # resample target (Hz)
EPOCH_DURATION = 4.0                            # seconds per epoch
EPOCH_OVERLAP  = 2.0                            # overlap (seconds)
FREQ_BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30),
    "gamma": (30, 45),
}
RANDOM_STATE = 42
N_CV_FOLDS   = 5

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# STEP 1: DOWNLOAD DATASET
# ─────────────────────────────────────────────
def download_dataset():
    """
    Download only FTD (sub-073 to sub-095) and CN (sub-001 to sub-029)
    subjects using openneuro-py.
    
    Alternative: AWS CLI
      aws s3 sync --no-sign-request s3://openneuro.org/ds004504 ./data/ds004504 \
        --exclude "*" --include "sub-0[0-2]*" --include "sub-07*" --include "sub-08*" \
        --include "sub-09*" --include "participants.tsv" --include "*.json"
    """
    try:
        import openneuro
        print("Downloading ds004504 (FTD + CN subjects only)...")
        # Download participants.tsv first to identify subject groups
        openneuro.download(
            dataset=DATASET_ID,
            target_dir=str(DATA_DIR),
            include=["participants.tsv", "dataset_description.json"],
        )
        # Full download (filter by participants.tsv after)
        openneuro.download(
            dataset=DATASET_ID,
            target_dir=str(DATA_DIR),
        )
        print(f"Dataset saved to {DATA_DIR}")
    except ImportError:
        print("openneuro-py not installed. Run:")
        print("  pip install openneuro-py")
        print("Or manually download from: https://openneuro.org/datasets/ds004504")
        print("Or via AWS (no sign-in):")
        print("  aws s3 sync --no-sign-request s3://openneuro.org/ds004504 ./data/ds004504")


# ─────────────────────────────────────────────
# STEP 2: LOAD PARTICIPANTS + FILTER FTD & CN
# ─────────────────────────────────────────────
def load_participants(data_dir: Path) -> pd.DataFrame:
    tsv_path = data_dir / "participants.tsv"
    if not tsv_path.exists():
        raise FileNotFoundError(f"participants.tsv not found at {tsv_path}. "
                                 "Please download the dataset first.")
    df = pd.read_csv(tsv_path, sep="\t")
    # Map group letters to expected strings
    df["Group"] = df["Group"].replace({"F": "FTD", "C": "CN"})
    # Keep only FTD and CN groups
    df = df[df["Group"].isin(CLASSES)].reset_index(drop=True)
    print(f"\nSubjects loaded: {df['Group'].value_counts().to_dict()}")
    return df


# ─────────────────────────────────────────────
# STEP 3: LOAD & PREPROCESS ONE SUBJECT
# ─────────────────────────────────────────────
def load_and_preprocess(set_path: str) -> mne.io.Raw | None:
    """Load .set file, apply bandpass filter, resample."""
    try:
        raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose=False)
        # Set standard 10-20 montage
        raw.set_montage("standard_1020", on_missing="ignore")
        # Re-reference to average
        raw.set_eeg_reference("average", projection=False, verbose=False)
        # Bandpass 1–45 Hz
        raw.filter(l_freq=1.0, h_freq=45.0, fir_design="firwin", verbose=False)
        # Notch filter (50 Hz for European data)
        raw.notch_filter(freqs=50.0, verbose=False)
        # Resample
        if raw.info["sfreq"] != SFREQ_TARGET:
            raw.resample(SFREQ_TARGET, verbose=False)
        return raw
    except Exception as e:
        print(f"  [SKIP] {Path(set_path).name}: {e}")
        return None


# ─────────────────────────────────────────────
# STEP 4: EPOCH
# ─────────────────────────────────────────────
def make_epochs(raw: mne.io.Raw) -> np.ndarray:
    """
    Sliding-window epochs. Returns array: (n_epochs, n_channels, n_samples).
    """
    sfreq = raw.info["sfreq"]
    epoch_len   = int(EPOCH_DURATION * sfreq)
    epoch_step  = int((EPOCH_DURATION - EPOCH_OVERLAP) * sfreq)
    data = raw.get_data()  # (n_channels, n_times)
    n_ch, n_times = data.shape
    starts = range(0, n_times - epoch_len + 1, epoch_step)
    epochs = np.stack([data[:, s:s+epoch_len] for s in starts], axis=0)
    return epochs  # (n_epochs, n_ch, epoch_len)


# ─────────────────────────────────────────────
# STEP 5: FEATURE EXTRACTION
# ─────────────────────────────────────────────
def bandpower(signal: np.ndarray, sfreq: float, band: tuple) -> float:
    """Relative band power via Welch's method."""
    f, psd = welch(signal, fs=sfreq, nperseg=min(256, len(signal)//2))
    freq_res = f[1] - f[0]
    band_mask = (f >= band[0]) & (f <= band[1])
    total_mask = (f >= 1) & (f <= 45)
    bp = np.trapezoid(psd[band_mask], dx=freq_res)
    tp = np.trapezoid(psd[total_mask], dx=freq_res)
    return bp / (tp + 1e-12)


def spectral_entropy(signal: np.ndarray, sfreq: float) -> float:
    """Normalized spectral entropy."""
    _, psd = welch(signal, fs=sfreq, nperseg=min(256, len(signal)//2))
    psd_norm = psd / (psd.sum() + 1e-12)
    return scipy_entropy(psd_norm + 1e-12)


def extract_features_epoch(epoch: np.ndarray, sfreq: float,
                            ch_names: list) -> np.ndarray:
    """
    Per-epoch features:
      - Band power (5 bands) per channel → 5 * n_ch
      - Spectral entropy per channel → n_ch
      - Alpha/theta ratio per channel → n_ch
      - Global field power (scalar) → 1
      Total: 7 * n_ch + 1
    """
    n_ch = epoch.shape[0]
    features = []
    for ch_idx in range(n_ch):
        sig = epoch[ch_idx]
        bps = [bandpower(sig, sfreq, b) for b in FREQ_BANDS.values()]
        se  = spectral_entropy(sig, sfreq)
        alpha_theta = (bps[2] / (bps[1] + 1e-12))  # alpha / theta
        features.extend(bps + [se, alpha_theta])
    # Global field power
    gfp = np.std(epoch, axis=0).mean()
    features.append(gfp)
    return np.array(features, dtype=np.float32)


def extract_subject_features(raw: mne.io.Raw) -> np.ndarray:
    """
    Returns mean feature vector across all epochs for one subject.
    Shape: (n_features,)
    """
    epochs = make_epochs(raw)  # (n_epochs, n_ch, n_samples)
    sfreq  = raw.info["sfreq"]
    ch_names = raw.ch_names
    epoch_feats = [extract_features_epoch(ep, sfreq, ch_names) for ep in epochs]
    return np.mean(epoch_feats, axis=0)


def get_feature_names(n_ch: int, ch_names: list) -> list:
    names = []
    for ch in ch_names:
        for band in FREQ_BANDS:
            names.append(f"{ch}_{band}_power")
        names.append(f"{ch}_spectral_entropy")
        names.append(f"{ch}_alpha_theta_ratio")
    names.append("global_field_power")
    return names


# ─────────────────────────────────────────────
# STEP 6: BUILD DATASET
# ─────────────────────────────────────────────
def build_feature_matrix(participants: pd.DataFrame,
                          data_dir: Path) -> tuple[np.ndarray, np.ndarray, list]:
    X_list, y_list = [], []
    ch_names_ref = None

    print("\n[Feature Extraction]")
    for _, row in tqdm(participants.iterrows(), total=len(participants)):
        sub_id = row["participant_id"]   # e.g., "sub-073"
        label  = row["Group"]            # "FTD" or "CN"

        # Locate .set file (BIDS path: sub-XXX/eeg/sub-XXX_task-eyesclosed_eeg.set)
        pattern = str(data_dir / sub_id / "eeg" / f"{sub_id}_*_eeg.set")
        matches = glob.glob(pattern)
        if not matches:
            # Fallback: search recursively
            matches = list(data_dir.rglob(f"{sub_id}_*_eeg.set"))
        if not matches:
            print(f"  [SKIP] {sub_id}: .set file not found")
            continue

        raw = load_and_preprocess(matches[0])
        if raw is None:
            continue

        if ch_names_ref is None:
            ch_names_ref = raw.ch_names

        feats = extract_subject_features(raw)
        X_list.append(feats)
        y_list.append(label)

    X = np.array(X_list)
    y = np.array(y_list)
    feat_names = get_feature_names(len(ch_names_ref), ch_names_ref) if ch_names_ref else []
    print(f"\nFinal dataset: {X.shape[0]} subjects, {X.shape[1]} features")
    print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    return X, y, feat_names


# ─────────────────────────────────────────────
# STEP 7: CLASSIFICATION
# ─────────────────────────────────────────────
def build_classifiers():
    return {
        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    SVC(kernel="rbf", C=1.0, probability=True,
                          random_state=RANDOM_STATE))
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(n_estimators=200,
                                              max_depth=6,
                                              random_state=RANDOM_STATE))
        ]),
        "XGBoost": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    xgb.XGBClassifier(n_estimators=200,
                                         max_depth=4,
                                         learning_rate=0.05,
                                         use_label_encoder=False,
                                         eval_metric="logloss",
                                         random_state=RANDOM_STATE,
                                         verbosity=0))
        ]),
    }


def evaluate_classifiers(X: np.ndarray, y: np.ndarray,
                          feat_names: list) -> dict:
    le = LabelEncoder()
    y_enc = le.fit_transform(y)   # CN=0, FTD=1
    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True,
                         random_state=RANDOM_STATE)
    classifiers = build_classifiers()
    results = {}

    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    fig_cm, axes_cm = plt.subplots(1, len(classifiers), figsize=(5*len(classifiers), 4))

    print(f"\n[Classification — {N_CV_FOLDS}-fold Stratified CV]")
    for i, (name, pipe) in enumerate(classifiers.items()):
        y_pred  = cross_val_predict(pipe, X, y_enc, cv=cv, method="predict")
        y_proba = cross_val_predict(pipe, X, y_enc, cv=cv, method="predict_proba")[:, 1]
        auc     = roc_auc_score(y_enc, y_proba)
        report  = classification_report(y_enc, y_pred,
                                        target_names=le.classes_, output_dict=True)
        results[name] = {"auc": auc, "report": report,
                         "y_pred": y_pred, "y_proba": y_proba}

        print(f"\n  {name}  |  AUC = {auc:.3f}")
        print(classification_report(y_enc, y_pred, target_names=le.classes_))

        # Confusion matrix
        cm = confusion_matrix(y_enc, y_pred)
        ax = axes_cm[i] if len(classifiers) > 1 else axes_cm
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
        ax.set_title(f"{name}\nAUC={auc:.3f}")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")

        # ROC curve
        RocCurveDisplay.from_predictions(
            y_enc, y_proba, name=f"{name} (AUC={auc:.3f})", ax=ax_roc)

    ax_roc.plot([0,1],[0,1],"k--"); ax_roc.set_title("ROC Curves — FTD vs CN")
    fig_roc.tight_layout(); fig_roc.savefig(OUTPUT_DIR/"roc_curves.png", dpi=150)
    fig_cm.tight_layout();  fig_cm.savefig(OUTPUT_DIR/"confusion_matrices.png", dpi=150)
    plt.close("all")
    print(f"\nPlots saved to {OUTPUT_DIR}/")
    return results


# ─────────────────────────────────────────────
# STEP 8: FEATURE IMPORTANCE
# ─────────────────────────────────────────────
def plot_feature_importance(X: np.ndarray, y: np.ndarray, feat_names: list):
    """Fit RF on full data and plot top-20 feature importances."""
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    rf = RandomForestClassifier(n_estimators=300, max_depth=6,
                                random_state=RANDOM_STATE)
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    rf.fit(X_sc, y_enc)

    imp = pd.Series(rf.feature_importances_, index=feat_names)
    top20 = imp.nlargest(20)

    fig, ax = plt.subplots(figsize=(10, 6))
    top20.sort_values().plot.barh(ax=ax, color="steelblue")
    ax.set_title("Top 20 Feature Importances (Random Forest)")
    ax.set_xlabel("Mean Decrease in Impurity")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150)
    plt.close()
    print("Feature importance plot saved.")


# ─────────────────────────────────────────────
# STEP 9: SAVE RESULTS SUMMARY
# ─────────────────────────────────────────────
def save_summary(results: dict):
    rows = []
    for clf_name, res in results.items():
        rpt = res["report"]
        rows.append({
            "Classifier": clf_name,
            "AUC": round(res["auc"], 4),
            "Accuracy": round(rpt["accuracy"], 4),
            "FTD_Precision": round(rpt["FTD"]["precision"], 4),
            "FTD_Recall":    round(rpt["FTD"]["recall"], 4),
            "FTD_F1":        round(rpt["FTD"]["f1-score"], 4),
            "CN_Precision":  round(rpt["CN"]["precision"], 4),
            "CN_Recall":     round(rpt["CN"]["recall"], 4),
            "CN_F1":         round(rpt["CN"]["f1-score"], 4),
        })
    df = pd.DataFrame(rows)
    out = OUTPUT_DIR / "classification_summary.csv"
    df.to_csv(out, index=False)
    print(f"\nSummary saved to {out}")
    print(df.to_string(index=False))


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print(" EEG Classifier: FTD vs Normal  |  OpenNeuro ds004504")
    print("=" * 60)

    # 1. Download (skip if already present)
    if not DATA_DIR.exists() or not any(DATA_DIR.iterdir()):
        download_dataset()
    else:
        print(f"Data directory found: {DATA_DIR}")

    # 2. Load participants
    participants = load_participants(DATA_DIR)

    # 3. Build feature matrix
    X, y, feat_names = build_feature_matrix(participants, DATA_DIR)

    if len(X) < 10:
        print("\n[WARNING] Very few subjects loaded. "
              "Ensure the dataset is downloaded and .set files are present.")
        return

    # 4. Save features
    feat_df = pd.DataFrame(X, columns=feat_names if feat_names else
                           [f"f{i}" for i in range(X.shape[1])])
    feat_df.insert(0, "label", y)
    feat_df.to_csv(Path("data/features.csv"), index=False)

    # 5. Evaluate classifiers
    results = evaluate_classifiers(X, y, feat_names)

    # 6. Feature importance
    if feat_names:
        plot_feature_importance(X, y, feat_names)

    # 7. Summary
    save_summary(results)

    print("\n[Done] All outputs saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()