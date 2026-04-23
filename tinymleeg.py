"""
TinyML EEG Classifier: FTD vs Normal — Edge Device Edition
===========================================================
Builds the smallest possible classifier that still achieves
good accuracy, then exports it for microcontrollers / RPi / ESP32.

Compression pipeline:
  1. Feature reduction   — SelectKBest + optional PCA
  2. Tiny model          — Decision Tree or Linear SVM or MLP (1 layer)
  3. Quantization        — float32 → int8 (weights only)
  4. Export              — C array (Arduino), ONNX, or plain C via m2cgen

Requirements:
  pip install scikit-learn scipy numpy pandas matplotlib seaborn
  pip install onnx skl2onnx m2cgen micromlgen
  pip install mne tqdm   (only needed if re-extracting features)

Usage:
  # If you already ran eeg_ftd_classifier.py and have features.csv:
  python tinyml_eeg_classifier.py --features outputs/features.csv

  # Full re-run from raw .set files:
  python tinyml_eeg_classifier.py --data ./data/ds004504
"""

import argparse
import json
import struct
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, RocCurveDisplay)
import joblib

warnings.filterwarnings("ignore")

OUTPUT_DIR   = Path("./outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_CV_FOLDS   = 5
TOP_K_FEATURES = 20   # start here — tune down to 10 for MCU

# ─────────────────────────────────────────────────────
# STAGE 1 — FEATURE REDUCTION
# ─────────────────────────────────────────────────────

def select_features(X: np.ndarray, y_enc: np.ndarray,
                    feat_names: list, k: int = TOP_K_FEATURES):
    """
    SelectKBest with mutual information.
    Returns reduced X, selected feature names, and fitted selector.
    """
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    X_sel = selector.fit_transform(X, y_enc)
    mask = selector.get_support()
    sel_names = [n for n, m in zip(feat_names, mask) if m]

    scores = selector.scores_
    top_idx = np.argsort(scores)[::-1][:k]
    top_scores = scores[top_idx]
    top_names  = [feat_names[i] for i in top_idx]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(top_names[::-1], top_scores[np.argsort(top_scores)], color="steelblue")
    ax.set_title(f"Top {k} features by mutual information (FTD vs CN)")
    ax.set_xlabel("Mutual information score")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "tinyml_feature_scores.png", dpi=150)
    plt.close()

    print(f"\n[Feature selection] {X.shape[1]} → {X_sel.shape[1]} features")
    print("Top 5:", top_names[:5])
    return X_sel, sel_names, selector


def apply_pca(X: np.ndarray, variance: float = 0.95):
    """Optional: further compress with PCA keeping `variance` % explained."""
    pca = PCA(n_components=variance, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X)
    print(f"[PCA] {X.shape[1]} → {X_pca.shape[1]} components "
          f"({variance*100:.0f}% variance retained)")
    return X_pca, pca


# ─────────────────────────────────────────────────────
# STAGE 2 — TINY MODELS
# ─────────────────────────────────────────────────────

def build_tiny_models(max_depth: int = 8):
    """
    Three tiny model options ranked by size:
      - Decision Tree  → smallest, easiest to export to C
      - Linear SVM     → very small (just a weight vector)
      - MLP (1 layer)  → most accurate, still small
    """
    return {
        "Decision tree (depth 8)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    DecisionTreeClassifier(max_depth=max_depth,
                                              class_weight="balanced",
                                              random_state=RANDOM_STATE))
        ]),
        "Linear SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LinearSVC(C=0.5, max_iter=5000,
                                 class_weight="balanced",
                                 random_state=RANDOM_STATE))
        ]),
        "MLP (32 hidden)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    MLPClassifier(hidden_layer_sizes=(32,),
                                     activation="relu",
                                     max_iter=500,
                                     early_stopping=True,
                                     random_state=RANDOM_STATE))
        ]),
    }


def evaluate_tiny_models(X: np.ndarray, y: np.ndarray,
                          feat_names: list) -> dict:
    le = LabelEncoder()
    y_enc = le.fit_transform(y)   # CN=0, FTD=1
    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True,
                         random_state=RANDOM_STATE)
    models = build_tiny_models()
    results = {}

    fig_cm, axes_cm = plt.subplots(1, len(models),
                                    figsize=(5*len(models), 4))
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))

    print(f"\n[Tiny model evaluation — {N_CV_FOLDS}-fold CV]")
    for i, (name, pipe) in enumerate(models.items()):
        y_pred = cross_val_predict(pipe, X, y_enc, cv=cv)

        # AUC: LinearSVC doesn't have predict_proba, use decision_function
        if "Linear SVM" in name:
            scores = cross_val_predict(pipe, X, y_enc, cv=cv,
                                       method="decision_function")
            auc = roc_auc_score(y_enc, scores)
            RocCurveDisplay.from_predictions(y_enc, scores,
                                             name=f"{name} AUC={auc:.3f}",
                                             ax=ax_roc)
        else:
            proba = cross_val_predict(pipe, X, y_enc, cv=cv,
                                      method="predict_proba")[:, 1]
            auc = roc_auc_score(y_enc, proba)
            RocCurveDisplay.from_predictions(y_enc, proba,
                                             name=f"{name} AUC={auc:.3f}",
                                             ax=ax_roc)

        results[name] = {"auc": auc, "y_pred": y_pred,
                         "le": le, "pipe": pipe}
        report = classification_report(y_enc, y_pred,
                                       target_names=le.classes_, output_dict=True)
        results[name]["report"] = report

        print(f"\n  {name}  |  AUC = {auc:.3f}")
        print(classification_report(y_enc, y_pred, target_names=le.classes_))

        cm = confusion_matrix(y_enc, y_pred)
        ax = axes_cm[i]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
        ax.set_title(f"{name}\nAUC={auc:.3f}")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")

    ax_roc.plot([0,1],[0,1],"k--")
    ax_roc.set_title("ROC Curves — Tiny models")
    fig_roc.tight_layout(); fig_roc.savefig(OUTPUT_DIR/"tinyml_roc.png", dpi=150)
    fig_cm.tight_layout();  fig_cm.savefig(OUTPUT_DIR/"tinyml_cm.png", dpi=150)
    plt.close("all")
    return results, le, y_enc


# ─────────────────────────────────────────────────────
# STAGE 3 — QUANTIZATION (float32 → int8)
# ─────────────────────────────────────────────────────

def quantize_weights(weights: np.ndarray, bits: int = 8):
    """
    Symmetric linear quantization.
    Returns: quantized int8 array, scale factor, zero point.
    """
    w_max = np.max(np.abs(weights))
    scale = w_max / (2**(bits-1) - 1)          # float per quant unit
    zero_point = 0                               # symmetric
    w_q = np.round(weights / scale).astype(np.int8)
    # Dequantized: w_q * scale ≈ original weights
    return w_q, scale, zero_point


def quantize_scaler_params(scaler: StandardScaler, bits: int = 8):
    """Quantize the scaler mean and std to int8 for MCU inference."""
    mean_q, mean_s, _ = quantize_weights(scaler.mean_, bits)
    std_q,  std_s, _  = quantize_weights(scaler.scale_, bits)
    return {"mean_q": mean_q.tolist(), "mean_scale": float(mean_s),
            "std_q":  std_q.tolist(),  "std_scale":  float(std_s)}


def quantize_decision_tree(clf: DecisionTreeClassifier) -> dict:
    """Extract DT structure with quantized thresholds (int16)."""
    tree = clf.tree_
    # Thresholds: scale to int16 (already normalized by StandardScaler upstream)
    thresh = tree.threshold.copy()
    valid = thresh > -2   # leaf nodes have threshold = -2
    t_max = np.max(np.abs(thresh[valid])) if valid.any() else 1.0
    scale = t_max / 32767.0
    thresh_q = np.zeros_like(thresh, dtype=np.int16)
    thresh_q[valid] = np.round(thresh[valid] / scale).astype(np.int16)
    return {
        "n_nodes":       int(tree.node_count),
        "children_left": tree.children_left.tolist(),
        "children_right":tree.children_right.tolist(),
        "feature":       tree.feature.tolist(),
        "threshold_q":   thresh_q.tolist(),
        "threshold_scale": float(scale),
        "value":         tree.value.tolist(),
        "n_features":    int(tree.n_features_in_),
        "n_classes":     int(tree.n_classes_),
    }


def estimate_model_size(pipe: Pipeline, model_name: str) -> dict:
    """Estimate RAM and flash footprint in bytes."""
    clf = pipe.named_steps["clf"]
    scaler = pipe.named_steps["scaler"]
    n_feat = scaler.mean_.shape[0]

    if isinstance(clf, DecisionTreeClassifier):
        n_nodes = clf.tree_.node_count
        # Each node: left(4B) + right(4B) + feature(2B) + threshold(4B) = 14B
        model_bytes = n_nodes * 14
        inference_ops = clf.get_depth() * 2   # comparisons
    elif isinstance(clf, LinearSVC):
        n_coef = clf.coef_.size + clf.intercept_.size
        model_bytes = n_coef * 4   # float32
        inference_ops = n_feat     # dot product
    elif isinstance(clf, MLPClassifier):
        total = sum(w.size for w in clf.coefs_)
        total += sum(b.size for b in clf.intercepts_)
        model_bytes = total * 4   # float32
        inference_ops = total
    else:
        model_bytes = 0
        inference_ops = 0

    scaler_bytes = n_feat * 2 * 4   # mean + std, float32
    feature_bytes = n_feat * 4       # one sample buffer

    return {
        "model_name":    model_name,
        "n_features":    n_feat,
        "model_bytes":   model_bytes,
        "scaler_bytes":  scaler_bytes,
        "feature_buffer": feature_bytes,
        "total_bytes":   model_bytes + scaler_bytes + feature_bytes,
        "approx_ops":    inference_ops,
    }


# ─────────────────────────────────────────────────────
# STAGE 4 — EDGE EXPORT
# ─────────────────────────────────────────────────────

def export_to_c_array(pipe: Pipeline, feat_names: list,
                       label_names: list, filename: str = "eeg_model.h"):
    """
    Export a decision tree OR linear SVM to a self-contained C header.
    Works on Arduino, STM32, ESP32 (no external libraries needed).
    """
    clf = pipe.named_steps["clf"]
    scaler = pipe.named_steps["scaler"]
    n_feat = len(feat_names)

    lines = []
    lines.append("/* Auto-generated EEG TinyML classifier */")
    lines.append("/* Target: Arduino / STM32 / ESP32        */")
    lines.append("#pragma once")
    lines.append('#include <stdint.h>')
    lines.append("")
    lines.append(f"#define N_FEATURES {n_feat}")
    lines.append(f"#define N_CLASSES  {len(label_names)}")
    lines.append("")

    # Scaler (float arrays)
    lines.append("/* StandardScaler parameters */")
    mean_str = ", ".join(f"{v:.6f}f" for v in scaler.mean_)
    std_str  = ", ".join(f"{v:.6f}f" for v in scaler.scale_)
    lines.append(f"const float SCALER_MEAN[{n_feat}] = {{{mean_str}}};")
    lines.append(f"const float SCALER_STD[{n_feat}]  = {{{std_str}}};")
    lines.append("")

    if isinstance(clf, DecisionTreeClassifier):
        tree = clf.tree_
        n_nodes = tree.node_count
        lines.append("/* Decision tree structure */")
        lines.append(f"#define N_NODES {n_nodes}")
        left_s  = ", ".join(str(v) for v in tree.children_left)
        right_s = ", ".join(str(v) for v in tree.children_right)
        feat_s  = ", ".join(str(v) for v in tree.feature)
        thr_s   = ", ".join(f"{v:.6f}f" for v in tree.threshold)
        # leaf class: argmax of value
        leaf_classes = np.argmax(tree.value[:, 0, :], axis=1)
        leaf_s = ", ".join(str(v) for v in leaf_classes)
        lines.append(f"const int16_t DT_LEFT[{n_nodes}]    = {{{left_s}}};")
        lines.append(f"const int16_t DT_RIGHT[{n_nodes}]   = {{{right_s}}};")
        lines.append(f"const int16_t DT_FEATURE[{n_nodes}] = {{{feat_s}}};")
        lines.append(f"const float   DT_THRESH[{n_nodes}]  = {{{thr_s}}};")
        lines.append(f"const uint8_t DT_CLASS[{n_nodes}]   = {{{leaf_s}}};")
        lines.append("")
        lines.append("/* Inference function */")
        lines.append("static inline uint8_t eeg_predict(const float* x) {")
        lines.append("  /* Normalize */")
        lines.append("  float xn[N_FEATURES];")
        lines.append("  for (int i = 0; i < N_FEATURES; i++)")
        lines.append("    xn[i] = (x[i] - SCALER_MEAN[i]) / SCALER_STD[i];")
        lines.append("  /* Traverse tree */")
        lines.append("  int node = 0;")
        lines.append("  while (DT_LEFT[node] != -1) {")
        lines.append("    if (xn[DT_FEATURE[node]] <= DT_THRESH[node])")
        lines.append("      node = DT_LEFT[node];")
        lines.append("    else")
        lines.append("      node = DT_RIGHT[node];")
        lines.append("  }")
        lines.append("  return DT_CLASS[node];  /* 0=CN, 1=FTD */")
        lines.append("}")

    elif isinstance(clf, LinearSVC):
        coef = clf.coef_[0]
        intercept = clf.intercept_[0]
        coef_s = ", ".join(f"{v:.6f}f" for v in coef)
        lines.append("/* Linear SVM weights */")
        lines.append(f"const float SVM_COEF[{n_feat}] = {{{coef_s}}};")
        lines.append(f"const float SVM_BIAS = {intercept:.6f}f;")
        lines.append("")
        lines.append("static inline uint8_t eeg_predict(const float* x) {")
        lines.append("  float xn[N_FEATURES];")
        lines.append("  for (int i = 0; i < N_FEATURES; i++)")
        lines.append("    xn[i] = (x[i] - SCALER_MEAN[i]) / SCALER_STD[i];")
        lines.append("  float score = SVM_BIAS;")
        lines.append("  for (int i = 0; i < N_FEATURES; i++)")
        lines.append("    score += SVM_COEF[i] * xn[i];")
        lines.append("  return score > 0 ? 1 : 0;  /* 0=CN, 1=FTD */")
        lines.append("}")

    lines.append("")
    lines.append(f'/* Label names: {label_names} */') 
    lines.append('const char* CLASS_NAMES[] = {' +
                 ", ".join(f'"{n}"' for n in label_names) + '};')

    out_path = OUTPUT_DIR / filename
    out_path.write_text("\n".join(lines))
    print(f"\n[Export] C header saved: {out_path}")
    return out_path


def export_to_onnx(pipe: Pipeline, n_features: int,
                    filename: str = "eeg_model.onnx"):
    """Export to ONNX for ONNX Runtime on Raspberry Pi / Jetson."""
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        initial_type = [("float_input", FloatTensorType([None, n_features]))]
        # LinearSVC doesn't support zipmap probabilities, so we must request raw_scores instead
        options = {}
        from sklearn.svm import LinearSVC
        if isinstance(pipe.named_steps.get("clf"), LinearSVC):
            options = {LinearSVC: {'raw_scores': True}}
            
        onnx_model = convert_sklearn(pipe, initial_types=initial_type, options=options)
        out_path = OUTPUT_DIR / filename
        with open(out_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        size_kb = out_path.stat().st_size / 1024
        print(f"[Export] ONNX model saved: {out_path}  ({size_kb:.1f} KB)")
        return out_path
    except ImportError:
        print("[Export] skl2onnx not installed. Run: pip install skl2onnx")
        return None


def export_to_pure_c(pipe: Pipeline, filename: str = "eeg_model_m2c.c"):
    """
    Export using m2cgen — generates pure C code, no sklearn needed at runtime.
    Ideal for bare-metal MCU deployment.
    """
    try:
        import m2cgen as m2c
        clf = pipe.named_steps["clf"]
        # m2cgen works on the classifier directly (after we bake scaler in)
        code = m2c.export_to_c(clf)
        out_path = OUTPUT_DIR / filename
        header = ("/* m2cgen pure-C export: deploy with no dependencies */\n"
                  "/* Link with your scaler code to pre-normalize inputs  */\n\n")
        out_path.write_text(header + code)
        print(f"[Export] m2cgen C code saved: {out_path}")
        return out_path
    except ImportError:
        print("[Export] m2cgen not installed. Run: pip install m2cgen")
        return None


def print_model_size_report(sizes: list[dict]):
    print("\n" + "="*65)
    print(f"{'Model':<25} {'Features':>8} {'Model':>8} {'Scaler':>8} "
          f"{'Buffer':>8} {'Total':>8} {'Ops':>8}")
    print("-"*65)
    for s in sizes:
        print(f"{s['model_name']:<25} {s['n_features']:>8} "
              f"{s['model_bytes']:>7}B {s['scaler_bytes']:>7}B "
              f"{s['feature_buffer']:>7}B {s['total_bytes']:>7}B "
              f"{s['approx_ops']:>8}")
    print("="*65)
    print("Arduino Uno SRAM: 2 KB  |  ESP32: 520 KB  |  STM32F4: 192 KB")


def save_tinyml_summary(results: dict, sizes: list, use_pca: bool):
    rows = []
    for name, res in results.items():
        rpt = res["report"]
        size = next((s for s in sizes if s["model_name"] == name), {})
        rows.append({
            "Model": name,
            "AUC":   round(res["auc"], 4),
            "Accuracy": round(rpt["accuracy"], 4),
            "FTD_F1":  round(rpt.get("FTD", {}).get("f1-score", 0), 4),
            "Total_bytes": size.get("total_bytes", "?"),
            "Inference_ops": size.get("approx_ops", "?"),
            "PCA_used": use_pca,
        })
    df = pd.DataFrame(rows)
    out = OUTPUT_DIR / "tinyml_summary.csv"
    df.to_csv(out, index=False)
    print(f"\nSummary → {out}")
    print(df.to_string(index=False))


# ─────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TinyML EEG FTD vs CN")
    parser.add_argument("--k", type=int, default=TOP_K_FEATURES,
                        help="Number of features to keep (default: 20)")
    parser.add_argument("--depth", type=int, default=8,
                        help="Max decision tree depth (default: 8)")
    parser.add_argument("--pca", action="store_true",
                        help="Apply PCA after feature selection")
    parser.add_argument("--export", choices=["c", "onnx", "m2c", "all"],
                        default="c", help="Export format (default: c)")
    args = parser.parse_args()

    print("=" * 60)
    print(" TinyML EEG Classifier: FTD vs Normal")
    print("=" * 60)

    # ── Load features ────────────────────────────────
    feat_path = Path("data/features.csv")
    if not feat_path.exists():
        print(f"[ERROR] features.csv not found at {feat_path}")
        print("Run eeg_ftd_classifier.py first to extract features.")
        return

    df = pd.read_csv(feat_path)
    y  = df["label"].values
    X  = df.drop(columns=["label"]).values
    feat_names = list(df.drop(columns=["label"]).columns)
    print(f"\nLoaded: {X.shape[0]} subjects, {X.shape[1]} features")
    print(f"Classes: {dict(zip(*np.unique(y, return_counts=True)))}")

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # ── Stage 1: Feature reduction ───────────────────
    X_sel, sel_names, selector = select_features(X, y_enc, feat_names, k=args.k)

    if args.pca:
        X_sel, pca = apply_pca(X_sel, variance=0.95)
        sel_names = [f"PC{i+1}" for i in range(X_sel.shape[1])]

    # ── Stage 2: Tiny model evaluation ───────────────
    results, le2, y_enc2 = evaluate_tiny_models(X_sel, y, sel_names)

    # ── Stage 3: Size estimation ─────────────────────
    sizes = []
    for name, res in results.items():
        pipe = res["pipe"]
        pipe.fit(X_sel, y_enc2)   # refit on full data for export
        sz = estimate_model_size(pipe, name)
        sizes.append(sz)
    print_model_size_report(sizes)

    # ── Stage 4: Export best model ───────────────────
    # Pick best by AUC
    best_name = max(results, key=lambda n: results[n]["auc"])
    best_pipe  = results[best_name]["pipe"]
    print(f"\n[Export] Best model: {best_name} (AUC={results[best_name]['auc']:.3f})")
    label_names = list(le2.classes_)

    if args.export in ("c", "all"):
        export_to_c_array(best_pipe, sel_names, label_names, "eeg_model.h")

    if args.export in ("onnx", "all"):
        export_to_onnx(best_pipe, X_sel.shape[1], "eeg_model.onnx")

    if args.export in ("m2c", "all"):
        export_to_pure_c(best_pipe, "eeg_model_m2c.c")

    # Decision tree text dump (human-readable)
    clf = best_pipe.named_steps["clf"]
    if isinstance(clf, DecisionTreeClassifier):
        tree_text = export_text(clf, feature_names=sel_names[:clf.n_features_in_])
        (OUTPUT_DIR / "decision_tree_rules.txt").write_text(tree_text)
        print("[Export] Decision tree rules → outputs/decision_tree_rules.txt")

    # Save sklearn pipeline
    joblib.dump(best_pipe, OUTPUT_DIR / "tinyml_pipeline.joblib")
    print(f"[Export] sklearn pipeline → outputs/tinyml_pipeline.joblib")

    # ── Summary ──────────────────────────────────────
    save_tinyml_summary(results, sizes, use_pca=args.pca)

    # Print feature budget for common MCUs
    n_feat = X_sel.shape[1]
    print(f"\n[Memory breakdown for {n_feat} features (float32)]")
    print(f"  Feature buffer:   {n_feat*4:>6} bytes")
    print(f"  Scaler (mean+std):{n_feat*8:>6} bytes")
    print(f"  Decision tree (~depth 8, 2^8 nodes): ~3.5 KB")
    print(f"  Linear SVM ({n_feat} weights):        {n_feat*4+4:>4} bytes")
    print(f"\n  → Fits in Arduino Mega (8 KB SRAM) with depth ≤ 6")
    print(f"  → Fits in ESP32 (520 KB SRAM) with depth ≤ 15")
    print(f"  → Fits in STM32F4 (192 KB SRAM) comfortably")


if __name__ == "__main__":
    main()