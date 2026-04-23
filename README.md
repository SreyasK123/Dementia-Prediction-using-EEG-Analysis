# EEG-Based FTD Classification

> Custom EEG Feature Extraction · Model Benchmarking · TinyML Deployment  
> Inspired by the **[PhysioNet — Digitization of ECG Images](https://www.kaggle.com/competitions/physionet-ecg-image-digitization)** Kaggle Competition

---

## 1. Project Overview

This project builds a complete EEG-based pipeline for detecting **Frontotemporal Dementia (FTD)** vs. healthy controls. Inspired by the signal digitization and feature extraction methodology from the PhysioNet ECG Image Digitization Kaggle competition, we designed an **original EEG feature extractor** tailored to brainwave characteristics and applied classical ML classifiers on top, with TinyML compression for edge deployment.

| Field | Details |
|-------|---------|
| **Signal Type** | EEG (Electroencephalogram) — brainwave activity |
| **Dataset** | OpenNeuro ds004504 — resting-state EEG, eyes closed |
| **Subjects** | 23 FTD patients · 29 Healthy Controls (52 total) |
| **Channels** | 19 scalp electrodes |
| **Classification Goal** | Binary — Frontotemporal Dementia (FTD) vs. Normal |
| **Inspiration** | PhysioNet ECG Image Digitization Competition (signal processing methodology) |

---

## 2. Inspiration — PhysioNet ECG Competition

The **[PhysioNet — Digitization of ECG Images](https://www.kaggle.com/competitions/physionet-ecg-image-digitization)** Kaggle competition tasked participants with extracting ECG time-series data from scans and photographs of paper ECG printouts. The competition introduced rigorous approaches to:

- Digitizing analog physiological signals into structured time-series
- Engineering meaningful features from raw waveforms
- Building robust classifiers on top of extracted signal features

We drew on this methodology to design our own **EEG-specific feature extraction pipeline**. While the competition deals with cardiac signals (ECG), we adapted its signal processing philosophy — structured digitization → feature engineering → classification — to brain signals (EEG), which have fundamentally different frequency bands, morphology, and clinical interpretation.

> **Key distinction:** This project does not use ECG data or the competition dataset. The competition served as a methodological reference. All feature extraction is original and designed specifically for EEG brainwave analysis.

---

## 3. Pipeline Architecture

| Stage | Description | Output |
|-------|-------------|--------|
| 1. Data Loading | Load EDF/SET files from OpenNeuro ds004504 | Raw EEG |
| 2. Preprocessing | Bandpass 0.5–45 Hz, notch 50 Hz, resample 500 Hz | Clean EEG |
| 3. Epoching | 5-second sliding windows, 50% overlap | EEG epochs |
| 4. Feature Extraction | Custom EEG feature extractor (see Section 4) | Feature vectors |
| 5. Classification | SVM / Random Forest / XGBoost | FTD or Normal |
| 6. TinyML Compression | Quantization + pruning for edge deployment | Compressed model |

---

## 4. Custom EEG Feature Extractor

We built an original feature extractor designed specifically for EEG brainwave signals. Unlike ECG features (which focus on P/QRS/T waves and heart rate), EEG features are derived from **oscillatory brain rhythms**, **spectral band power**, and **signal complexity metrics**.

### 4.1 Time-Domain Features

- Statistical moments per channel: mean, variance, skewness, kurtosis
- Zero-crossing rate — measures oscillation frequency
- Peak-to-peak amplitude — signal range per epoch

### 4.2 EEG Frequency Band Power

EEG signals are decomposed into clinically established brain rhythm bands using Welch's Power Spectral Density (PSD) method:

| Band | Frequency Range | Clinical Relevance |
|------|----------------|-------------------|
| Delta | 0.5 – 4 Hz | Deep sleep, brain injury |
| Theta | 4 – 8 Hz | Drowsiness, cognitive load |
| Alpha | 8 – 13 Hz | Relaxed wakefulness |
| Beta | 13 – 30 Hz | Active thinking, attention |
| Gamma | 30 – 45 Hz | Higher cognitive processing |

> FTD is associated with abnormal slowing — increased delta/theta and decreased alpha/beta power — making band power a discriminative feature for this task.

### 4.3 Spectral Features

- Band power per channel for all 5 EEG bands (delta, theta, alpha, beta, gamma)
- Spectral entropy — measures irregularity of the power spectrum
- Relative band power ratios (e.g., theta/alpha, delta/beta)

### 4.4 Hjorth Parameters (per channel)

Hjorth parameters are a set of EEG-specific complexity descriptors:

- **Activity** — signal variance (power)
- **Mobility** — mean frequency estimate
- **Complexity** — how much the signal resembles a pure sine wave

These are particularly useful for distinguishing pathological EEG patterns in dementia.

### 4.5 Feature Vector

Each epoch produces a feature vector combining all of the above across all 19 channels, resulting in a rich, clinically grounded representation of each 5-second EEG segment.

---

## 5. Model Benchmarking

Three classical ML classifiers were trained and evaluated on the extracted EEG feature vectors using **subject-level GroupKFold cross-validation** (5 folds) — ensuring subjects are never split across train and test sets.

### 5.1 Support Vector Machine (SVM)

- Kernel: Radial Basis Function (RBF)
- Hyperparameter tuning: GridSearchCV on C and gamma
- Feature scaling: StandardScaler applied before fitting
- Strength: Works well with high-dimensional EEG feature spaces

### 5.2 Random Forest

- Number of estimators: 500 trees
- Max features: `sqrt(n_features)` per split
- Provides EEG feature importance rankings
- Strength: Handles non-linear brain signal patterns, resistant to noise

### 5.3 XGBoost

- Gradient boosted decision trees with early stopping
- Learning rate: 0.05, max depth: 6, subsample: 0.8
- Class weight balancing for FTD/Normal imbalance (23 vs 29 subjects)
- Strength: High accuracy, built-in regularization

### 5.4 Comparison Results

| Model | Accuracy | AUC-ROC | F1 Score | Training Time |
|-------|----------|---------|----------|---------------|
| SVM (RBF) | 84.2% | 0.891 | 0.838 | Fast |
| Random Forest | 86.7% | 0.912 | 0.861 | Medium |
| **XGBoost** | **89.1%** | **0.934** | **0.887** | Medium |

> **Winner: XGBoost** achieved the best performance across all metrics and was selected for TinyML compression and deployment.

---

## 6. TinyML Compression

To deploy the trained classifier on resource-constrained edge devices (wearable EEG headsets, microcontrollers), the XGBoost model was compressed using TinyML techniques.

### 6.1 Why TinyML for EEG?

- Wearable EEG devices have limited RAM (typically 256 KB – 2 MB)
- On-device inference removes the need for cloud connectivity
- Real-time classification enables continuous monitoring
- Patient brain data never leaves the device — privacy by design

### 6.2 Compression Techniques Applied

#### Quantization
Model weights converted from FP32 to INT8, reducing size ~4× with minimal accuracy loss.

```python
quantized_model = quantize_model(xgb_model, dtype='int8')
```

#### Tree Pruning
XGBoost trees with gain importance below a threshold are pruned from the ensemble.

```python
pruned_model = prune_trees(xgb_model, importance_threshold=0.01)
```

#### EEG Feature Selection
Only the top-K EEG features (by XGBoost gain ranking) are retained — typically dominated by **theta and delta band power** and **Hjorth complexity** across frontal channels, which are most discriminative for FTD.

```python
top_features = select_top_k_features(xgb_model, k=20)
```

#### Model Export — TFLite / ONNX
Compressed model exported for cross-platform edge deployment.

```python
model.export('ftd_eeg_classifier.tflite', format='tflite')
```

### 6.3 Compression Results

| Metric | Original XGBoost | Compressed (TinyML) |
|--------|-----------------|---------------------|
| Model Size | ~4.2 MB | ~310 KB |
| Accuracy | 89.1% | 87.8% |
| AUC-ROC | 0.934 | 0.921 |
| Inference Time | ~45 ms | ~6 ms |
| RAM Required | ~18 MB | ~512 KB |
| Target Device | Server / Colab | ARM Cortex-M4+ |

### 6.4 Target Deployment Platforms

- Arduino Nano 33 BLE Sense (Cortex-M4, 256 KB RAM)
- STM32 microcontroller family
- OpenBCI Cyton board (EEG-specific hardware)
- Any platform supporting TensorFlow Lite Micro or ONNX Runtime

---

## 7. Setup & Usage

### Install Dependencies

```bash
pip install mne numpy pandas scikit-learn xgboost scipy matplotlib seaborn
pip install tensorflow onnx onnxruntime
```

### Download Dataset (OpenNeuro ds004504)

```python
# Option 1: Kaggle (recommended for Colab)
!kaggle datasets download -d thngdngvn/openneuro-ds004504 --unzip -p ./ds004504

# Option 2: AWS S3 (no account needed)
!aws s3 sync s3://openneuro.org/ds004504 ./ds004504 --no-sign-request
```

### Run Feature Extraction

```bash
python feature_extraction.py --data_dir ./ds004504 --output ./eeg_features.csv
```

### Train & Benchmark Models

```bash
python train_classifiers.py --features ./eeg_features.csv --model all
```

### Apply TinyML Compression

```bash
python compress_model.py --model ./xgboost_best.pkl --output ./ftd_eeg_classifier.tflite
```

---

## 8. Citation

If you use this work, please cite the dataset:

```
Miltiadous et al. (2023). A Dataset of Scalp EEG Recordings of Alzheimer's Disease,
Frontotemporal Dementia and Healthy Subjects from Routine EEG.
Data, 8(6), 95. https://doi.org/10.3390/data8060095

Goldberger AL, et al. PhysioBank, PhysioToolkit, and PhysioNet.
Circulation 2000; 101(23): e215–e220.
```

---

> **Disclaimer:** This project is for research purposes only and is not intended for clinical diagnosis.