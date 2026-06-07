# Lung and Colon Cancer Histopathology — Machine Learning Pipeline

**Author:** Odyek Henry  
**Student ID:** 2025/HD07/26018U  
**Institution:** Makerere University, Kampala, Uganda  
**Course:** MSB7216 — Machine Learning in Biomedicine 

---

## Research Gap

Existing deep learning approaches to histopathology classification reach high accuracy but offer little direct explanation of their decisions. Pathologists working in low-resource settings also face GPU and infrastructure limits that make large neural networks impractical to run routinely. This project tests how far handcrafted features and classical machine learning can go on the same task, using the same dataset and the same train/test split as the companion deep learning project, so the two pipelines can be compared fairly.

---

## Objectives

1. Extract a rich set of colour, texture, and edge features from H and E histopathology images after Macenko stain normalisation.
2. Train and evaluate four classical classifiers: logistic regression, k nearest neighbours, random forest, and XGBoost.
3. Identify which feature families drive classification decisions using SHAP, permutation importance, and native feature importance.
4. Compare the best classical model against the ResNet-50 result from the companion deep learning pipeline on identical data splits.

---

## Dataset

**LC25000 Lung and Colon Histopathological Image Dataset**  
Borkowski, A. A., Bui, M. M., Thomas, L. B., Wilson, C. P., DeLand, L. A., & Mastorides, S. M. (2019). *Lung and colon cancer histopathological image dataset (LC25000)*. arXiv:1912.12142.  
Source: https://github.com/tampapath/lung_colon_image_set

| Class | Type | Images |
|---|---|---|
| colon_aca | Colon adenocarcinoma | 5,000 |
| colon_n | Benign colon tissue | 5,000 |
| lung_aca | Lung adenocarcinoma | 5,000 |
| lung_n | Benign lung tissue | 5,000 |
| lung_scc | Lung squamous cell carcinoma | 5,000 |

All images are 768×768 JPEG, resized to 224×224 before feature extraction.  
**Download:** [Kaggle mirror](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images) or the source GitHub above.  
Place the extracted folder at `data/lung_colon_image_set/` on your Google Drive project folder.

---

## Results

All metrics reported on the held-out test set (3,750 images, stratified 15% split, seed = 42).

| Model | Accuracy | AUC-ROC | F1 (macro) | Sensitivity | Specificity | Train–Test Gap |
|---|---|---|---|---|---|---|
| **XGBoost** | **0.9987** | **1.0000** | **0.9987** | **0.9987** | **0.9997** | 0.0013 |
| Random Forest | 0.9941 | 0.9999 | 0.9941 | 0.9941 | 0.9985 | 0.0059 |
| Support Vector Machine | 0.9776 | 0.9958 | 0.9776 | 0.9776 | 0.9944 | 0.0212 |
| Logistic Regression | 0.9709 | 0.9984 | 0.9709 | 0.9709 | 0.9927 | 0.0094 |
| k Nearest Neighbours | 0.9030 | 0.9820 | 0.9027 | 0.9030 | 0.9757 | — |

**XGBoost matches ResNet-50 (deep learning) exactly at 99.87% test accuracy, training on CPU in under one minute after feature extraction.**

The dominant classification error across all models is confusion between `lung_aca` and `lung_scc`, reflecting genuine morphological similarity between the two lung cancer subtypes at the patch level.

### Feature Importance (XGBoost + SHAP)

| Feature Family | Dimensions | SHAP Importance | Permutation Importance |
|---|---|---|---|
| Colour (RGB/HSV/LAB moments + HSV histograms) | 75 | 78% | 82% |
| HOG (edge orientation, 56×56px cells) | 324 | 13% | 1% |
| GLCM texture (Haralick properties) | 12 | 6% | 10% |
| LBP micro-texture | 10 | 3% | 7% |

Top individual feature by both methods: `color_RGB_c0_skew` (red-channel asymmetry, capturing haematoxylin density variation).

### Companion Deep Learning Pipeline (prior work)

| Model | Accuracy | AUC-ROC | F1 (macro) | Hardware |
|---|---|---|---|---|
| ResNet-50 | 0.9987 | 0.9999 | 0.9987 | GPU |
| **XGBoost (this work)** | **0.9987** | **1.0000** | **0.9987** | **CPU** |

Companion repository: https://github.com/hodyek/lung-colon-cancer-histopathology

---

## Repository Structure

```
lung-colon-cancer-histopathology-ml/
├── notebooks/
│   ├── 01_data_understanding_eda.ipynb       # EDA, class balance, texture/colour plots
│   ├── 02_preprocessing_features.ipynb       # split, Macenko normalisation, feature extraction
│   ├── 03_baseline_models.ipynb              # logistic regression + k-NN
│   ├── 04_random_forest.ipynb                # random forest with learning curves
│   ├── 05_svm.ipynb                          # SVM RBF with validation curve
│   ├── 06_xgboost.ipynb                      # XGBoost with early stopping
│   ├── 07_explainability_comparison.ipynb    # SHAP, permutation importance, ML vs DL
│   └── 03_07_all_models.ipynb                # combined notebook: runs parts 3–7 in one session
├── src/
│   ├── dataset.py      # image listing, split, Macenko normalisation
│   ├── features.py     # colour, GLCM, LBP, HOG extraction (421 features)
│   ├── train.py        # fit helpers, GridSearchCV wrapper, model saving
│   ├── evaluate.py     # metrics, confusion matrix, ROC, learning/validation curves
│   └── explain.py      # family importance, permutation importance, SHAP helpers
├── figures/            # all saved plots (generated on Colab)
├── reports/            # final report and Overleaf paper
├── data/               # .gitkeep only — download dataset separately
├── models/             # .gitkeep only — models saved to Google Drive
├── requirements.txt
└── README.md
```

---

## Setup Instructions

**Step 1 — Clone the repository:**
```bash
git clone https://github.com/hodyek/lung-colon-cancer-histopathology-ml.git
```

**Step 2 — Download the dataset.**  
Download LC25000 from [Kaggle](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images) or the [source GitHub](https://github.com/tampapath/lung_colon_image_set). Extract and place the `lung_colon_image_set` folder on your Google Drive. The notebooks expect it at:
```
/content/drive/MyDrive/lung-colon-cancer-histopathology/data/lung_colon_image_set/
```
> **Note:** The dataset is stored under the companion deep learning project folder (`lung-colon-cancer-histopathology`) and shared by both pipelines. If your Drive path differs, update `DATA_DIR` in the setup cell of each notebook.

**Step 3 — Open notebooks in Google Colab.**  
Each notebook mounts Drive, clones this repository for `src/` imports, and installs its own dependencies automatically. Run in order: `01 → 02 → 03_07` (or `03 → 04 → 05 → 06 → 07` individually). Notebook 02 is the slowest — it extracts features from all 25,000 images and caches them to Drive. Subsequent notebooks load from cache and run in seconds.

**Step 4 — (Optional) Local install:**
```bash
pip install -r requirements.txt
```

---

## Notebook Run Order

| Notebook | Purpose | Key output |
|---|---|---|
| 01 | Data understanding and EDA | Class distribution, colour/texture plots, Macenko before/after |
| 02 | Feature extraction (run once, ~90 min) | 421-feature arrays cached to Drive |
| 03 | Logistic regression and k-NN | Baseline metrics, confusion matrix, ROC curves |
| 04 | Random forest | Tuned model, learning curve, validation curve |
| 05 | Support vector machine | RBF kernel model, validation curve over C |
| 06 | XGBoost | Early stopping model, validation curve |
| 07 | Explainability and comparison | SHAP, feature importance, ML vs DL table |
| **03_07** | **Combined: all models in one session** | **Runs parts 3–7 sequentially, recommended for presentations** |

> Run `01` then `02` first. After features are cached, you can use either the individual notebooks (03–07) or the combined notebook (`03_07_all_models.ipynb`) for the models and explainability.

---

## Reproducibility Note

The train/validation/test split uses `random_state=42` and stratifies by class. The manifest is saved in Notebook 02 as `features/split_manifest.csv` on Google Drive. All subsequent notebooks load labels and features from this manifest so every run uses exactly the same split.

---

## Ethical Considerations

The LC25000 dataset contains de-identified histopathology patches with no patient identifiers. It is released for research and educational use. All work in this project is academic and does not involve clinical decision making. Results should not be interpreted as clinical diagnostic tools.

---

## License

MIT License. See `LICENSE` for details.
