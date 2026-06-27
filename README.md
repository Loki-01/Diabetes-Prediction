# Diabetes Prediction with Multi-Layer Perceptron (MLP)

A deep learning classification system built to predict diabetes onset from medical indicators using a Multi-Layer Perceptron neural network in TensorFlow/Keras.

---

## Dataset

**Source:** [Kaggle — Diabetes Detection Dataset](https://www.kaggle.com/datasets/mathchi/diabetes-data-set)  
**Rows:** 768 patients  
**Features:** 8 medical indicators + 1 target  
**Target:** `Outcome` — 0 (Non-Diabetic) / 1 (Diabetic)

| Feature | Description |
|---|---|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skin fold thickness (mm) |
| Insulin | 2-Hour serum insulin (mu U/ml) |
| BMI | Body mass index |
| DiabetesPedigreeFunction | Diabetes hereditary risk score |
| Age | Age in years |
| **Outcome** | **Target: 0 or 1** |

---

## Project Structure

```
├── TP2_Diabetes_MLP.ipynb   # Full notebook (11 parts, 16 exercises)
├── diabetes.csv             # Raw dataset
├── README.md
└── assets/
    ├── univariate_analysis.png
    ├── bivariate_heatmap.png
    ├── pairplot.png
    ├── basic_mlp_curves.png
    ├── deep_mlp_curves.png
    ├── reg_mlp_curves.png
    ├── confusion_matrix_basic.png
    └── model_comparison.png
```

---

## Workflow

```
Raw Data → EDA → Zero Imputation → Feature Engineering → Scaling → MLP Training → Regularization → Comparison
```

### 1. Exploratory Data Analysis
- Univariate: histograms + boxplots for all 8 features
- Bivariate: correlation heatmap, boxplots by Outcome, pairplot on top 4 features
- Finding: **Glucose** is the strongest predictor (r ≈ 0.47)

### 2. Preprocessing
- Replaced medically impossible zeros in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI` with **median imputation**
- Stratified 80/20 train-test split
- StandardScaler applied (fit on train, transform on test)

### 3. Feature Engineering
| Feature | Description |
|---|---|
| `BMI_Category` | WHO obesity classification (0–3) |
| `Age_Category` | Age group bins (0–3) |
| `Glucose_BMI` | Interaction term: Glucose × BMI |
| `Insulin_Glucose_Ratio` | Metabolic ratio |

### 4. Models Built

| Model | Architecture | Regularization |
|---|---|---|
| MLP Basic | 64 → 32 → 1 | None |
| MLP Deep | 128 → 64 → 32 → 16 → 1 | None |
| MLP Regularized | 128 → 64 → 32 → 16 → 1 | Dropout (0.3/0.2) + EarlyStopping |
| MLP Optimized | 128 → 64 → 32 → 1 | Dropout (0.3) + EarlyStopping + lr tuning |

All models use:
- Hidden layers: **ReLU** activation
- Output layer: **Sigmoid** activation
- Optimizer: **Adam**
- Loss: **Binary Crossentropy**

---

## Results

| Model | Acc Train | Acc Test | Precision | Recall | F1-Score |
|---|---|---|---|---|---|
| MLP Basic | ~0.83 | ~0.77 | ~0.74 | ~0.68 | ~0.71 |
| MLP Deep | ~0.89 | ~0.75 | ~0.72 | ~0.65 | ~0.68 |
| MLP Regularized | ~0.82 | ~0.78 | ~0.75 | ~0.70 | ~0.72 |
| **MLP Optimized** | **~0.83** | **~0.79** | **~0.76** | **~0.72** | **~0.74** |

> **Recall** is prioritized as the key metric — a false negative (missed diabetic patient) carries higher clinical risk than a false positive.

---

## Key Findings

- **MLP Deep** overfits — train accuracy climbs while validation loss diverges
- **Dropout + EarlyStopping** closes the train/val gap and improves generalization
- **MLP Optimized** achieves the best balance between precision and recall
- **MLP Regularized** is the recommended model for real-world deployment

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Keras](https://img.shields.io/badge/Keras-Sequential_API-red)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-preprocessing-green)
![Pandas](https://img.shields.io/badge/Pandas-EDA-lightgrey)
![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-9cf)

---

## How to Run

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn

jupyter notebook TP2_Diabetes_MLP.ipynb
```

Place `diabetes.csv` in the same directory before running.

---

## Author

**NAIT ALI Mohamed Abderrahim**  
Energy & Mechanical Engineer | Data Scientist  
[LinkedIn](https://linkedin.com/in/nait-ali-mohamed-abderrahim-663a252a5) · [GitHub](https://github.com/Loki-01)
