# LLM Request Performance

A small end-to-end Machine Learning project that simulates LLM request telemetry and explores two related problems:

1. **Regression** — predict request latency.
2. **Classification** — detect requests that are likely to violate a latency SLA.

The goal of the project is not only to train models, but to understand the full ML workflow: data generation, preprocessing, model evaluation, cross-validation, regularization, hyperparameter tuning, probability thresholds, and business-driven model selection.

---

## Problem

LLM-powered systems can experience latency changes depending on factors such as model size, input token count, output token count, concurrent requests, and deployment region.

This project models those effects using a synthetic dataset and answers two questions.

### Regression

> How much latency should we expect for a request?

Target:

```text
latency_ms
```

### Classification

> Is this request likely to exceed the SLA threshold?

The classification target is:

```text
slow_request = latency_ms > 3000
```

where:

- `0` = normal request
- `1` = slow request / SLA violation

---

## Dataset

The dataset contains **500 synthetic LLM requests**.

| Feature | Type | Description |
|---|---|---|
| `model` | Categorical | LLM size: `small`, `medium`, `large` |
| `input_tokens` | Numerical | Number of input tokens |
| `output_tokens` | Numerical | Number of generated output tokens |
| `concurrent_requests` | Numerical | Number of concurrent requests |
| `region` | Categorical | Deployment region |
| `latency_ms` | Numerical | Observed request latency in milliseconds |

The dataset is generated reproducibly using a fixed random seed.

---

## Project Structure

```text
llm-request-performance/
├── data/
│   └── llm_requests.csv
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_classification.ipynb
├── src/
│   └── generate_data.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Preprocessing

Numerical features:

```text
input_tokens
output_tokens
concurrent_requests
```

are transformed using `StandardScaler`.

Categorical features:

```text
model
region
```

are transformed using `OneHotEncoder`.

The preprocessing logic is implemented using a `ColumnTransformer`.

For classification, preprocessing and model training are combined in a Scikit-learn `Pipeline` so preprocessing is fitted only on training data during cross-validation. This avoids data leakage.

---

# Regression

## Baseline Model

The regression task uses `LinearRegression`.

### Final test results

| Metric | Result |
|---|---:|
| MAE | 98.94 ms |
| RMSE | 114.20 ms |
| R² | 0.9608 |

The model explains approximately **96% of the observed latency variance** in the test set.

Residual analysis showed no obvious systematic pattern, suggesting that the linear model captures most of the underlying signal in the synthetic dataset.

## Cross-Validation

5-fold cross-validation produced:

```text
R² scores:
0.9593
0.9692
0.9660
0.9722
0.9651
```

Summary:

```text
Mean R² = 0.9664
Std R²  = 0.0043
```

This indicates strong and stable regression performance across folds.

## Regularization

Two regularized linear models were evaluated:

- Ridge Regression
- Lasso Regression

Neither significantly improved performance over the baseline Linear Regression model.

This makes sense because the dataset is intentionally linear, the number of features is small, all features contain useful signal, and multicollinearity is limited.

Large regularization strengths reduced performance and produced underfitting.

---

# Classification

## Target Definition

The binary target is created using:

```python
slow_request = latency_ms > 3000
```

Class distribution:

```text
Normal requests: 355 (71%)
Slow requests:   145 (29%)
```

Because the classes are moderately imbalanced, Accuracy alone is not sufficient for model evaluation.

## Baseline Classifier

The initial classifier uses `LogisticRegression`.

The pipeline returns both binary predictions and probabilities for `slow_request = 1`.

The default classification threshold is:

```text
0.5
```

## Evaluation Metrics

The classification workflow evaluates Accuracy, Precision, Recall, F1 Score, ROC Curve, ROC-AUC, and the Confusion Matrix.

For this use case, **Recall is especially important**.

A False Negative means:

> A real SLA violation was classified as a normal request.

Because missing a real SLA violation is considered expensive, the system prioritizes reducing False Negatives.

## ROC and ROC-AUC

The ROC curve evaluates the classifier across multiple classification thresholds.

It compares:

```text
TPR = True Positive Rate = Recall
```

against:

```text
FPR = False Positive Rate
```

The classifier achieved:

```text
ROC-AUC = 0.9990
```

This indicates excellent discrimination between normal and slow requests.

ROC-AUC is not the same as Accuracy. Accuracy depends on a particular classification threshold, while ROC-AUC measures the classifier's ability to rank positive examples above negative examples across many thresholds.

## Classification Cross-Validation

5-fold cross-validation with the default Logistic Regression configuration produced:

| Metric | Mean |
|---|---:|
| Accuracy | 0.9225 |
| Precision | 0.9028 |
| Recall | 0.8366 |
| F1 | 0.8617 |
| ROC-AUC | 0.9875 |

Recall standard deviation:

```text
0.0833
```

The classifier showed excellent discrimination across folds, but Recall at the default threshold was lower and more variable.

This suggested that both model regularization and the classification threshold could be improved.

## Hyperparameter Tuning

`GridSearchCV` was used to tune the Logistic Regression `C` parameter.

Values tested:

```text
0.01
0.1
1
10
100
```

Recall was used as the optimization metric.

| C | Mean CV Recall |
|---:|---:|
| 0.01 | 0.0866 |
| 0.1 | 0.6989 |
| 1 | 0.8366 |
| 10 | 0.8793 |
| 100 | 0.8967 |

Best configuration:

```text
C = 100
```

with:

```text
Best CV Recall = 0.8967
```

Increasing `C` weakens regularization and gave the classifier more flexibility to detect the positive class.

## Threshold Selection

Hyperparameter tuning and threshold tuning were treated as separate decisions.

After selecting the best model, out-of-fold probabilities were generated using cross-validation.

The business objective was:

```text
Recall >= 95%
```

Among thresholds satisfying that constraint, the threshold with the highest Precision was selected.

Selected threshold:

```text
0.2358
```

Validation performance at this threshold:

```text
Precision = 0.8473
Recall    = 0.9569
```

A lower threshold is appropriate here because the cost of missing a real SLA violation is higher than the cost of generating a small number of false alerts.

---

# Final Classification Model

Final configuration:

```text
Preprocessing:
  StandardScaler
  OneHotEncoder

Model:
  LogisticRegression(C=100)

Classification threshold:
  0.2358
```

### Final test results

| Metric | Result |
|---|---:|
| Accuracy | 0.9800 |
| Precision | 0.9355 |
| Recall | 1.0000 |
| F1 Score | 0.9667 |
| ROC-AUC | 0.9990 |

Confusion Matrix:

```text
                 Predicted
                Normal   Slow
Actual Normal      69      2
Actual Slow         0     29
```

Result:

```text
True Negatives:  69
False Positives:  2
False Negatives:  0
True Positives:  29
```

The final system detected **all SLA violations in the test set**, producing zero False Negatives.

That result matches the business objective of prioritizing detection of slow requests while accepting a small number of false alarms.

---

# ML Workflow

```text
Synthetic Data Generation
        ↓
Exploratory Data Analysis
        ↓
Train / Test Split
        ↓
Preprocessing
        ↓
Regression
        ├── Linear Regression
        ├── MAE / RMSE / R²
        ├── Residual Analysis
        ├── Cross-Validation
        └── Ridge / Lasso
        ↓
Classification
        ├── Logistic Regression
        ├── Confusion Matrix
        ├── Precision / Recall / F1
        ├── Probability Thresholds
        ├── ROC / ROC-AUC
        ├── Cross-Validation
        ├── GridSearchCV
        ├── Out-of-Fold Predictions
        └── Threshold Selection
        ↓
Final Evaluation
```

---

# Key Lessons

- `fit()` learns parameters from training data.
- `transform()` applies already learned preprocessing rules.
- Test data should never be used to fit preprocessing.
- Pipelines help prevent data leakage.
- Cross-validation evaluates model stability.
- Regularization controls model complexity.
- Hyperparameters and classification thresholds solve different problems.
- Accuracy alone can be misleading for imbalanced classification.
- Precision and Recall represent different business costs.
- ROC-AUC evaluates discrimination across many thresholds.
- The default threshold of `0.5` is not automatically the best business threshold.
- Threshold selection should be based on validation data and business requirements.
- Final test data should be reserved for final evaluation.

---

# Installation

Clone the repository:

```bash
git git@github.com:luisEnriqueSM/llm-request-performance.git
cd llm-request-performance
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it.

macOS / Linux:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\\Scripts\\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Generate the Dataset

Run:

```bash
python src/generate_data.py
```

This creates:

```text
data/llm_requests.csv
```

---

# Run the Notebooks

Start Jupyter Lab:

```bash
jupyter lab
```

Then run the notebooks in order:

```text
1. notebooks/01_data_exploration.ipynb
2. notebooks/02_classification.ipynb
```

---

# Tech Stack

- Python
- pandas
- NumPy
- Scikit-learn
- Matplotlib
- JupyterLab

---

# Why This Project Exists

This project is intentionally small.

Its purpose is to build a strong understanding of the fundamentals behind production ML workflows before moving into larger AI systems.

Instead of treating Machine Learning as:

```text
model.fit()
model.predict()
```

the project focuses on understanding:

```text
data
→ preprocessing
→ model
→ probabilities
→ evaluation
→ validation
→ tuning
→ business decision
```

That distinction becomes increasingly important when ML models are integrated into real AI systems.
