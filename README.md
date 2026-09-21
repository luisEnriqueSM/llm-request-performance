# ML Regression & Classification Lab

Small end-to-end Machine Learning lab focused on understanding the complete
workflow for regression and classification using an AI Engineering use case.

## Use case

Analyze simulated LLM request telemetry.

## Dataset

The dataset represents simulated LLM inference requests.

| Feature | Type | Description |
|---|---|---|
| model | categorical | LLM model size |
| input_tokens | numerical | Number of input tokens |
| output_tokens | numerical | Number of generated tokens |
| concurrent_requests | numerical | Number of simultaneous requests |
| region | categorical | Deployment region |
| latency_ms | numerical | Request latency in milliseconds |

### Regression

Predict request latency.

Target:

- `latency_ms`

### Classification

Predict whether an LLM request will exceed an SLA threshold.

Target:

- `slow_request`

## Learning goals

- Data exploration
- Train/test split
- Missing values
- Categorical features
- Feature scaling
- Linear Regression
- Ridge and Lasso
- Regression metrics
- Cross-validation
- Logistic Regression
- Classification metrics
- ROC / ROC-AUC
- Pipelines
- Hyperparameter tuning