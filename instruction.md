# Tabular Binary Classification with Fairness Evaluation

Train a binary classification model on the provided tabular dataset and evaluate both predictive performance and fairness.

## Dataset & Task Overview
The dataset is located at `/app/environment/task_inputs/data.csv`. 
- The target column is named `target` (binary: 0/1).
- The protected attribute column is named `platform` (used for fairness/disparity evaluation, evaluated as `platform_group` in test predictions).
- All other columns are input features.

## Requirements & Pipeline

1. Load and preprocess the dataset, handling any missing values and scaling features appropriately.
2. Split the data into train, validation, and test partitions using a fixed random seed.
3. Train a binary classification model.
4. Evaluate performance on the test set.

The final solution must achieve:

- auroc >= 0.80
- max_platform_fpr_disparity <= 0.25

where max_platform_fpr_disparity is the maximum absolute difference in False Positive Rate between platform groups.

## Deliverables
Save the following deliverables using absolute paths:
- `/app/artifacts/metrics.json`: JSON object containing exact keys `"auroc"` and `"max_platform_fpr_disparity"`.
- `/app/artifacts/predictions.csv`: CSV file containing test set predictions and true labels with columns `probability`, `predicted_label`, and `platform_group`.
- `/app/reports/report.md`: Summary report detailing methodology, results, and fairness analysis (including sections on methodology, results, and fairness).