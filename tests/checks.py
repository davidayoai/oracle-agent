"""Task-specific verification checks for Otter Tabular Binary Classification.

This is the ONLY scoring file you edit. Replace the placeholder checks
below with your task's actual logic. rubric_score.py loads this file and
calls evaluate(context) exactly once.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

APP_DIR = Path(os.environ.get("OTTER_APP_DIR", "/app"))


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise AssertionError(f"Missing file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def metrics_path() -> Path:
    return APP_DIR / "artifacts" / "metrics.json"


def predictions_path() -> Path:
    return APP_DIR / "artifacts" / "predictions.csv"


def report_path() -> Path:
    return APP_DIR / "reports" / "report.md"


def input_data_path() -> Path:
    return APP_DIR / "environment" / "task_inputs" / "data.csv"


# --- Independent Metric Calculation Logic ---

def compute_platform_fpr_disparity(truth_df: pd.DataFrame, preds_df: pd.DataFrame) -> float:
    merged = truth_df.copy()
    merged["predicted_label"] = (
        preds_df["predicted_label"].values 
        if "predicted_label" in preds_df.columns 
        else (preds_df["probability"] >= 0.5).astype(int)
    )
    
    fprs = []
    for _, group in merged.groupby("platform_group"):
        y_true = group["target"]
        y_pred = group["predicted_label"]
        
        if len(y_true.unique()) < 2 and 0 not in y_true.values:
            fprs.append(0.0)
            continue
            
        tn, fp, _, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        negatives = tn + fp
        fpr = fp / negatives if negatives > 0 else 0.0
        fprs.append(fpr)
        
    if not fprs:
        return 0.0
    return float(max(fprs) - min(fprs))


def calculate_independent_metrics() -> dict[str, float]:
    pred_file = predictions_path()
    data_file = input_data_path()
    
    if not pred_file.exists() or not data_file.exists():
        return {"auroc": 0.0, "max_platform_fpr_disparity": 1.0}
        
    preds_df = pd.read_csv(pred_file)
    full_data = pd.read_csv(data_file)
    
    _, test_df = train_test_split(
        full_data, 
        test_size=0.3, 
        random_state=42, 
        stratify=full_data["target"] if "target" in full_data.columns else None
    )

    target_col = "target" if "target" in test_df.columns else full_data.columns[-1]
    platform_col = (
        "platform_group" if "platform_group" in test_df.columns 
        else ("platform" if "platform" in test_df.columns else full_data.columns[0])
    )
    
    y_true = test_df[target_col].values[:len(preds_df)]
    prob_col = "probability" if "probability" in preds_df.columns else preds_df.columns[0]
    y_prob = preds_df[prob_col].values
    
    try:
        calc_auroc = float(roc_auc_score(y_true, y_prob))
    except (ValueError, TypeError):
        calc_auroc = 0.5
        
    eval_truth = pd.DataFrame({
        "target": y_true,
        "platform_group": test_df[platform_col].values[:len(preds_df)]
    })
    calc_disparity = compute_platform_fpr_disparity(eval_truth, preds_df)
    
    return {
        "auroc": calc_auroc,
        "max_platform_fpr_disparity": calc_disparity
    }


# --- Check Functions ---

def check_data_deliverables() -> str:
    assert predictions_path().exists(), "predictions.csv missing"
    return "predictions file exists"


def check_data_schema() -> str:
    preds = predictions_path()
    assert preds.stat().st_size > 0, "predictions file is empty"
    df = pd.read_csv(preds)
    assert len(df) > 0, "predictions dataframe has no rows"
    for col in ["probability", "predicted_label", "platform_group"]:
        assert col in df.columns, f"Missing required prediction column: {col}"
    return "predictions file valid and contains required schema"


def check_model_deliverables() -> str:
    assert metrics_path().exists(), "metrics.json missing"
    return "metrics file exists"


def check_metrics_schema() -> str:
    metrics = load_json(metrics_path())
    required = {"auroc", "max_platform_fpr_disparity"}
    missing = required - set(metrics.keys())
    assert not missing, f"Missing keys: {missing}"
    assert 0 <= metrics["auroc"] <= 1
    assert 0 <= metrics["max_platform_fpr_disparity"] <= 1
    return "metrics schema valid"


def check_metrics_thresholds() -> str:
    computed = calculate_independent_metrics()
    assert computed["auroc"] >= 0.80, f"Computed AUROC {computed['auroc']} below threshold 0.80"
    assert computed["max_platform_fpr_disparity"] <= 0.25, f"Computed disparity {computed['max_platform_fpr_disparity']} exceeds threshold 0.25"
    return "metrics meet strict evaluation thresholds"


def check_final_deliverables() -> str:
    assert report_path().exists(), "report missing"
    return "report exists"


def check_report_content() -> str:
    report = report_path().read_text(encoding="utf-8").lower()
    assert len(report.split()) >= 50, "report is too short (less than 50 words)"
    for section in ["methodology", "result", "fairness"]:
        assert section in report, f"Report missing required section/topic: {section}"
    return "report has sufficient content and covers required analysis topics"


def _criterion(
    criterion_id: str,
    fn,
    milestone_id: str,
) -> dict[str, Any]:
    try:
        return {
            "id": criterion_id,
            "passed": True,
            "detail": fn(),
            "milestone_id": milestone_id,
        }
    except (AssertionError, FileNotFoundError, ValueError, TypeError) as exc:
        return {
            "id": criterion_id,
            "passed": False,
            "detail": str(exc),
            "milestone_id": milestone_id,
        }


def evaluate(context: dict[str, Any]) -> dict[str, Any]:
    try:
        computed_metrics = calculate_independent_metrics()
    except (ValueError, TypeError, RuntimeError):
        computed_metrics = {"auroc": 0.0, "max_platform_fpr_disparity": 1.0}

    criteria = [
        _criterion("data_deliverables", check_data_deliverables, "data-preparation"),
        _criterion("data_schema", check_data_schema, "data-preparation"),
        _criterion("model_deliverables", check_model_deliverables, "model-evaluation"),
        _criterion("metrics_schema", check_metrics_schema, "model-evaluation"),
        _criterion("metrics_thresholds", check_metrics_thresholds, "model-evaluation"),
        _criterion("final_deliverables", check_final_deliverables, "deliverables-and-report"),
        _criterion("report_content", check_report_content, "deliverables-and-report"),
    ]

    return {
        "criteria": criteria,
        "metrics": computed_metrics,
    }