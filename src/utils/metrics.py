"""Metric utilities without a scikit-learn dependency."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


def confusion_matrix_np(y_true: Iterable[int], y_pred: Iterable[int], num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true, pred in zip(y_true, y_pred):
        matrix[int(true), int(pred)] += 1
    return matrix


def classification_stats(matrix: np.ndarray, class_names: Sequence[str]) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    total = int(matrix.sum())
    weighted_f1_sum = 0.0
    precisions = []
    recalls = []
    f1s = []
    supports = []

    for idx, name in enumerate(class_names):
        tp = float(matrix[idx, idx])
        fp = float(matrix[:, idx].sum() - matrix[idx, idx])
        fn = float(matrix[idx, :].sum() - matrix[idx, idx])
        support = int(matrix[idx, :].sum())
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        rows.append(
            {
                "class": name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)
        weighted_f1_sum += f1 * support

    macro_precision = float(np.mean(precisions)) if precisions else 0.0
    macro_recall = float(np.mean(recalls)) if recalls else 0.0
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    weighted_f1 = weighted_f1_sum / total if total > 0 else 0.0
    accuracy = float(np.trace(matrix) / total) if total > 0 else 0.0

    rows.extend(
        [
            {
                "class": "macro_avg",
                "precision": macro_precision,
                "recall": macro_recall,
                "f1": macro_f1,
                "support": total,
            },
            {
                "class": "weighted_avg",
                "precision": "",
                "recall": "",
                "f1": weighted_f1,
                "support": total,
            },
            {
                "class": "accuracy",
                "precision": "",
                "recall": "",
                "f1": accuracy,
                "support": total,
            },
        ]
    )
    return rows


def summarize_predictions(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int) -> dict[str, float]:
    matrix = confusion_matrix_np(y_true, y_pred, num_classes)
    rows = classification_stats(matrix, [str(i) for i in range(num_classes)])
    macro_row = next(row for row in rows if row["class"] == "macro_avg")
    weighted_row = next(row for row in rows if row["class"] == "weighted_avg")
    accuracy_row = next(row for row in rows if row["class"] == "accuracy")
    return {
        "accuracy": float(accuracy_row["f1"]),
        "balanced_accuracy": float(macro_row["recall"]),
        "macro_f1": float(macro_row["f1"]),
        "weighted_f1": float(weighted_row["f1"]),
    }


def save_confusion_matrix(path: str | Path, matrix: np.ndarray, class_names: Sequence[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label"] + list(class_names))
        for name, row in zip(class_names, matrix):
            writer.writerow([name] + [int(value) for value in row])


def save_classification_report(path: str | Path, rows: Sequence[dict[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "precision", "recall", "f1", "support"])
        writer.writeheader()
        writer.writerows(rows)

