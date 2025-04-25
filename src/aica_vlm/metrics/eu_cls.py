# src/metrics/eu_classification.py

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    top_k_accuracy_score,
)

from .base import Metric


class EmotionClassificationMetrics(Metric):
    def compute(self, predictions, references, topk_preds=None, labels=None):
        result = {
            "Accuracy": accuracy_score(references, predictions),
            "Macro F1": f1_score(references, predictions, average="macro"),
            "Confusion Matrix": confusion_matrix(references, predictions).tolist(),
        }
        if topk_preds is not None and labels is not None:
            result["Top-k Accuracy"] = top_k_accuracy_score(
                references, topk_preds, labels=labels
            )
        return result
