from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from .base import Metric


class EmotionClassificationMetrics(Metric):
    def compute(self, predictions, references):
        normalized_predictions = [pred.strip().lower() for pred in predictions]
        normalized_references = [ref.strip().lower() for ref in references]

        binary_predictions = [
            1 if pred in ref else 0
            for pred, ref in zip(normalized_predictions, normalized_references)
        ]
        binary_references = [1] * len(references)

        result = {
            "Accuracy": sum(binary_predictions) / len(binary_predictions),
            "Macro F1": f1_score(
                binary_references, binary_predictions, average="macro"
            ),
            "Weighted F1": f1_score(
                binary_references, binary_predictions, average="weighted"
            ),
            "Confusion Matrix": confusion_matrix(
                binary_references, binary_predictions
            ).tolist(),
        }

        return result
