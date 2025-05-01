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


def compute_cls_metrics_manually(json_path):
    """Extract predictions and true labels from a output JSON file."""

    import json

    emotion_list = [
        "Affection",
        "Amusement",
        "Anger",
        "Annoyance",
        "Anticipation",
        "Apprehension",
        "Awe",
        "Boredom",
        "Confidence",
        "Contentment",
        "Contempt",
        "Disapproval",
        "Disconnection",
        "Disgust",
        "Disquietment",
        "Distraction",
        "Doubt/Confusion",
        "Ecstasy",
        "Embarrassment",
        "Engagement",
        "Esteem",
        "Excitement",
        "Fear",
        "Fatigue",
        "Grief",
        "Happiness",
        "Interest",
        "Joy",
        "Loathing",
        "Pain",
        "Peace",
        "Pensiveness",
        "Pleasure",
        "Rage",
        "Sadness",
        "Sensitivity",
        "Serenity",
        "Suffering",
        "Surprise",
        "Sympathy",
        "Terror",
        "Trust",
        "Vigilance",
        "Yearning",
        "Acceptance",
        "Admiration",
        "Amazement",
    ]

    emotion_set_lower = {e.lower(): e for e in emotion_list}
    pred_res_list = []
    true_res_list = []

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data["results"]:
        output_text = item.get("output_result", "")
        true_label = item.get("true_answer", "").strip()
        output_text_lower = output_text.lower()

        found_emotions = []
        for emo_lower, emo_original in emotion_set_lower.items():
            if emo_lower in output_text_lower:
                found_emotions.append(
                    (output_text_lower.rfind(emo_lower), emo_original)
                )

        if found_emotions:
            found_emotions.sort(key=lambda x: x[0])
            selected_emotion = found_emotions[-1][1]
        else:
            selected_emotion = "UNKNOWN"

        pred_res_list.append(selected_emotion)
        true_res_list.append(true_label)

    result = EmotionClassificationMetrics().compute(
        predictions=pred_res_list, references=true_res_list
    )

    # update the json file with the new result
    data["metrics"] = result
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Updated metrics in {json_path}")
