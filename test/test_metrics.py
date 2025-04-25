import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/aica_vlm")))


from metrics import (
    EmotionClassificationMetrics,
    EmotionRegressionMetrics,
    EmotionReasoningMetrics
)

def test_all_metrics():
    cls = EmotionClassificationMetrics().compute(
        predictions=["happy", "sad", "angry"],
        references=["happy", "sad", "happy"]
    )
    print("Emotion Classification:", cls)

    reg = EmotionRegressionMetrics().compute(
        y_pred=[0.8, 0.1, 0.4],
        y_true=[1.0, 0.2, 0.5]
    )
    print("Emotion Regression:", reg)

    er = EmotionReasoningMetrics().compute(
        predictions=["The person looks happy.", "She is angry."],
        references=["The person is happy.", "She looks angry."]
    )
    print("ER Reasoning:", er)

if __name__ == '__main__':
    test_all_metrics()
