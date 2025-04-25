# src/metrics/er_generation.py

import evaluate
from bert_score import score as bert_score

from .base import Metric

bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")


from bert_score import score as bert_score


class EmotionReasoningMetrics(Metric):
    def compute(self, predictions, references):
        results = {}

        bleu_result = bleu.compute(
            predictions=predictions, references=[[r] for r in references]
        )
        results["BLEU"] = bleu_result["bleu"]

        rouge_result = rouge.compute(predictions=predictions, references=references)
        results.update({k.upper(): float(v) for k, v in rouge_result.items()})

        P, R, F1 = bert_score(predictions, references, lang="en")
        results["BERTScore"] = F1.mean().item()

        return results
