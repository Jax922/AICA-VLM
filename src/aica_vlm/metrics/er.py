from bert_score import score as bert_score
from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer

class EmotionReasoningMetrics:
    # Please download the 'bert-base-uncased' model first
    def compute(self, predictions, references, model_path="./models/bert-base-uncased"):
        results = {}

        # BLEU
        ref_list = [[ref.split()] for ref in references]
        pred_list = [pred.split() for pred in predictions]
        results["BLEU"] = corpus_bleu(ref_list, pred_list)

        # ROUGE
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_totals = {"ROUGE1": 0, "ROUGE2": 0, "ROUGEL": 0}
        for pred, ref in zip(predictions, references):
            scores = scorer.score(ref, pred)
            rouge_totals["ROUGE1"] += scores["rouge1"].fmeasure
            rouge_totals["ROUGE2"] += scores["rouge2"].fmeasure
            rouge_totals["ROUGEL"] += scores["rougeL"].fmeasure
        for key in rouge_totals:
            results[key] = rouge_totals[key] / len(predictions)

        # BERTScore
        try:
            P, R, F1 = bert_score(
                predictions,
                references,
                lang="en",
                model_type=model_path,
                num_layers=12,
                all_layers=False,
                verbose=False
            )
            results["BERTScore"] = F1.mean().item()
        except Exception as e:
            print(f"BERTScore computing failed, error: {e}")
            results["BERTScore"] = None

        return results
