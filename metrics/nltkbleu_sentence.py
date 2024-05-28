from torchmetrics.text import BLEUScore
import datasets
import evaluate
import torch
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
weights_list = [(1, 0, 0, 0),(0, 1, 0, 0),(0, 0, 1, 0),(0, 0, 0, 1)]

def compute_metrics(predictions, references):
    meg_hyp = [dt.split() for dt in predictions]
    meg_ref = [[ref.split()] for ref in references]
    
    result={}
    for i, weight in enumerate(weights_list):
        corpus_bleu_score = corpus_bleu(meg_ref, meg_hyp, weights = weight)
        result[f'nltkbleu_sentence-{i+1}']=corpus_bleu_score
    return result

class FullEval(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description='None',
            citation='None',
            inputs_description='None',
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string", id="sequence"),
                    "references": datasets.Value("string", id="sequence"),
                }
            ),
            codebase_urls=[""],
            reference_urls=[
                "",
            ],
        )
    def _compute(self, predictions, references):

        return compute_metrics(predictions, references)