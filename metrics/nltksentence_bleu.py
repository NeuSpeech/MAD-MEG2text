from torchmetrics.text import BLEUScore
import datasets
import evaluate
import torch
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

def compute_metrics(predictions, references):
    weights=[0.25, 0.25, 0.25, 0.25]
    bleu_scores=[]
    for i in range(len(predictions)):
        refs=[predictions[j] for j in range(len(predictions)) if j!=i]
        bleu_scores.append(sentence_bleu(refs, predictions[i], weights=weights))
    bleu_scores=sum(bleu_scores)/len(bleu_scores)
    result={}
    result[f'self_bleu'] = bleu_scores
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