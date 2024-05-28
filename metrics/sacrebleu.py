from torchmetrics.text import BLEUScore
import datasets
import evaluate
import torch
metric = evaluate.load("sacrebleu")
def compute_metrics(preds, labels):
    labels=[[label] for i,label in enumerate(labels)]
    result={}
    bleu = metric.compute(predictions=preds, references=labels)
    for i in range(4):
        result[f'sacre bleu-{i+1}']=bleu['precisions'][i]
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
        return compute_metrics(predictions,references)