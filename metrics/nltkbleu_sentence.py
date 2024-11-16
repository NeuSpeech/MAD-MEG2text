import datasets
import evaluate
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
    


import math
from collections import Counter
from fractions import Fraction
from nltk.translate.bleu_score import SmoothingFunction, closest_ref_length, brevity_penalty
from nltk.util import ngrams

def modified_precision(references, hypothesis, n):
    # Extracts all ngrams in hypothesis
    # Set an empty Counter if hypothesis is empty.
    counts = Counter(ngrams(hypothesis, n)) if len(hypothesis) >= n else Counter()
    # Extract a union of references' counts.
    # max_counts = reduce(or_, [Counter(ngrams(ref, n)) for ref in references])
    max_counts = {}
    for reference in references:
        reference_counts = (
            Counter(ngrams(reference, n)) if len(reference) >= n else Counter()
        )
        for ngram in counts:
            max_counts[ngram] = max(max_counts.get(ngram, 0), reference_counts[ngram])

    # Assigns the intersection between hypothesis and references' counts.
    clipped_counts = {
        ngram: min(count, max_counts[ngram]) for ngram, count in counts.items()
    }

    numerator = sum(clipped_counts.values())
    # Ensures that denominator is minimum 1 to avoid ZeroDivisionError.
    # Usually this happens when the ngram order is > len(reference).
    denominator = max(1, sum(counts.values()))

    return Fraction(numerator, denominator)#, _normalize=False)


def corpus_bleu(
    list_of_references,
    hypotheses,
    weights=(0.25, 0.25, 0.25, 0.25),
    smoothing_function=None,
    auto_reweigh=False,
):
    # Before proceeding to compute BLEU, perform sanity checks.
    p_numerators = Counter()
    p_denominators = Counter()
    hyp_lengths, ref_lengths = 0, 0

    assert len(list_of_references) == len(hypotheses), (
        "The number of hypotheses and their reference(s) should be the " "same "
    )

    try:
        weights[0][0]
    except TypeError:
        weights = [weights]
    max_weight_length = max(len(weight) for weight in weights)

    # Iterate through each hypothesis and their corresponding references.
    for references, hypothesis in zip(list_of_references, hypotheses):
        # For each order of ngram, calculate the numerator and
        # denominator for the corpus-level modified precision.
        for i in range(1, max_weight_length + 1):
            p_i = modified_precision(references, hypothesis, i)
            p_numerators[i] += p_i.numerator
            p_denominators[i] += p_i.denominator

        # Calculate the hypothesis length and the closest reference length.
        # Adds them to the corpus-level hypothesis and reference counts.
        hyp_len = len(hypothesis)
        hyp_lengths += hyp_len
        ref_lengths += closest_ref_length(references, hyp_len)

    # Calculate corpus-level brevity penalty.
    bp = brevity_penalty(ref_lengths, hyp_lengths)

    # Collects the various precision values for the different ngram orders.
    p_n = [
        Fraction(p_numerators[i], p_denominators[i]) # _normalize=False => modify by hj 240521
        for i in range(1, max_weight_length + 1)
    ]

    # Returns 0 if there's no matching n-grams
    # We only need to check for p_numerators[1] == 0, since if there's
    # no unigrams, there won't be any higher order ngrams.
    if p_numerators[1] == 0:
        return 0 if len(weights) == 1 else [0] * len(weights)

    # If there's no smoothing, set use method0 from SmoothinFunction class.
    if not smoothing_function:
        smoothing_function = SmoothingFunction().method0
    # Smoothen the modified precision.
    # Note: smoothing_function() may convert values into floats;
    #       it tries to retain the Fraction object as much as the
    #       smoothing method allows.
    p_n = smoothing_function(
        p_n, references=references, hypothesis=hypothesis, hyp_len=hyp_lengths
    )

    bleu_scores = []
    for weight in weights:
        # Uniformly re-weighting based on maximum hypothesis lengths if largest
        # order of n-grams < 4 and weights is set at default.
        if auto_reweigh:
            if hyp_lengths < 4 and weight == (0.25, 0.25, 0.25, 0.25):
                weight = (1 / hyp_lengths,) * hyp_lengths

        s = (w_i * math.log(p_i) for w_i, p_i in zip(weight, p_n) if p_i > 0)
        s = bp * math.exp(math.fsum(s))
        bleu_scores.append(s)
        
    return bleu_scores[0] if len(weights) == 1 else bleu_scores