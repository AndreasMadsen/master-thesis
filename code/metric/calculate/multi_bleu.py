# Original version: https://gist.github.com/basaundi/4fc0a61419858c748b59
# Original Author: Ander Martinez Sanchez
# This is a modfied version, the changes are:
#  * turn it into an API, than a CLI tool
#  * change API such it takes a pretokenized dataset

from math import exp, log
from collections import Counter
from functools import reduce


def ngram_count(words, n):
    if n <= len(words):
        return Counter(zip(*[words[i:] for i in range(n)]))
    return Counter()


def max_count(c1, c2):
    return Counter({k: max(c1[k], c2[k]) for k in c1})


def min_count(c1, c2):
    return Counter({k: min(c1[k], c2[k]) for k in c1})


def closest_min_length(candidate, references):
    l0 = len(candidate)
    return min((abs(len(r) - l0), len(r)) for r in references)[1]


def safe_log(n):
    if n <= 0:
        return -9999999999
    return log(n)


def precision_n(candidate, references, n):
    ref_max = reduce(max_count, [ngram_count(ref, n) for ref in references])
    candidate_ngram_count = ngram_count(candidate, n)
    total = sum(candidate_ngram_count.values())
    correct = sum(reduce(min_count, (ref_max, candidate_ngram_count)).values())
    score = (correct / total) if total else 0
    return score, correct, total


def bleu(candidate, references, maxn=4):
    precs = [precision_n(candidate, references, n) for n in range(1, maxn+1)]
    bp = exp(1 - closest_min_length(candidate, references) / len(candidate))
    return bp * exp(sum(safe_log(precs[n]) for n in range(maxn)) / maxn)


def tokenize(txt):
    return txt.strip().split()


def tokenize_lower(txt):
    return txt.strip().lower().split()


def multi_bleu(candidates, all_references, maxn=4):
    correct = [0] * maxn
    total = [0] * maxn
    cand_tot_length = 0
    ref_closest_length = 0

    for candidate, references in zip(candidates, zip(*all_references)):
        cand_tot_length += len(candidate)
        ref_closest_length += closest_min_length(candidate, references)
        for n in range(maxn):
            sc, cor, tot = precision_n(candidate, references, n + 1)
            correct[n] += cor
            total[n] += tot

    precisions = [
        (correct[n] / total[n]) if correct[n] else 0 for n in range(maxn)
    ]

    if 0 < cand_tot_length < ref_closest_length:
        brevity_penalty = exp(1 - ref_closest_length / cand_tot_length)
    elif cand_tot_length == 0:
        brevity_penalty = 0
    else:
        brevity_penalty = 1

    score = 100 * brevity_penalty * exp(
                    sum(safe_log(precisions[n]) for n in range(maxn)) / maxn)
    prec_pc = [100 * p for p in precisions]
    return score, prec_pc, brevity_penalty, cand_tot_length, ref_closest_length
