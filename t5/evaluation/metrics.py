# Copyright 2022 The T5 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functions for computing metrics.

Every function must accept a list of targets and a list of predictions and
return a dict of metrics.

Functions should assume all text inputs are unicode strings.
"""

import collections
import itertools
import re
import string
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

from absl import logging
import editdistance
import numpy as np
import sacrebleu
import scipy.stats
import sklearn.metrics
from t5.evaluation import qa_utils

from rouge_score import rouge_scorer
from rouge_score import scoring


def bleu(targets, predictions, tokenizer="intl"):
  """Computes BLEU score.

  Args:
    targets: list of strings or list of list of strings if multiple references
      are present.
    predictions: list of strings
    tokenizer: tokenizer option for corpus_bleu

  Returns:
    bleu_score across all targets and predictions
  """
  if isinstance(targets[0], list):
    targets = [[x for x in target] for target in targets]
  else:
    # Need to wrap targets in another list for corpus_bleu.
    targets = [targets]

  bleu_score = sacrebleu.corpus_bleu(predictions, targets,
                                     smooth_method="exp",
                                     smooth_value=0.0,
                                     force=False,
                                     lowercase=False,
                                     tokenize=tokenizer,
                                     use_effective_order=False)
  return {"bleu": bleu_score.score}


def rouge(targets, predictions, score_keys=None, tokenizer=None):
  """Computes rouge score.

  Args:
    targets: list of strings
    predictions: list of strings
    score_keys: list of strings with the keys to compute.
    tokenizer: Tokenizer object which has a tokenize() method.
  Returns:
    dict with score_key: rouge score across all targets and predictions
  """

  if score_keys is None:
    score_keys = ["rouge1", "rouge2", "rougeLsum"]
  scorer = rouge_scorer.RougeScorer(score_keys, tokenizer=tokenizer)
  aggregator = scoring.BootstrapAggregator()

  def _prepare_summary(summary):
    # Make sure the summary is not bytes-type
    # Add newlines between sentences so that rougeLsum is computed correctly.
    summary = summary.replace(" . ", " .\n")
    return summary

  for prediction, target in zip(predictions, targets):
    target = _prepare_summary(target)
    prediction = _prepare_summary(prediction)
    aggregator.add_scores(scorer.score(target=target, prediction=prediction))
  result = aggregator.aggregate()
  for key in score_keys:
    logging.info(
        "%s = %.2f, 95%% confidence [%.2f, %.2f]",
        key,
        result[key].mid.fmeasure*100,
        result[key].low.fmeasure*100,
        result[key].high.fmeasure*100,
    )
  return {key: result[key].mid.fmeasure*100 for key in score_keys}


def span_squad(targets, predictions):
  """Computes SQuAD metrics for span prediction tasks.

  Uses qa metric function to compute EM and F1 score.

  Args:
    targets: list of dict of answers (list of strings) and context (string)
    predictions: list of strings, each string is contains the space tokenized
      ids in the format: "start: 3 end: 6"

  Returns:
    dict with score_key: squad score across all targets and predictions
  """
  assert len(targets) == len(predictions)

  def space_tok(s):
    return re.sub(r"\W", " ", s).split()

  def get_answer_text_from_context(context, answer_tokens):
    """Find the answer in the context given the answer tokens."""
    # In the initial training iterations, the model can output garbage.
    # Returning an empty string in such cases.
    if len(answer_tokens) < 4:
      return ""

    # Model sometimes predicts words instead of numbers in the answer. Return
    # an empty string in that case.
    try:
      start_index = int(answer_tokens[1])
      end_index = int(answer_tokens[3])
    except ValueError:
      return ""

    return " ".join(context[start_index:end_index+1])

  contexts = [space_tok(t["context"]) for t in targets]
  answers = [t["answers"] for t in targets]

  predictions = [space_tok(p) for p in predictions]
  final_predictions = [
      get_answer_text_from_context(c, p) for c, p in zip(contexts, predictions)
  ]

  return squad(answers, final_predictions)


def squad(targets, predictions):
  """Computes SQuAD metrics, maximizing over answers per question.

  Args:
    targets: list of lists of strings
    predictions: list of strings

  Returns:
    dict with score_key: squad score across all targets and predictions
  """
  targets = [[qa_utils.normalize_squad(t) for t in u] for u in targets]
  predictions = [qa_utils.normalize_squad(p) for p in predictions]
  return qa_utils.qa_metrics(targets, predictions)


def trivia_qa(targets, predictions):
  """Computes TriviaQA metrics, maximizing over answers per question.

  Args:
    targets: list of lists of strings
    predictions: list of strings

  Returns:
    dict with score_key: squad score across all targets and predictions
  """
  targets = [[qa_utils.normalize_trivia_qa(t) for t in u] for u in targets]
  predictions = [qa_utils.normalize_trivia_qa(p) for p in predictions]
  return qa_utils.qa_metrics(targets, predictions)


def accuracy(targets, predictions):
  return {"accuracy": 100*sklearn.metrics.accuracy_score(targets, predictions)}


def sequence_accuracy(targets, predictions):
  """Computes per-sequence accuracy.

  For each example, returns 1.0 if the target sequence EXACTLY matches the
  predicted sequence. Else, 0.0.

  Args:
    targets: list of strings
    predictions: list of strings

  Returns:
    float. Average sequence-level accuracy.
  """
  assert len(targets) == len(predictions)
  seq_acc = 100 * np.mean([p == t for p, t in zip(predictions, targets)])
  return {"sequence_accuracy": seq_acc}


def pearson_corrcoef(targets, predictions):
  """Pearson correlation coefficient."""
  return {"pearson_corrcoef":
              100 * scipy.stats.pearsonr(targets, predictions)[0]}


def spearman_corrcoef(targets, predictions):
  """Spearman correlation coefficient."""
  return {"spearman_corrcoef":
              100 * scipy.stats.spearmanr(targets, predictions)[0]}


def mean_multiclass_f1(num_classes, **metric_fn_kwargs):
  """Computes the unweighted average of the F1 per class."""
  return sklearn_metrics_wrapper(
      "fbeta_score",
      metric_dict_str="mean_%dclass_f1" % num_classes,
      metric_post_process_fn=lambda x: 100 * x,
      beta=1,
      labels=range(num_classes),
      average="macro",
      **metric_fn_kwargs)


def all_match(targets, predictions):
  """Computes whether all targets match all predictions exactly."""
  return {"exact_match": 100 * float(np.array_equal(targets, predictions))}


def f1_score_with_invalid(targets, predictions):
  """Compute F1 score, but any prediction != 0 or 1 is counted as incorrect.

  Args:
    targets: np.ndarray of targets, either 0 or 1
    predictions: np.ndarray of predictions, any integer value

  Returns:
    F1 score, where any prediction != 0 or 1 is counted as wrong.
  """
  targets, predictions = np.asarray(targets), np.asarray(predictions)
  # Get indices of invalid predictions
  invalid_idx_mask = np.logical_and(predictions != 0, predictions != 1)
  # For any prediction != 0 or 1, set it to the opposite of what the target is
  predictions[invalid_idx_mask] = 1 - targets[invalid_idx_mask]
  return {"f1": 100 * sklearn.metrics.f1_score(targets, predictions)}


def deduplicate_metric(metric_fn,
                       group_key: str = "group",
                       value_key: str = "value"):
  """Returns a metric that only considers one example per group.

  Useful for things like ReCoRD where inputs may be replicated during training
  to handle multiple labels, but where at eval we only want a single copy of
  each example.

  Args:
    metric_fn: function, the metric to compute on the unique examples.
    group_key: the key for the grouping value in the target dictionary.
    value_key: the key for the value in the dictionaries.

  Returns:
    A metric function that deduplicated based on the grouping key before
    returning a metric.
  """
  def _deduplicated_metric(targets, predictions):
    """Deduplicate targets and predictions and pass that to the metric fn."""
    processed_groups = set()
    deduplicated_targets = []
    deduplicated_predictions = []
    for targ, pred in zip(targets, predictions):
      group = targ[group_key]
      if group in processed_groups:
        continue
      processed_groups.add(group)
      deduplicated_targets.append(targ[value_key])
      deduplicated_predictions.append(pred[value_key])
    return metric_fn(deduplicated_targets, deduplicated_predictions)
  return _deduplicated_metric


def mean_group_metric(metric_fn,
                      group_key="group",
                      value_key="value",
                      return_subgroup_scores=False):
  """Returns a metric that averages `metric_fn` on sub-groups of results.

  The sub-groups are defined by aggregating results (targets and predictions)
  by accessing the feature specified by `group_key` in the target dicts.

  **WARNING**: Using this function can produce unreliable results if you do not
  pass in full groups. For example, if you evaluate over a random subsample of a
  validation set and do not retain all of the examples in each group, you may
  get results which aren't directly comparable to using the full validation set.

  Args:
    metric_fn: function, the metric to compute on the subgroups.
    group_key: string, the key for the grouping value in the target dictionary.
    value_key: string, the key for the value in the dictionaries.
    return_subgroup_scores: If true, include the scores for each sub-group.
  """
  def my_metric(targets, predictions):
    """Computes mean of `metric_fn` over subgroups of results."""
    grouped_values = collections.defaultdict(lambda: ([], []))
    for targ, pred in zip(targets, predictions):
      g = targ[group_key]
      grouped_values[g][0].append(targ[value_key])
      grouped_values[g][1].append(pred[value_key])
    group_scores = collections.defaultdict(list)
    for group, (targets, predictions) in grouped_values.items():
      for metric, score in metric_fn(targets, predictions).items():
        group_scores[metric].append(score)
        if return_subgroup_scores:
          group_scores["%s-%s" % (group, metric)].append(score)
    return {metric: np.mean(scores) for metric, scores in group_scores.items()}
  return my_metric


def multirc_f1_over_all_answers(targets, predictions):
  """Special metric for MultiRC which computes F1 score over all examples.

  This is necessary because the targets/predictions for MultiRC are dicts and
  the f1_score_with_invalid expects a list of True/False labels, not dicts. As
  a result we just need to key in the "value" for each of the example dicts
  before feeding into f1_score_with_invalid.

  Args:
    targets: list of dicts, where each dict has a "value" key.
    predictions: list of dicts, where each dict has a "value" key.

  Returns:
    F1 score over values, where any prediction != 0 or 1 is counted as wrong.
  """
  return f1_score_with_invalid(
      [t["value"] for t in targets], [p["value"] for p in predictions]
  )


def auc(targets, predictions, targets_threshold=None):
  """Compute Area Under the ROC and PR curves.

  ROC - Receiver Operating Characteristic
  PR  - Precision and Recall

  Args:
    targets: np.ndarray of targets, either 0 or 1, or continuous values.
    predictions: np.ndarray of predictions, any value.
    targets_threshold: float, if target values are continuous values, this
      threshold binarizes them.

  Returns:
    A dictionary with AUC-ROC and AUC-PR scores.
  """

  if targets_threshold is not None:
    targets = np.array(targets)
    targets = np.where(targets < targets_threshold,
                       np.zeros_like(targets, dtype=np.int32),
                       np.ones_like(targets, dtype=np.int32))

  return {
      "auc-roc": sklearn.metrics.roc_auc_score(targets, predictions),
      "auc-pr": sklearn.metrics.average_precision_score(targets, predictions),
  }


def score_auc(targets, scores, targets_threshold=None):
  """Compute Area Under the ROC and PR curves.

  ROC - Receiver Operating Characteristic
  PR  - Precision and Recall

  Args:
    targets: np.ndarray of targets, either 0 or 1, or continuous values.
    scores: np.ndarray of scores, any value.
    targets_threshold: float, if target values are continuous values, this
      threshold binarizes them.

  Returns:
    A dictionary with AUC-ROC and AUC-PR scores.
  """

  return auc(
      targets=targets, predictions=scores, targets_threshold=targets_threshold)


def sklearn_metrics_wrapper(metric_str,
                            metric_dict_str=None,
                            metric_post_process_fn=None,
                            **metric_fn_kwargs):
  """Wraps any sklearn.metric function and returns a t5 metric function.

  Args:
    metric_str: string, the function from `sklearn.metrics` to use.
    metric_dict_str: optional string, if not specified `metric_str` is used as
      the key in the returned dictionary.
    metric_post_process_fn: callable, if specified the final computed metric
      will be passed through this.
    **metric_fn_kwargs: kwargs, passed to the metric function we are calling.

  Returns:
    the function that calculates the metric in a dict.
  """
  if not hasattr(sklearn.metrics, metric_str):
    raise ValueError("sklearn.metrics does not have: %s" % metric_str)

  def fn(targets, predictions):
    metric_fn = getattr(sklearn.metrics, metric_str)
    metric_val = metric_fn(targets, predictions, **metric_fn_kwargs)
    if metric_post_process_fn is not None:
      metric_val = metric_post_process_fn(metric_val)
    return {metric_dict_str or metric_str: metric_val}
  return fn


def rank_classification(
    targets: Sequence[Tuple[Sequence[int], bool, float, int]],
    scores: Sequence[float],
    num_classes: Optional[int] = None,
    normalize_by_target_length: bool = False,
    idx_len: int = 2,
) -> Dict[str, Union[float, int]]:
  """Computes standard metrics classification based on log likelihood ranking.

  This metric is intended to be used along with the `rank_classification`
  preprocessor and postprocessor. Each example is scored (by log likelihood)
  for every possible label, and the label with the best score is selected as the
  prediction.

  In the case of multiple labels, a prediction matching any will be considered
  correct.

  For problems with two labels, AUC-pr and AUC-roc retrieval metrics will be
  reported for the positive class, which is assumed to have an 'idx' of 1. If
  more labels are present, only accuracy and F-1 will be reported.

  Args:
    targets: list of tuples, the 'idx', 'is_correct', 'weight' fields, and
      length of target tokens from ground truth examples.
    scores: list of float, a flat list of log likelihood scores for each
      possible label for each example.
    num_classes: int or None, the number of possible classes for the label or
      None if the number of classes vary.
    normalize_by_target_length: bool, if True the scores are normalized by the
      target token lengths.
    idx_len: int, The number of elems in the idx field in the targets. This is
      generally 2 (input_id, target_id).

  Returns:
    Accuracy, f1, and AUC scores.

  Raises:
    ValueError: if `targets` is not a sequence of 3-tuples.
  """
  assert len(targets) == len(scores)
  if len(targets[0]) != 4:
    raise ValueError(
        f"`targets` should contain 4 elements but has {len(targets[0])}.")

  normalized_scores = []
  if normalize_by_target_length:
    for target, score in zip(targets, scores):
      _, _, _, target_length = target
      score = score / target_length
      normalized_scores.append(score)

    scores = normalized_scores

  idx_0 = targets[0][0]
  if not hasattr(idx_0, "__len__") or len(idx_0) != idx_len:
    raise ValueError("The first element of `targets` ('idx') should be "
                     f"{idx_len}-dimensional. Got {idx_0}.")

  # Sort by 'idx' since the function relies on this assumption.
  # ((idx, is_correct, weight), score)
  get_idx = lambda x: x[0][0]
  targets, scores = zip(*sorted(zip(targets, scores), key=get_idx))

  if not num_classes:
    # Assuming variable classes. Can only compute accuracy.
    num_correct = 0
    total = 0

    # (((input idx, output idx), is_correct, weight), score)
    get_grp = lambda x: x[0][0][0]

    for _, grp in itertools.groupby(zip(targets, scores), get_grp):
      exs, log_likelihoods = zip(*grp)
      prediction = np.argmax(log_likelihoods)
      weights = exs[prediction][2]
      num_correct += exs[prediction][1] * weights
      total += weights
    return {"accuracy": 100 * num_correct / total}

  assert len(targets) % num_classes == 0, f"{len(targets)} % {num_classes} != 0"

  labels_indicator = np.array([is_correct for _, is_correct, _, _ in targets
                              ]).reshape((-1, num_classes))
  weights = np.array([weight for _, _, weight, _ in targets]).reshape(
      (-1, num_classes))[:, 0]
  log_likelihoods = np.array(scores, np.float32).reshape((-1, num_classes))
  predictions = log_likelihoods.argmax(-1)

  if np.any(labels_indicator.sum(axis=-1) > 1):
    # multiple-answer case
    logging.info(
        "Multiple labels detected. Predictions matching any label will be "
        "considered correct.")
    num_examples = len(labels_indicator)
    return {
        "accuracy": (100 * np.average(
            labels_indicator[np.arange(num_examples), predictions],
            weights=weights))
    }

  predictions_indicator = np.eye(num_classes)[predictions]

  def exp_normalize(x):
    b = x.max(-1)[:, np.newaxis]
    y = np.exp(x - b)
    return y / y.sum(-1)[:, np.newaxis]
  probs = exp_normalize(log_likelihoods)

  metrics = {
      "accuracy":
          100 * sklearn.metrics.accuracy_score(
              labels_indicator, predictions_indicator, sample_weight=weights),
  }

  if num_classes > 2:
    metrics.update(
        mean_multiclass_f1(num_classes,
                           sample_weight=weights)(labels_indicator,
                                                  predictions_indicator))
    logging.warning("AUC-pr and AUC-roc are not supported when num_classes > 2")
  else:
    metrics.update({
        "f1":
            100 * sklearn.metrics.f1_score(
                labels_indicator.argmax(-1), predictions, sample_weight=weights)
    })
    labels_indicator = labels_indicator[:, 1]
    probs = probs[:, 1]

    metrics.update({
        "auc-roc":
            100 * sklearn.metrics.roc_auc_score(
                labels_indicator, probs, multi_class="ovr",
                sample_weight=weights, average="macro"),
        "auc-pr":
            100 * sklearn.metrics.average_precision_score(
                labels_indicator, probs, sample_weight=weights,
                average="macro"),
    })

  return metrics


def _coqa_tokenize(inp: str) -> Sequence[str]:
  """Normalize English text and tokenize into words based on spaces.

  Adapted from official evaluation tokenization at
  https://stanfordnlp.github.io/coqa/.

  Args:
    inp: string.

  Returns:
    Tokenization of normalized text as List[str]
  """

  def remove_articles(text):
    regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
    return re.sub(regex, " ", text)

  def normalize_whitespace(text):
    return " ".join(text.split())

  def remove_punc(text):
    exclude = set(string.punctuation)
    return "".join(ch for ch in text if ch not in exclude)

  return normalize_whitespace(remove_articles(remove_punc(inp.lower()))).split()


def _sequence_f1(target_tokens: Sequence[str],
                 prediction_tokens: Sequence[str]) -> float:
  """Given target and prediction tokens, return token-wise F1 score."""

  if not (target_tokens or prediction_tokens):
    return int(target_tokens == prediction_tokens)

  common_token_counts = (
      collections.Counter(target_tokens) &
      collections.Counter(prediction_tokens))
  sum_common = sum(common_token_counts.values())
  if sum_common == 0:
    return 0

  precision = 1.0 * sum_common / len(prediction_tokens)
  recall = 1.0 * sum_common / len(target_tokens)
  f1 = (2 * precision * recall) / (precision + recall)
  return f1


def coqa_f1(
    targets: Sequence[Sequence[str]], predictions: Sequence[str]
) -> Mapping[str, float]:
  """Return mean sequence F1 score over all QA turns."""
  f1s = []
  for (target, p) in zip(targets, predictions):
    assert isinstance(target, Sequence)
    prediction_tokens = _coqa_tokenize(p)
    example_f1s = [
        _sequence_f1(_coqa_tokenize(t), prediction_tokens) for t in target
    ]
    f1s.append(max(example_f1s))
  return {"f1": np.mean(np.array(f1s)) * 100}


def edit_distance(targets, predictions, lower=True):
  """Word-level edit distance between targets and predictions."""
  edit_distances = []
  for pred, target in zip(predictions, targets):
    if lower:
      pred = pred.lower()
      target = target.lower()

    # For simplicity, use regex-based tokenization that treats each
    # contiguous chunk of characters matched by \w as a word.
    pred = re.split("[^\\w]", pred)
    target = re.split("[^\\w]", target)
    edit_distances.append(editdistance.distance(pred, target))

  return {"min_edit": min(edit_distances),
          "max_edit": max(edit_distances),
          "mean_edit": np.mean(edit_distances),
          "median_edit": np.median(edit_distances),
          "sum_edit": sum(edit_distances)}
