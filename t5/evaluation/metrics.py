# Copyright 2020 The T5 Authors.
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

import re
from absl import logging
import numpy as np
import sacrebleu
import scipy.stats
import sklearn.metrics
from t5.evaluation import qa_utils

from rouge_score import rouge_scorer
from rouge_score import scoring


def bleu(targets, predictions):
  """Computes BLEU score.

  Args:
    targets: list of strings or list of list of strings if multiple references
      are present.
    predictions: list of strings

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
                                     tokenize="intl",
                                     use_effective_order=False)
  return {"bleu": bleu_score.score}


def rouge(targets, predictions, score_keys=None):
  """Computes rouge score.

  Args:
    targets: list of strings
    predictions: list of strings
    score_keys: list of strings with the keys to compute.
  Returns:
    dict with score_key: rouge score across all targets and predictions
  """

  if score_keys is None:
    score_keys = ["rouge1", "rouge2", "rougeLsum"]
  scorer = rouge_scorer.RougeScorer(score_keys)
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
  logging.info("sequence_accuracy = %.2f", seq_acc)
  return {"sequence_accuracy": seq_acc}


def pearson_corrcoef(targets, predictions):
  """Pearson correlation coefficient."""
  return {"pearson_corrcoef":
              100 * scipy.stats.pearsonr(targets, predictions)[0]}


def spearman_corrcoef(targets, predictions):
  """Spearman correlation coefficient."""
  return {"spearman_corrcoef":
              100 * scipy.stats.spearmanr(targets, predictions)[0]}


def matthews_corrcoef(targets, predictions):
  """Matthews correlation coefficient."""
  return {
      "matthews_corrcoef":
          100 * sklearn.metrics.matthews_corrcoef(targets, predictions)
  }


def mean_multiclass_f1(num_classes):
  """Computes the unweighted average of the F1 per class."""
  def my_metric(targets, predictions):
    return {
        "mean_%dclass_f1" % num_classes: 100 * sklearn.metrics.fbeta_score(
            targets, predictions, beta=1, labels=range(num_classes),
            average="macro")
    }
  return my_metric


def exact_match(targets, predictions):
  """Computes whether the targets match predictions exactly."""
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


def mean_group_metric(metric_fn, group_key="group", value_key="value"):
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
  """
  def my_metric(targets, predictions):
    """Computes mean of `metric_fn` over subgroups of results."""
    grouped_values = collections.defaultdict(lambda: ([], []))
    for targ, pred in zip(targets, predictions):
      g = targ[group_key]
      grouped_values[g][0].append(targ[value_key])
      grouped_values[g][1].append(pred[value_key])
    group_scores = collections.defaultdict(list)
    for (targets, predictions) in grouped_values.values():
      for metric, score in metric_fn(targets, predictions).items():
        group_scores[metric].append(score)
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
  """Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC).

  Args:
    targets: np.ndarray of targets, either 0 or 1, or continuous values.
    predictions: np.ndarray of predictions, any value.
    targets_threshold: float, if target values are continuous values, this
      threshold binarizes them.
  Returns:
    AUC ROC score.
  """

  if targets_threshold is not None:
    targets = np.array(targets)
    targets = np.where(targets < targets_threshold,
                       np.zeros_like(targets, dtype=np.int32),
                       np.ones_like(targets, dtype=np.int32))

  return {"auc": sklearn.metrics.roc_auc_score(targets, predictions)}
