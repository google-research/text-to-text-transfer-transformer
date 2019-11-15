# Copyright 2019 The T5 Authors.
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
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import re
from absl import logging
from allennlp.tools import squad_eval
import numpy as np
import sacrebleu
import scipy.stats
import sklearn.metrics
import tensorflow.compat.v1 as tf

from rouge_score import rouge_scorer
from rouge_score import scoring


def bleu(targets, predictions):
  """Computes BLEU score.

  Args:
    targets: list of strings
    predictions: list of strings

  Returns:
    bleu_score across all targets and predictions
  """
  # sacrebleu expects unicode
  predictions = [tf.compat.as_text(x) for x in predictions]
  targets = [tf.compat.as_text(x) for x in targets]

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
    summary = tf.compat.as_text(summary)
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


def span_qa(targets, predictions):
  """Computes question answering metrics for span prediction tasks.

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

  contexts = [space_tok(tf.compat.as_text(t["context"])) for t in targets]
  answers = [t["answers"] for t in targets]

  predictions = [space_tok(tf.compat.as_text(p)) for p in predictions]
  final_predictions = [
      get_answer_text_from_context(c, p) for c, p in zip(contexts, predictions)
  ]

  return qa(answers, final_predictions)


def qa(targets, predictions):
  """Computes question answering metrics, maximizing over answers per question.

  Args:
    targets: list of lists of strings
    predictions: list of strings

  Returns:
    dict with score_key: squad score across all targets and predictions
  """
  assert len(targets) == len(predictions)
  targets = [[tf.compat.as_text(t) for t in u] for u in targets]
  predictions = [tf.compat.as_text(p) for p in predictions]
  em = np.mean([
      squad_eval.metric_max_over_ground_truths(  # pylint:disable=g-complex-comprehension
          squad_eval.exact_match_score, p, t)
      for p, t in zip(predictions, targets)
  ])
  f1 = np.mean([
      squad_eval.metric_max_over_ground_truths(squad_eval.f1_score, p, t)
      for p, t in zip(predictions, targets)
  ])
  em *= 100
  f1 *= 100
  logging.info("EM = %.2f, F1 = %.2f", em, f1)
  return {"em": em, "f1": f1}


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
