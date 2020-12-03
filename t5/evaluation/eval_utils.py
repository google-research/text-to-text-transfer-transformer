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

"""Utility functions for running offline evaluation."""

import collections
import functools
import os

from typing import Any, Callable, Iterable, Mapping, MutableSequence, Optional, Sequence, Union

from absl import logging
import numpy as np
import pandas as pd
import t5.data
from t5.models import mesh_transformer
from t5.models import utils as model_utils
import tensorflow.compat.v1 as tf
import typing_extensions


class Metric(object):

  def __init__(self, name, group=None):
    self.name = name
    self.group = group or name

# This OrderedDict maps TensorBoard tags to nice-looking metric names.
# The order of the keys in the dict determine the order they get logged.
METRIC_NAMES = collections.OrderedDict([
    ("glue_average", Metric("Average GLUE Score")),
    ("glue_cola_v002/matthews_corrcoef", Metric("CoLA")),
    ("glue_sst2_v002/accuracy", Metric("SST-2")),
    ("glue_mrpc_v002/f1", Metric("MRPC (F1)", "MRPC")),
    ("glue_mrpc_v002/accuracy", Metric("MRPC (accuracy)", "MRPC")),
    ("glue_stsb_v002/pearson_corrcoef", Metric("STSB (Pearson)", "STSB")),
    ("glue_stsb_v002/spearman_corrcoef", Metric("STSB (Spearman)", "STSB")),
    ("glue_qqp_v002/f1", Metric("QQP (F1)", "QQP")),
    ("glue_qqp_v002/accuracy", Metric("QQP (accuracy)", "QQP")),
    ("glue_mnli_matched_v002/accuracy", Metric("MNLIm", "MNLI")),
    ("glue_mnli_mismatched_v002/accuracy", Metric("MNLImm", "MNLI")),
    ("glue_qnli_v002/accuracy", Metric("QNLI")),
    ("glue_rte_v002/accuracy", Metric("GLUE RTE")),
    ("cnn_dailymail_v002/rouge1", Metric("CNN/DM (ROUGE-1)", "CNN/DM")),
    ("cnn_dailymail_v002/rouge2", Metric("CNN/DM (ROUGE-2)", "CNN/DM")),
    ("cnn_dailymail_v002/rougeL", Metric("CNN/DM (ROUGE-L)", "CNN/DM")),
    ("cnn_dailymail_v002/rougeLsum", Metric("CNN/DM (ROUGE-L)", "CNN/DM")),
    ("squad_v010_allanswers/em", Metric("SQuAD (EM)", "SQuAD")),
    ("squad_v010_allanswers/f1", Metric("SQuAD (F1)", "SQuAD")),
    ("squad_v010_allanswers_span/em", Metric("SQuAD (EM)", "SQuAD")),
    ("squad_v010_allanswers_span/f1", Metric("SQuAD (F1)", "SQuAD")),
    ("squad_v010/em", Metric("SQuAD (EM)", "SQuAD")),
    ("squad_v010/f1", Metric("SQuAD (F1)", "SQuAD")),
    ("super_glue_average", Metric("Average SuperGLUE Score")),
    ("super_glue_boolq_v102/accuracy", Metric("BoolQ (accuracy)")),
    ("super_glue_cb_v102/mean_3class_f1", Metric("CB (F1)", "CB")),
    ("super_glue_cb_v102/accuracy", Metric("CB (accuracy)", "CB")),
    ("super_glue_copa_v102/accuracy", Metric("CoPA")),
    ("super_glue_multirc_v102/f1", Metric("MultiRC (F1)", "MultiRC")),
    ("super_glue_multirc_v102/exact_match", Metric("MultiRC (EM)", "MultiRC")),
    ("super_glue_record_v102/f1", Metric("ReCoRD (F1)", "ReCoRD")),
    ("super_glue_record_v102/em", Metric("ReCoRD (EM)", "ReCoRD")),
    ("super_glue_rte_v102/accuracy", Metric("SuperGLUE RTE")),
    ("super_glue_wic_v102/accuracy", Metric("WiC")),
    ("super_glue_wsc_v102_simple_eval/accuracy", Metric("WSC")),
    ("dpr_v001_simple/accuracy", Metric("DPR")),
    ("wmt_t2t_ende_v003/bleu", Metric("WMT T2T En-De")),
    ("wmt14_ende_v003/bleu", Metric("WMT14 En-De")),
    ("wmt15_enfr_v003/bleu", Metric("WMT15 En-Fr")),
    ("wmt16_enro_v003/bleu", Metric("WMT16 En-Ro")),
])

Event = collections.namedtuple("event", ["step", "value"])


def parse_events_files(tb_summary_dir):
  """Parse all TensorBoard events files in tb_summary_dir.

  Args:
    tb_summary_dir: str, path to look for events files in.

  Returns:
    A dict, where each key is a TensorBoard tag and each value is a list of
    Event tuples with step and value attributes.
  """
  events = collections.defaultdict(list)
  for events_file in tf.io.gfile.glob(os.path.join(tb_summary_dir, "events.*")):
    try:
      for e in tf.train.summary_iterator(events_file):
        for v in e.summary.value:
          events[v.tag].append(Event(e.step, v.simple_value))
    except tf.errors.DataLossError:
      logging.info("Skipping %s due to truncated record.", events_file)
  return events


def get_eval_metric_values(events):
  """Filter TensorBoard events to only include those for eval metrics.

  Args:
    events: dict of list of (step, value) tuples where keys are tags.

  Returns:
    Dict where key is task_name/metric_name and value is (step, value) tuple.
  """
  eval_values = {}
  for tag, event_values in events.items():
    if tag.startswith("eval"):
      _, task_name, metric_name = tag.split("/")
      eval_values["{}/{}".format(task_name, metric_name)] = event_values
  return eval_values


def sort_columns(df, metric_names=None):
  metric_names = metric_names or METRIC_NAMES
  column_order = list(collections.OrderedDict.fromkeys(
      [m.name for m in metric_names.values() if m.name in df.columns]
  ))
  return df.reindex(columns=column_order)


def compute_avg_glue(df, metric_names=None):
  """Compute average GLUE and SuperGLUE scores from a DataFrame.

  Will only compute a given average score if all of the metrics for that
  benchmark appear as columns in the DataFrame.

  Args:
    df: pandas.DataFrame, columns should be metric names.
    metric_names: dict mapping tensorboard tag to metric name.
  Returns:
    A pandas.DataFrame which has GLUE and SuperGLUE averages calculated.
  """
  # Use METRIC_NAMES defined at the top as default
  metric_names = metric_names or METRIC_NAMES
  all_glue_tags = {
      k for k in metric_names.keys() if "glue" in k and "average" not in k
  }
  superglue_tags = {k for k in all_glue_tags if "super" in k}
  glue_tags = all_glue_tags - superglue_tags
  average_keys = ["Average GLUE Score", "Average SuperGLUE Score"]
  for average_key, tags in zip(average_keys, [glue_tags, superglue_tags]):
    # Only compute average if all metric names appear as columns in the DF
    if {metric_names[t].name for t in tags}.issubset(set(df.columns)):
      # Compute average over each metric group
      group_to_metrics = collections.defaultdict(set)
      for tag in tags:
        metric = metric_names[tag]
        group_to_metrics[metric.group].add(metric.name)
      accum = None
      for metrics in group_to_metrics.values():
        group_avg = np.mean([df[k] for k in metrics], axis=0)
        accum = group_avg if accum is None else accum + group_avg
      # Compute average across all groups
      average = accum/len(group_to_metrics)
      df[average_key] = average
  return df


def scores_to_df(scores, metric_names=None):
  """Convert `scores` into a pandas DataFrame."""
  # Use METRIC_NAMES defined at the top as default
  metric_names = metric_names or METRIC_NAMES
  for tag in scores.keys():
    if tag not in metric_names:
      metric_names[tag] = Metric(tag)
      logging.warning(
          "TensorBoard tag %s not found in metric_names. "
          "Using tag as metric name.",
          tag)

  # Sort the tags in scores according to metric_names order
  sorted_tags = sorted(
      scores.keys(), key=lambda x: list(metric_names.keys()).index(x)
  )
  columns = [metric_names[t].name for t in sorted_tags]

  # Convert scores to dict with the format
  # {step_number: {tag1: value, tag2: value, ...}}
  step_scores = collections.defaultdict(
      lambda: collections.OrderedDict([(t, np.nan) for t in sorted_tags])
  )
  for tag in sorted_tags:
    for step, value in scores[tag]:
      step_scores[step][tag] = value
  sorted_items = sorted(list(step_scores.items()))
  data = [list(r.values()) for _, r in sorted_items]
  index = [s for s, _ in sorted_items]
  df = pd.DataFrame(data, index, columns)
  df.index.name = "step"
  return df


def metric_group_max(df, metric_names=None):
  """Find the step which achieves the highest mean value for a group of metrics."""
  # Use METRIC_NAMES defined at the top as default
  metric_names = metric_names or METRIC_NAMES
  group_to_metrics = collections.defaultdict(set)
  for metric in metric_names.values():
    group_to_metrics[metric.group].add(metric.name)
  group_df = pd.DataFrame()
  for group, metrics in group_to_metrics.items():
    if not all(m in df for m in metrics):
      continue
    group_df[group] = df[metrics].mean(axis=1)
  # Need to replace nan with large negative value for idxmax
  group_max_step = group_df.fillna(-1e9).idxmax(axis=0)
  metric_max = pd.Series()
  metric_max_step = pd.Series()
  for group_name, max_step in group_max_step.iteritems():
    for metric in group_to_metrics[group_name]:
      metric_max[metric] = df[metric][max_step]
      metric_max_step[metric] = max_step
  metric_max = metric_max.reindex(df.columns)
  metric_max_step = metric_max_step.reindex(df.columns)
  return metric_max, metric_max_step


def log_csv(df, metric_names=None, output_file=None):
  """Log scores to be copy/pasted into a spreadsheet."""
  logging.info(",".join(df.columns))
  metric_max, metric_max_step = metric_group_max(df, metric_names)
  max_row = "max," + ",".join("{:.3f}".format(m) for m in metric_max)
  logging.info(max_row)
  idx_row = "step," + ",".join("{:d}".format(i) for i in metric_max_step)
  logging.info(idx_row)

  if output_file is not None:
    with tf.io.gfile.GFile(output_file, "w") as f:
      csv_string = df.to_csv(float_format="%.3f")
      f.write(csv_string + max_row + "\n" + idx_row)


class PredictOrScoreFnCallable(typing_extensions.Protocol):
  """Signature for `predict_or_score_fn` passed to `run_eval`."""

  def __call__(
      self,
      checkpoint_step: int,
      vocabulary: Any,
      tasks: Sequence[t5.data.Task],
      examples: Sequence[Mapping[str, Mapping[str, str]]],
      datasets: Mapping[str, tf.data.Dataset],
      sequence_length: Union[None, Mapping[str, int]]
  ) -> MutableSequence[Union[str, float]]: ...


def run_eval(
    mixture_or_task_name: str,
    predict_or_score_fn: PredictOrScoreFnCallable,
    checkpoint_steps: Iterable[int],
    dataset_fn: Optional[Callable[
        [t5.data.Task, Mapping[str, int], int, str, Optional[bool]],
        tf.data.Dataset]] = None,
    summary_dir: Optional[str] = None,
    split: Optional[str] = "validation",
    sequence_length: Optional[Mapping[str, int]] = None,
    batch_size: Optional[int] = None):
  """Run evaluation on the given mixture or task.

  Args:
    mixture_or_task_name: str, the name of the Mixture or Task to evaluate
      on. Must be pre-registered in the global `TaskRegistry` or
      `MixtureRegistry.`
    predict_or_score_fn: function, This function takes in the sequence length,
      checkpoint step, tasks to evaluate, an eval_dataset_fn, a dict mapping
      task names to cached examples, a dict mapping task names to datasets,
      and returns a list of outputs or a list of scores.
    checkpoint_steps: an iterator with integers for checkpoint steps to
      evaluate on.
    dataset_fn: function, This function takes a task and returns the dataset
      associated with it. If None, the default mesh_eval_dataset_fn is used.
    summary_dir: str, path to write TensorBoard events file summaries for
      eval. If None, use model_dir/eval_{split}.
    split: str, the mixture/task split to evaluate on.
    sequence_length: an integer or a dict from feature-key to integer
      the sequence length to pad or truncate to,
      e.g. {"inputs": 512, "targets": 128}.
      If None, sequence length is automatically computed during eval.
    batch_size: integer, used only to check that expected padding matches the
      targets. If None, the check is skipped.
  """

  vocabulary = model_utils.get_vocabulary(mixture_or_task_name)

  tasks = t5.data.get_subtasks(
      t5.data.get_mixture_or_task(mixture_or_task_name))
  tasks = model_utils.get_valid_eval_tasks(tasks, split)

  if not tasks:
    logging.info(
        "All provided tasks have metric_fns=[] or no matching splits; "
        "eval is not possible.")
    return

  if not dataset_fn:
    def _get_task_eval_dataset(task, sequence_length, split):
      # TODO(sharannarang): Replace with more general function.
      eval_datasets = mesh_transformer.mesh_eval_dataset_fn(
          sequence_length=sequence_length,
          dataset_split=split,
          mixture_or_task_name=task.name,
      )

      return eval_datasets[0].dataset_fn()

    dataset_fn = _get_task_eval_dataset

  summary_writer = None

  cached_examples, cached_targets, cached_datasets, max_sequence_length = \
      model_utils.get_targets_and_examples(
          tasks=tasks,
          dataset_fn=functools.partial(
              dataset_fn, split=split, sequence_length=None))

  if summary_dir:
    model_utils.write_targets_and_examples(
        summary_dir, cached_targets, cached_examples)

  if sequence_length is None:
    logging.info("Setting sequence lengths to %s", max_sequence_length)
    sequence_length = max_sequence_length
  elif (sequence_length["inputs"] < max_sequence_length["inputs"] or
        sequence_length["targets"] < max_sequence_length["targets"]):
    logging.warning(
        "Given sequence lengths are insufficient for some evaluation inputs "
        "or targets. These sequences will be truncated to fit, likely "
        "leading to sub-optimal results. Consider passing `None` for "
        "sequence_length to have them be automatically computed.\n Got: %s, "
        "\n Max Lengths:%s", sequence_length, max_sequence_length)
  elif (sequence_length["inputs"] > max_sequence_length["inputs"] or
        sequence_length["targets"] > max_sequence_length["targets"]):
    logging.warning(
        "Given sequence lengths are longer than necessary for some "
        "evaluation inputs or targets, resulting in wasted computation. "
        "Consider passing `None` for sequence_length to have them be "
        "automatically computed.\n Got: %s,\n Max Lengths: %s",
        sequence_length, max_sequence_length)

  for step in checkpoint_steps:
    logging.info("Evaluating checkpoint step: %d", step)
    outputs = predict_or_score_fn(
        checkpoint_step=step,
        vocabulary=vocabulary,
        tasks=tasks,
        examples=cached_examples,
        datasets=cached_datasets,
        sequence_length=sequence_length)

    for task in tasks:
      # Extract the portion of decodes corresponding to this dataset
      examples = cached_examples[task.name]
      dataset_size = len(examples)

      predictions = [
          task.postprocess_fn(d, example=ex)
          for d, ex in zip(outputs[:dataset_size], examples)
      ]

      # Remove the used decodes.
      del outputs[:dataset_size]

      if summary_dir:
        predictions_filename = os.path.join(
            summary_dir,
            "{}_{}_predictions".format(task.name, step))
        model_utils.write_lines_to_file(predictions, predictions_filename)

      with tf.Graph().as_default():
        if summary_dir:
          summary_writer = summary_writer or tf.summary.FileWriter(
              summary_dir)

        for metric_fn in task.metric_fns:
          if summary_dir:
            summary = tf.Summary()
          targets = cached_targets[task.name]
          metric_result = metric_fn(targets, predictions)
          for metric_name, metric_value in metric_result.items():
            tag = "eval/{}/{}".format(task.name, metric_name)
            logging.info("%s at step %d: %.3f", tag, step, metric_value)
            if summary_dir:
              summary.value.add(tag=tag, simple_value=metric_value)
              summary_writer.add_summary(summary, step)  # pytype: disable=attribute-error
        if summary_dir:
          summary_writer.flush()  # pytype: disable=attribute-error

    # Only padding should remain.
    if batch_size:
      expected_pad = -sum(len(t)
                          for t in cached_targets.values()) % batch_size
      if outputs and len(outputs) != expected_pad:
        raise ValueError("{} padded outputs, {} expected.".format(
            len(outputs), expected_pad))

