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

"""Utilities for the class-based evaluation."""

from typing import Any, Mapping, Optional, Sequence, Tuple

from absl import logging
import t5.data
from t5.models import utils as model_utils
import tensorflow.compat.v2 as tf
import typing_extensions


AllOutputsType = Mapping[str, Sequence[Any]]
AllMetricsType = Mapping[str, Sequence[Mapping[str, Any]]]
OutputsAndMetricsType = Tuple[AllOutputsType, Optional[AllMetricsType]]


class PredictFnCallable(typing_extensions.Protocol):
  """Signature for `predict_fn` passed to `Evaluator.evaluate`."""

  def __call__(self, ds: tf.data.Dataset,
               **predict_fn_kwargs) -> Sequence[Tuple[int, Any]]: ...


class Evaluator:
  """A class to encapsulate all eval-related information.

  Users should define `predict_fn` and then pass it to `evaulate` method.
  `predict_fn` should operate with enumerated tf.data.Dataset. See `evaluate`
  method for more detail.

  evaluation data is cached once and will be used for arbitrary number of
  evaluation runs.

  Attributes:
    eval_tasks: a mapping from a mixture or a task name to t5.data.Task
      object(s).
    cached_ds: cached evaluation datasets.
    cached_examples: cached evaluation examples.
    cached_targets: cached evaluation targets.
    summary_writer: a tf summary writer for writing the evaluation results.
  """

  def __init__(self,
               mixture_or_task_name: str,
               feature_converter: t5.data.FeatureConverter = t5.data
               .EncDecFeatureConverter,
               eval_split: str = "validation",
               use_cached: bool = False,
               sequence_lengths: Mapping[str, int] = None,
               summary_dir: Optional[str] = None):
    """Evaluator constructor.

    Args:
      mixture_or_task_name: a registered task or mixture name.
      feature_converter: a feature converter object to use to convert the task
        features to model features. Must be a subclass of
        t5.data.FeatureConverter.
      eval_split: evaluation split. Typically "validation" or "test".
      use_cached: whether to use the cached dataset instead of processing it on
        the fly.
      sequence_lengths: an optional length specification. If specified, these
        will be the hard-limit on the evaluation data used for prediction. If
        unspecified, the maximum length for each feature will be used. These
        lengths are computed while caching the datasets.
      summary_dir: an optional directory to save the evaluation results in Event
        protocol buffer format.
    """
    eval_tasks = t5.data.get_subtasks(
        t5.data.get_mixture_or_task(mixture_or_task_name))
    self._eval_tasks = model_utils.get_valid_eval_tasks(eval_tasks, eval_split)

    if not self._eval_tasks:
      raise ValueError("No eval task with valid split and metric fn found.")

    def dataset_fn(task: t5.data.TaskV3) -> tf.data.Dataset:
      return task.get_dataset(
          sequence_length=None,
          split=eval_split,
          shuffle=False,
          use_cached=use_cached)

    # TODO(hwchung): move this function to eval or data utils.
    cached_examples, cached_targets, task_datasets, max_lengths = \
        model_utils.get_targets_and_examples(
            tasks=self._eval_tasks,
            dataset_fn=dataset_fn)

    if sequence_lengths is None:
      logging.info("Setting sequence lengths to %s", max_lengths)
      lengths = max_lengths
    elif (sequence_lengths["inputs"] < max_lengths["inputs"] or
          sequence_lengths["targets"] < max_lengths["targets"]):
      logging.warning(
          "Given sequence lengths are insufficient for some evaluation inputs "
          "or targets. These sequences will be truncated to fit, likely "
          "leading to sub-optimal results. Consider passing `None` for "
          "sequence_lengths to have them be automatically computed.\n Got: %s, "
          "\n Max Lengths:%s", sequence_lengths, max_lengths)
      lengths = sequence_lengths
    elif (sequence_lengths["inputs"] > max_lengths["inputs"] or
          sequence_lengths["targets"] > max_lengths["targets"]):
      logging.warning(
          "Given sequence lengths are longer than necessary for some "
          "evaluation inputs or targets, resulting in wasted computation. "
          "Consider passing `None` for sequence_lengths to have them be "
          "automatically computed.\n Got: %s,\n Max Lengths: %s",
          sequence_lengths, max_lengths)
      lengths = sequence_lengths

    self._cached_ds = {}
    # Convert the task features to model features
    for task in self._eval_tasks:
      eval_ds = feature_converter(task_datasets[task.name], lengths)

      # The eval dataset is enumerated to ensure that the order is preserved
      # throughout the entire evaluation process.
      eval_ds = eval_ds.enumerate()

      # Instead of caching a list of examples, we can cache in the form of
      # tf.data.Dataset.
      self._cached_ds[task.name] = eval_ds.cache()

    self._cached_examples = cached_examples
    self._cached_targets = cached_targets

    if summary_dir:
      with tf.compat.v1.Graph().as_default():
        self._summary_writer = tf.compat.v1.summary.FileWriter(summary_dir)

  def evaluate(self,
               *,
               compute_metrics: bool,
               step: Optional[int] = None,
               predict_fn: PredictFnCallable,
               **predict_fn_kwargs) -> OutputsAndMetricsType:
    """Predict and optionally compute metrics of self.eval.tasks.

    Evaluation must preserve the example ordering. This requirement is satisfied
    by using enumerated dataset. Each of the cached eval task datasets is an
    enumerated tf.data.Dataset where each element has (index, example) format.
    Therefore, each index serves as a unique integer id for the example.

    `predict_fn` takes as input the cached eval dataset. The output must be of
    the form (index, decoded) where `decoded` is the results of decoding
    `example` whose index matches `index`. Therefore, even if `predict_fn` mixes
    the order of the examples during prediction, the order can be corrected as
    long as the correct index for each example is maintained.

    A common example is the multi-host setup where the evaluation dataset is
    split into multiple hosts that independently make predictions and combine
    the results during which the ordering can be mixed.

    Args:
      compute_metrics: whether to compute metrics.
      step: an optional step number of the current evaluation. If unspecified, a
        dummy value of -1 will be used.
      predict_fn: a user-defined function, which takes in a tf.data.Dataset and
        outputs decoded predictions.
      **predict_fn_kwargs: kwargs to be passed to `predict_fn`.

    Returns:
      A tuple of outputs and metrics where the former corresponds to the output
      from `predict_fn` and the latter the computed metrics.
    """

    all_outputs = {}
    all_metrics = None
    for task in self.eval_tasks:
      outputs = predict_fn(self.cached_ds[task.name], **predict_fn_kwargs)

      if len(outputs[0]) != 2:
        raise ValueError("Output from the predict_fn should be a sequence of "
                         "length-2 tuple with (index, decoding) format")

      all_outputs[task.name]: Sequence[Any] = [
          x[1] for x in sorted(outputs, key=lambda x: x[0])
      ]

    if compute_metrics:
      all_metrics = {}
      for task in self.eval_tasks:
        metrics = []
        task_examples = self.cached_examples[task.name]
        for metric_fn in task.metric_fns:
          targets = self.cached_targets[task.name]
          predictions = [
              task.postprocess_fn(  # pylint:disable=g-complex-comprehension
                  d, example=ex, is_target=False)
              for d, ex in zip(all_outputs[task.name], task_examples)
          ]
          metrics.append(metric_fn(targets, predictions))
        all_metrics[task.name] = metrics

        self._log_eval_results(metrics, step, task_name=task.name)

    return all_outputs, all_metrics

  # TODO(hwchung): Support custom logging function metrics.
  def _log_eval_results(self, task_metrics: Sequence[Mapping[str, float]],
                        step: int, task_name: Optional[str] = None) -> None:
    """Log the eval results and optionally write summaries for TensorBoard."""
    if step is None:
      logging.warning("Step number for the logging session is not provided. "
                      "A dummy value of -1 will be used.")
      step = -1

    for task_metric in task_metrics:
      for metric_name, metric_value in task_metric.items():
        if self.summary_writer:
          summary = tf.compat.v1.Summary()

        tag = f"eval/{task_name}/{metric_name}"
        logging.info("%s at step %d: %.3f", tag, step, metric_value)

        if self.summary_writer:
          summary.value.add(tag=tag, simple_value=metric_value)
          self.summary_writer.add_summary(summary, step)

    if self.summary_writer:
      self.summary_writer.flush()

  @property
  def eval_tasks(self) -> Sequence[t5.data.TaskV3]:
    return self._eval_tasks

  @property
  def cached_ds(self) -> Mapping[str, tf.data.Dataset]:
    return self._cached_ds

  @property
  def cached_examples(self):
    return self._cached_examples

  @property
  def cached_targets(self) -> Mapping[str, Sequence[str]]:
    return self._cached_targets

  @property
  def summary_writer(self) -> tf.summary.SummaryWriter:
    return self._summary_writer
