# Copyright 2021 The T5 Authors.
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

from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

from absl import logging
from t5.seqio import dataset_providers
from t5.seqio import feature_converters
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import typing_extensions

Task = dataset_providers.Task
EncDecFeatureConverter = feature_converters.EncDecFeatureConverter
FeatureConverter = feature_converters.FeatureConverter

AllOutputsType = Mapping[str, Sequence[Any]]
AllMetricsType = Mapping[str, Sequence[Mapping[str, Any]]]
OutputsAndMetricsType = Tuple[AllOutputsType, Optional[AllMetricsType]]


def get_valid_eval_tasks(tasks: Sequence[Task], split: str) -> Sequence[Task]:
  """Get tasks that have the specified split and a metric function."""

  valid_tasks = []

  for task in tasks:
    if split not in task.splits:
      logging.info(
          "Task %s has no '%s' split; skipping eval.", task.name, split
      )
      continue
    if not task.metric_fns:
      logging.info("Task %s has no metric_fns; skipping eval.", task.name)
      continue
    valid_tasks.append(task)

  return valid_tasks


def get_targets_and_examples(
    tasks: Sequence[Task],
    dataset_fn: Callable[[Task], tf.data.Dataset]
) -> Tuple[
    Mapping[str, Any],
    Mapping[str, tf.data.Dataset],
    Mapping[str, int]]:
  """Get targets, cached datasets, and maximum sequence lengths per feature.

  Args:
    tasks: tasks objects to get targets and examples for.
    dataset_fn: function, returns the dataset from the task object.
  Returns:
    cached_targets: unpreprocessed targets for each task
    cached_task_datasets: cached datasets for each task, with cardinality set
    max_sequence_length: maximum sequence lengths for inputs and targets across
      all tasks.
  """
  # Pre-load in all of the targets once before entering continuous eval loop
  cached_targets = {}
  cached_task_datasets = {}

  max_sequence_length = {"inputs": 0, "targets": 0}

  for task in tasks:
    ds = dataset_fn(task).cache()

    targets = []

    for ex in tfds.as_numpy(ds):
      max_sequence_length["inputs"] = max(
          max_sequence_length["inputs"], len(ex["inputs"]))
      max_sequence_length["targets"] = max(
          max_sequence_length["targets"], len(ex["targets"]))

      # Create list of postprocessed targets
      if "targets_pretokenized" in ex:
        targets_pretokenized = ex["targets_pretokenized"]
        if isinstance(targets_pretokenized, bytes):
          targets_pretokenized = targets_pretokenized.decode("utf-8")
        targets.append(task.postprocess_fn(
            targets_pretokenized, example=ex, is_target=True))
      # TODO(hwchung): if fjord@ is using targets_pretokenized key for
      # evaluation, this else statement is no longer necessary. We may add the
      # detokenization logic here, i.e., detokenize ex["targets"] and
      # postprocess that instead of "targets_pretokenized".
      else:
        targets.append(task.postprocess_fn(
            tf.compat.as_text(ex["targets"]), example=ex, is_target=True))

    cached_targets[task.name] = targets
    cached_task_datasets[task.name] = ds.apply(
        tf.data.experimental.assert_cardinality(len(targets)))

  return cached_targets, cached_task_datasets, max_sequence_length


class PredictFnCallable(typing_extensions.Protocol):
  """Signature for `predict_fn` passed to `Evaluator.evaluate`."""

  def __call__(self,
               ds: tf.data.Dataset) -> Sequence[Tuple[int, Sequence[int]]]: ...


class Evaluator:
  """A class to encapsulate all eval-related information.

  Users should define `predict_fn` and then pass it to `evaulate` method.
  `predict_fn` should operate with enumerated tf.data.Dataset. See `evaluate`
  method for more detail.

  evaluation data is cached once and will be used for arbitrary number of
  evaluation runs.

  If none of the evaluation tasks has metrics functions defined, the evaluation
  will be skipped. `Evaluator.evaluate` will return ({}, {}) assuming that
  compute_metrics is True.

  Note that we cache two versions of the datasets. The first version
  (self.cached_task_datasets) has the task features (e.g., "inputs" and
  "targets"), which are returned from `seqio.Task.get_dataset`. The second
  version (self.cached_model_datasets) has model features (e.g.,
  "decoder_target_tokens"). This is returned from the feature converter. The
  former is used for postprocessing associated with the Task that requires the
  original task datasets. The latter is passed to `predict_fn` for evaluation.

  Attributes:
    eval_tasks: a mapping from a mixture or a task name to seqio.Task
      object(s).
    cached_model_datasets: cached evaluation datasets with model features.
    cached_task_datasets: cached evaluation datasets with task features.
    cached_targets: cached evaluation targets.
    summary_writer: a tf summary writer for writing the evaluation results.
  """

  def __init__(self,
               mixture_or_task_name: str,
               feature_converter: FeatureConverter = EncDecFeatureConverter,
               eval_split: str = "validation",
               use_cached: bool = False,
               sequence_lengths: Mapping[str, int] = None,
               summary_dir: Optional[str] = None):
    """Evaluator constructor.

    Args:
      mixture_or_task_name: a registered task or mixture name.
      feature_converter: a feature converter object to use to convert the task
        features to model features. Must be a subclass of
        seqio.FeatureConverter.
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
    eval_tasks = dataset_providers.get_subtasks(
        dataset_providers.get_mixture_or_task(mixture_or_task_name))
    self._eval_tasks = get_valid_eval_tasks(eval_tasks, eval_split)

    if not self._eval_tasks:
      logging.warning(
          "No eval task with valid split and metric fn found. Skipping eval.")
      return

    def dataset_fn(task: Task) -> tf.data.Dataset:
      return task.get_dataset(
          sequence_length=None,
          split=eval_split,
          shuffle=False,
          use_cached=use_cached)

    # `task_datasets` have the output features from seqio.Task.get_dataset.
    # These features will be converted to "model features" by the feature
    # converter before being cached.
    cached_targets, cached_task_datasets, max_lengths = (
        get_targets_and_examples(tasks=self._eval_tasks, dataset_fn=dataset_fn))

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

    self._cached_model_datasets = {}
    # Convert the task features to model features
    for task in self._eval_tasks:
      eval_ds = feature_converter(cached_task_datasets[task.name], lengths)

      # The eval dataset is enumerated to ensure that the order is preserved
      # throughout the entire evaluation process.
      self._cached_model_datasets[task.name] = eval_ds.enumerate()

    self._cached_targets = cached_targets
    self._cached_task_datasets = cached_task_datasets

    if summary_dir:
      with tf.compat.v1.Graph().as_default():
        self._summary_writer = tf.compat.v1.summary.FileWriter(summary_dir)

  def evaluate(self,
               *,
               compute_metrics: bool,
               step: Optional[int] = None,
               predict_fn: PredictFnCallable) -> OutputsAndMetricsType:
    """Predict and optionally compute metrics of self.eval.tasks.

    Evaluation must preserve the example ordering. This requirement is satisfied
    by using enumerated dataset. Each of the cached eval task datasets is an
    enumerated tf.data.Dataset where each element has (index, example) format.
    Therefore, each index serves as a unique integer id for the example.

    `predict_fn` takes as input the cached eval dataset. The output must be of
    the form Sequence[(index, token_ids)] where `token_ids` is the sequence of
    token ids output by the model with the input `example` whose index matches
    `index`. Therefore, even if `predict_fn` mixes the order of the examples
    during prediction, the order can be corrected as long as the correct index
    for each example is maintained.

    A common example is the multi-host setup where the evaluation dataset is
    split into multiple hosts that independently make predictions and combine
    the results during which the ordering can be mixed.


    Overall, there are 4 steps involved in the evaluation.

    1. Model returns indices and output_tokens: Sequence[Tuple[int,
       Sequence[int]]]
    2. output tokens are decoded by `vocab.decode`
    3. Postprocessors are applied to the decoded output. These are denoted as
       predictions.
    4. Each metric function is applied to the predictions and the cached
       targets.

    Note that non-string features are supported. For such use cases, users need
    to ensure the type consistency across the 4 steps. In other words, the
    return format of `vocab.decode` (which is not string in this case) is
    consistent with the input format of the registered postprocessor whose
    output format matches that of the input of the metric function.

    Args:
      compute_metrics: whether to compute metrics.
      step: an optional step number of the current evaluation. If unspecified, a
        dummy value of -1 will be used.
      predict_fn: a user-defined function, which takes in a tf.data.Dataset and
        outputs decoded predictions.

    Returns:
      A tuple of output_tokens and metrics where the former corresponds to the
      output tokens from `predict_fn` and the latter the computed metrics.
    """

    all_output_tokens = {}
    all_metrics = None

    for task in self.eval_tasks:
      indices_and_output_tokens = predict_fn(
          self.cached_model_datasets[task.name])

      if len(indices_and_output_tokens[0]) != 2:
        raise ValueError("Output from the predict_fn should be a sequence of "
                         "length-2 tuple with (index, decoding) format")

      all_output_tokens[task.name]: Sequence[Sequence[int]] = [
          x[1] for x in sorted(indices_and_output_tokens, key=lambda x: x[0])
      ]

    if compute_metrics:
      all_metrics = {}
      for task in self.eval_tasks:
        task_dataset = self.cached_task_datasets[task.name]
        targets = self.cached_targets[task.name]

        # output_tokens is a list of token_ids where each token_ids corresponds
        # to the model output of the input example.
        output_tokens = all_output_tokens[task.name]
        task_vocab = task.output_features["targets"].vocabulary
        outputs = [
            task_vocab.decode([int(token) for token in tokens])
            for tokens in output_tokens
        ]
        predictions = [
            task.postprocess_fn(d, example=ex, is_target=False)
            for d, ex in zip(outputs, tfds.as_numpy(task_dataset))
        ]

        metrics = [
            metric_fn(targets, predictions) for metric_fn in task.metric_fns
        ]
        all_metrics[task.name] = metrics

        self._log_eval_results(metrics, step, task_name=task.name)

    return all_output_tokens, all_metrics

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
  def eval_tasks(self) -> Sequence[Task]:
    return self._eval_tasks

  @property
  def cached_model_datasets(self) -> Mapping[str, tf.data.Dataset]:
    return self._cached_model_datasets

  @property
  def cached_task_datasets(self) -> Mapping[str, tf.data.Dataset]:
    return self._cached_task_datasets

  @property
  def cached_targets(self) -> Mapping[str, Sequence[str]]:
    return self._cached_targets

  @property
  def summary_writer(self) -> tf.summary.SummaryWriter:
    return self._summary_writer
