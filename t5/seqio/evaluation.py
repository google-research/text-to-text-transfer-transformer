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

import inspect
import itertools
import os
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

AllOutputTokensType = Mapping[str, Sequence[Sequence[int]]]
AllOutputScoresType = Mapping[str, Sequence[float]]
AllMetricsType = Mapping[str, Sequence[Mapping[str, Any]]]
MetricsAndOutputsType = Tuple[
    Optional[AllMetricsType],  # metrics
    AllOutputTokensType,  # output_tokens
    AllOutputScoresType]  # output_scores


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

  def __call__(
      self,
      dataset: tf.data.Dataset,
      model_feature_lengths: Mapping[str, int]
  ) -> Sequence[Tuple[int, Sequence[int]]]: ...


class ScoreFnCallable(typing_extensions.Protocol):

  def __call__(
      self,
      dataset: tf.data.Dataset,
      model_feature_lengths: Mapping[str, int]
  ) -> Sequence[Tuple[int, float]]: ...


class LogFnCallable(typing_extensions.Protocol):

  def __call__(
      self,
      task_metrics: Mapping[str, float],
      step: int,
      task_name: str
  ) -> None: ...


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
    eval_tasks: a mapping from a mixture or a task name to seqio.Task object(s).
    cached_model_datasets: cached evaluation datasets with model features.
    cached_task_datasets: cached evaluation datasets with task features.
    cached_targets: cached evaluation targets.
    model_feature_lengths: mapping from model feature to its length in the
      `cached_model_datasets`.
    log_fn: a function called to log results.
  """

  def __init__(self,
               mixture_or_task_name: str,
               feature_converter: FeatureConverter = EncDecFeatureConverter,
               eval_split: str = "validation",
               use_cached: bool = False,
               sequence_length: Mapping[str, int] = None,
               summary_dir: Optional[str] = None,
               log_fn: Optional[LogFnCallable] = None):
    """Evaluator constructor.

    Args:
      mixture_or_task_name: a registered task or mixture name.
      feature_converter: a feature converter object to use to convert the task
        features to model features. Must be a subclass of
        seqio.FeatureConverter.
      eval_split: evaluation split. Typically "validation" or "test".
      use_cached: whether to use the cached dataset instead of processing it on
        the fly.
      sequence_length: an optional length specification. If specified, these
        will be the hard-limit on the evaluation data used for prediction. If
        none of the preprocessors depend on the sequence length, it can be left
        unspecified and the maximum length for each feature will be used. These
        lengths are computed while caching the datasets.
      summary_dir: an optional directory to save the evaluation results in Event
        protocol buffer format. If provided `log_fn` should be None.
      log_fn: an optional function to use to log evaluation results. If a custom
        logging function is provided `summary_dir` should be None.

    Raises:
      ValueError if `sequence_length` is None but a preprocessor depends on its
      value.
    """
    logging.info("Initializing Evaluator for '%s'", mixture_or_task_name)
    eval_tasks = dataset_providers.get_subtasks(
        dataset_providers.get_mixture_or_task(mixture_or_task_name))
    self._eval_tasks = get_valid_eval_tasks(eval_tasks, eval_split)

    if not self._eval_tasks:
      logging.warning(
          "No eval task with valid split and metric fn found. Skipping eval.")
      return

    # Determine if sequence_length arg is required. This occurs when any of the
    # task preprocessors have a `sequence_length` arg with no default value.
    sequence_length_required = False
    for task in eval_tasks:
      for prep in task.preprocessors:
        prep_params = inspect.signature(prep).parameters
        if ("sequence_length" in prep_params and
            prep_params["sequence_length"].default == inspect.Parameter.empty):
          if sequence_length is None:
            raise ValueError(
                f"Preprocessor '{prep.__name__}' in task '{task.name}' has a "
                "`sequence_length` argument, making it incompatible with "
                "automatic sequence length detection. Pass a valid "
                "`sequence_length` to `Evaluator` and try again.")
          sequence_length_required = True
          break

    def dataset_fn(task: Task) -> tf.data.Dataset:
      return task.get_dataset(
          sequence_length=sequence_length,
          split=eval_split,
          shuffle=False,
          use_cached=use_cached)

    # `task_datasets` have the output features from seqio.Task.get_dataset.
    # These features will be converted to "model features" by the feature
    # converter before being cached.
    cached_targets, cached_task_datasets, max_lengths = (
        get_targets_and_examples(tasks=self._eval_tasks, dataset_fn=dataset_fn))

    if sequence_length is None:
      logging.info("Setting sequence lengths to %s", max_lengths)
      sequence_length = max_lengths
    elif (sequence_length["inputs"] > max_lengths["inputs"] or
          sequence_length["targets"] > max_lengths["targets"]):
      logging.warning(
          "Given sequence lengths are longer than necessary for some "
          "evaluation inputs or targets, resulting in wasted computation. "
          "Consider passing `None` for `sequence_length` to have them be "
          "automatically computed.\n Got: %s,\n Max Lengths: %s",
          sequence_length, max_lengths)
    elif not sequence_length_required and (
        sequence_length["inputs"] == max_lengths["inputs"] or
        sequence_length["targets"] == max_lengths["targets"]):
      logging.warning(
          "Given sequence lengths *may be* insufficient for some evaluation "
          "inputs or targets. Such sequences will be truncated to fit, "
          "likely leading to sub-optimal results. Consider passing `None` "
          "for `sequence_length` to have them be automatically computed.\n")

    self._cached_model_datasets = {}
    # Convert the task features to model features
    for task in self._eval_tasks:
      eval_ds = feature_converter(
          cached_task_datasets[task.name], sequence_length)

      # The eval dataset is enumerated to ensure that the order is preserved
      # throughout the entire evaluation process.
      self._cached_model_datasets[task.name] = eval_ds.enumerate()

    self._cached_targets = cached_targets
    self._cached_task_datasets = cached_task_datasets
    self._model_feature_lengths = feature_converter.get_model_feature_lengths(
        sequence_length)

    if summary_dir is not None and log_fn is not None:
      raise ValueError(
          "If using a custom logging function a summary dir should not be "
          f"provided. Got: `log_fn`={log_fn} `summary_dir`={summary_dir}")
    self._log_fn = None
    if log_fn is not None:
      self._log_fn = log_fn
    else:
      # If there is a summary dir but not a custom log use the default logger.
      if summary_dir is not None:
        self._log_fn = TensorboardLogging(summary_dir)

  def evaluate(self,
               *,
               compute_metrics: bool,
               step: Optional[int] = None,
               predict_fn: PredictFnCallable,
               score_fn: ScoreFnCallable) -> MetricsAndOutputsType:
    """Predict and score self.eval_tasks.

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

    Similarly, `score_fn` takes the cached eval dataset as input and returns
    Sequence[(index, score)] where `score` is the sequence of log likelihood
    scores for the targets in the eval dataset.

    A common example is the multi-host setup where the evaluation dataset is
    split into multiple hosts that independently make predictions and combine
    the results during which the ordering can be mixed.

    There are 4 steps involved in the evaluation using predicted tokens:

    1. Model returns indices and output_tokens: Sequence[Tuple[int,
       Sequence[int]]]
    2. output tokens are decoded by `vocab.decode`
    3. Postprocessors are applied to the decoded output. These are denoted as
       predictions.
    4. Each metric function is applied to the predictions and the cached
       targets.

    There are 2 steps involved in the evaluation using scores:

    1. Model returns indices and scores: Sequence[Tuple[int, float]]
    2. Each metric function is applied to the scores and the cached targets.

    Args:
      compute_metrics: whether to compute metrics.
      step: an optional step number of the current evaluation. If unspecified, a
        dummy value of -1 will be used.
      predict_fn: a user-defined function, which takes in a tf.data.Dataset and
        outputs the sequence of predicted tokens. Only called if predict metrics
        exist for the tasks.
      score_fn: a user-defined function, which takes in a tf.data.Dataset and
        outputs the log likelihood score of the targets. Only called if score
        metrics exist for the task.

    Returns:
      metrics: a mapping from task name to computed metrics, or None if
        `compute_metrics` is False.
      predicted_tokens: a mapping from task name to the output tokens
        from `predict_fn`, for tasks that have `predict_metric_fns`.
      scores: a mapping from task name to the output scores from
        `score_fn` for tasks that have `score_predict_fns`.
    """

    all_output_tokens = {}
    all_output_scores = {}

    def _infer_and_sort_outputs(infer_fn, task_name):
      indices_and_outputs = infer_fn(self.cached_model_datasets[task_name])
      if len(indices_and_outputs[0]) != 2:
        raise ValueError(
            "Expected a sequence of length-2 tuples with (index, *) format.")
      return [x[1] for x in sorted(indices_and_outputs, key=lambda x: x[0])]

    for task in self.eval_tasks:
      logging.info("Evaluating %s", task.name)
      if task.predict_metric_fns:
        # output_tokens is a list of token_ids where each token_ids
        # corresponds to the model output of the input example.
        all_output_tokens[task.name] = _infer_and_sort_outputs(
            predict_fn, task.name)
      if task.score_metric_fns:
        all_output_scores[task.name] = _infer_and_sort_outputs(
            score_fn, task.name)

    if compute_metrics:
      all_metrics = self._compute_metrics(
          all_output_tokens, all_output_scores, step)
    else:
      all_metrics = None

    return all_metrics, all_output_tokens, all_output_scores

  def _compute_metrics(
      self,
      predicted_tokens: AllOutputTokensType,
      scores: AllOutputScoresType,
      step: Optional[int] = None) -> AllMetricsType:
    """Computes and logs metrics given the predicted tokens and scores.

    Args:
      predicted_tokens: a mapping from task name to the output tokens from
        `predict_fn`, for tasks that have `predict_metric_fns`.
      scores: a mapping from task name to the output scores from
        `score_fn` for tasks that have `score_predict_fns`.
      step: an optional step number of the current evaluation. If unspecified, a
        dummy value of -1 will be used.
    Returns:
      A mapping from task name to computed metrics.
    """
    all_metrics = {}

    for task in self.eval_tasks:
      logging.info("Computing metrics for %s", task.name)
      task_dataset = self.cached_task_datasets[task.name]
      targets = self.cached_targets[task.name]

      task_metrics = []

      if task.predict_metric_fns:
        task_vocab = task.output_features["targets"].vocabulary
        outputs = [
            task_vocab.decode([int(token) for token in tokens])
            for tokens in predicted_tokens[task.name]
        ]
        predictions = [
            task.postprocess_fn(d, example=ex, is_target=False)
            for d, ex in zip(outputs, tfds.as_numpy(task_dataset))
        ]

        task_metrics.extend([
            metric_fn(targets, predictions) for metric_fn in
            task.predict_metric_fns
        ])

      if task.score_metric_fns:
        task_metrics.extend([
            metric_fn(targets, scores[task.name])
            for metric_fn in task.score_metric_fns
        ])

      all_metrics[task.name] = {}
      for k, v in itertools.chain(*[m.items() for m in task_metrics]):
        if k in all_metrics[task.name]:
          raise ValueError(
              f"Duplicate metric key '{k}' in Task '{task.name}'.")
        all_metrics[task.name][k] = v

      if self._log_fn is not None:
        self._log_fn(all_metrics[task.name], step, task_name=task.name)
    return all_metrics

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
  def model_feature_lengths(self) -> Mapping[str, int]:
    return self._model_feature_lengths


class TensorboardLogging:
  """A class the encapulates summary writers to implement custom logging."""

  def __init__(self, summary_dir: str):
    """Log metrics to tensorboard.

    Args:
      summary_dir: The base directory where all logs will be written.
    """
    self._summary_dir = summary_dir
    self._summary_writers = {}

  def _get_summary_writer(self, task_name: str) -> tf.summary.SummaryWriter:
    """Create (if needed) and return a SummaryWriter for a given task."""
    if task_name not in self._summary_writers:
      with tf.compat.v1.Graph().as_default():
        self._summary_writers[task_name] = tf.compat.v1.summary.FileWriter(
            os.path.join(self._summary_dir, task_name))
    return self._summary_writers[task_name]

  def __call__(self, task_metrics: Mapping[str, float], step: int,
               task_name: str) -> None:
    """Log the eval results and optionally write summaries for TensorBoard.

    Note:
      This is the default implementation using tensorflow v1 operations.

    Args:
      task_metrics: A mapping from series names to numeric datapoints to be
        added to that series.
      step: The timestep to place this datapoint at.
      task_name: The name of the task these datapoints are relevant to.
    """
    if step is None:
      logging.warning("Step number for the logging session is not provided. "
                      "A dummy value of -1 will be used.")
      step = -1

    summary_writer = self._get_summary_writer(task_name)

    for metric_name, metric_value in task_metrics.items():
      summary = tf.compat.v1.Summary()

      tag = f"eval/{metric_name}"
      logging.info("%s at step %d: %.3f", tag, step, metric_value)

      summary.value.add(tag=tag, simple_value=metric_value)
      summary_writer.add_summary(summary, step)

    summary_writer.flush()
