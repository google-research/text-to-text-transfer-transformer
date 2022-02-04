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

"""Utilities for models."""

import functools
import os
import re
from typing import Iterable, Mapping, MutableSequence, Optional, Sequence, Union

from absl import logging

import gin
import numpy as np
import seqio
import t5.data
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds
import typing_extensions

# List of features used by model.
_MODEL_FEATURES = [
    "inputs", "inputs_position", "inputs_segmentation", "targets",
    "targets_position", "targets_segmentation", "targets_subsegmentation"
]


def filter_features(ex):
  """Filters example features, keeping only valid model features."""
  return {k: v for k, v in ex.items() if k in _MODEL_FEATURES}


def write_lines_to_file(lines, filename):
  """Write each line to filename, replacing the file if it exists."""
  if tf.io.gfile.exists(filename):
    tf.io.gfile.remove(filename)
  with tf.io.gfile.GFile(filename, "w") as output_file:
    output_file.write("\n".join([str(l) for l in lines]))


def get_step_from_checkpoint_path(checkpoint_path):
  """Returns the global step for the checkpoint at `checkpoint_path`.

  **Note**: This function works for checkpoints that are saved in the TF format
  only.

  Assumes `checkpoint_path` corresponds to a file which contains the substring
  model.ckpt-{global_step}

  Args:
    checkpoint_path: str of path to a checkpoint file.

  Returns:
    int of the global step corresponding to the checkpoint file.

  Raises:
    ValueError if checkpoint_path does not correspond to a model checkpoint file
    which contains the global_step in its filename.
  """
  match = re.match(r".*model\.ckpt\-(\d+).*", checkpoint_path)
  if match is None:
    raise ValueError("Invalid checkpoint path {}".format(checkpoint_path))
  return int(match.group(1))


def write_targets_and_examples(summary_dir, targets, datasets):
  """Writes plaintext targets and inputs to the summary directory.

  Args:
    summary_dir: str, directory to store plaintext targets and examples
    targets: dict, task_name -> targets for each task.
    datasets: dict, task_name -> tf.data.Dataset for each task.
  """
  if targets.keys() != datasets.keys():
    raise ValueError("Targets and datasets must have the same tasks.")

  for task in targets.keys():
    targets_filename = os.path.join(
        summary_dir,
        "{}_targets".format(task),
    )
    write_lines_to_file(targets[task], targets_filename)

    inputs = []
    for ex in tfds.as_numpy(datasets[task]):
      if "inputs_pretokenized" in ex:
        inputs.append(ex["inputs_pretokenized"])
      else:
        inputs.append(ex["inputs"])

    inputs_filename = os.path.join(
        summary_dir,
        "{}_inputs".format(task))

    write_lines_to_file(inputs, inputs_filename)


def get_vocabulary(mixture_or_task_name=None):
  """Return vocabulary from the mixture or task."""
  if not mixture_or_task_name:
    # Attempt to extract the mixture/task name from the gin config.
    try:
      mixture_or_task_name = gin.query_parameter("%MIXTURE_NAME")
    except ValueError:
      logging.warning("Could not extract mixture/task name from gin config.")
  if mixture_or_task_name:
    provider = t5.data.get_mixture_or_task(mixture_or_task_name)
    features = provider.output_features
    if "inputs" in features and "targets" in features:
      return (features["inputs"].vocabulary, features["targets"].vocabulary)
    else:
      feature_values = list(features.values())
      vocabulary = feature_values[0].vocabulary
      for feature in feature_values[1:]:
        if feature.vocabulary != vocabulary:
          logging.warning("No feature_name was provided to get_vocabulary, but "
                          "output_features have different vocabularies.")
          vocabulary = None
          break
      if vocabulary:
        return vocabulary
  logging.warning("Using default vocabulary.")
  return t5.data.get_default_vocabulary()


def get_latest_checkpoint_from_dir(model_dir):
  """Helper function to return the latest checkpoint number from a directory.

  Args:
    model_dir: str, Directory with checkpoint files.

  Returns:
    an int, latest checkpoint number.

  Raises:
    ValueError: if no checkpoints are found.
  """
  ckpt = tf.train.latest_checkpoint(model_dir)
  if ckpt is None:
    raise ValueError("No checkpoints found in model directory: %s" % model_dir)
  return int(re.sub(".*ckpt-", "", ckpt))


def get_checkpoints_iterator(checkpoint_steps, model_dir):
  """Get checkpoints from model directory.

  **Note**: This only works for models checkpoints saved using Tensorflow.

  Args:
    checkpoint_steps: list, int or str. If checkpoint_step is an int, find the
      checkpoint with the closest global step and return a singleton list. If
      checkpoint_step is a list of ints, replace each int with the path to the
      checkpoint with the closest global step. If checkpoint_step == "all",
      return the path of every checkpoint in model_dir, starting from the
      earliest checkpoint. if the checkpoint_steps is None, returns step from
      the tf.train.checkpoint_iterator for continuous eval. If -1, get the
      latest checkpoint from the model directory.
    model_dir: str, model directory. If model_dir is None, then checkpoint_steps
      must be an integer or list of integers.
  Returns:
    a iterator with the checkpoint steps (integers).
  """

  def _get_closest_checkpoint(target_checkpoint):
    """Returns checkpoint with closest global step to `target_checkpoint`."""
    checkpoints = set()
    for f in tf.io.gfile.listdir(model_dir):
      try:
        checkpoints.add(int(get_step_from_checkpoint_path(f)))
      except ValueError:
        continue
    if not checkpoints:
      raise ValueError("No checkpoint files found in {}".format(model_dir))
    closest = float("inf")
    for c in checkpoints:
      if abs(target_checkpoint - c) < abs(target_checkpoint - closest):
        closest = c
    if closest != target_checkpoint:
      logging.info(
          "Using checkpoint at step %d which is closest to requested step %d",
          closest,
          target_checkpoint,
      )
    return closest

  if checkpoint_steps is None:
    if model_dir is None:
      raise ValueError("checkpoint_steps and model_dir both cannot be None.")

    def _generate_checkpoints():
      for c in tf.train.checkpoints_iterator(model_dir):
        yield get_step_from_checkpoint_path(c)

    return _generate_checkpoints()

  elif checkpoint_steps == "all":
    if model_dir is None:
      raise ValueError(
          "model_dir cannot be None when checkpoint_steps={}".format(
              checkpoint_steps))
    ckpt_paths = tf.gfile.Glob(os.path.join(model_dir, "model.ckpt*"))
    return [get_step_from_checkpoint_path(c) for c in ckpt_paths]
  elif isinstance(checkpoint_steps, int):
    if model_dir:
      if checkpoint_steps == -1:
        return [get_latest_checkpoint_from_dir(model_dir)]
      else:
        return [_get_closest_checkpoint(checkpoint_steps)]
    else:
      return [checkpoint_steps]
  else:
    if model_dir:
      closests = np.unique(
          [_get_closest_checkpoint(c) for c in checkpoint_steps])
      return closests
    else:
      return checkpoint_steps


class PredictOrScoreFnCallable(typing_extensions.Protocol):
  """Signature for `predict_or_score_fn` passed to `run_eval`."""

  def __call__(
      self,
      checkpoint_step: int,
      vocabulary: seqio.Vocabulary,
      tasks: Sequence[seqio.Task],
      datasets: Mapping[str, tf.data.Dataset],
      sequence_length: Union[None, Mapping[str, int]]
  ) -> MutableSequence[Union[str, float]]: ...


class DatasetFnCallable(typing_extensions.Protocol):

  def __call__(
      self,
      task: seqio.Task,
      sequence_length: Mapping[str, int],
      split: str,
  ) -> tf.data.Dataset: ...


def run_eval(
    mixture_or_task_name: str,
    predict_or_score_fn: PredictOrScoreFnCallable,
    checkpoint_steps: Iterable[int],
    dataset_fn: DatasetFnCallable,
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
      associated with it.
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

  vocabulary = get_vocabulary(mixture_or_task_name)

  tasks = t5.data.get_subtasks(
      t5.data.get_mixture_or_task(mixture_or_task_name))
  tasks = seqio.evaluation.get_valid_eval_tasks(tasks, split)

  if not tasks:
    logging.info(
        "All provided tasks have metric_fns=[] or no matching splits; "
        "eval is not possible.")
    return

  summary_writer = None

  cached_targets, cached_datasets, max_sequence_length = (
      seqio.evaluation.get_targets_and_examples(
          tasks=tasks,
          dataset_fn=functools.partial(
              dataset_fn, split=split, sequence_length=None),
          sequence_dims={}))

  if summary_dir:
    write_targets_and_examples(summary_dir, cached_targets, cached_datasets)

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
        datasets=cached_datasets,
        sequence_length=sequence_length)

    for task in tasks:
      # Extract the portion of decodes corresponding to this dataset
      dataset = cached_datasets[task.name]
      dataset_size = len(cached_targets[task.name])
      predictions = [
          task.postprocess_fn(d, example=ex)
          for d, ex in zip(outputs[:dataset_size], tfds.as_numpy(dataset))
      ]

      if summary_dir:
        outputs_filename = os.path.join(
            summary_dir,
            "{}_{}_outputs".format(task.name, step))
        write_lines_to_file(outputs[:dataset_size], outputs_filename)
        predictions_filename = os.path.join(
            summary_dir,
            "{}_{}_predictions".format(task.name, step))
        write_lines_to_file(predictions, predictions_filename)

      # Remove the used decodes.
      del outputs[:dataset_size]

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
