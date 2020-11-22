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

"""Utilities for models."""

import os
import re

from absl import logging

import gin
import numpy as np
import t5.data
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds


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


def get_valid_eval_tasks(tasks, split):
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


def write_targets_and_examples(summary_dir, targets, examples):
  """Writes plaintext targets and inputs to the summary directory.

  Args:
    summary_dir: str, directory to store plaintext targets and examples
    targets: dict, task_name -> targets for each task.
    examples: dict, task_name -> examples for each task.
  """
  if targets.keys() != examples.keys():
    raise ValueError("Targets and examples must have the same tasks.")

  for task in targets.keys():
    targets_filename = os.path.join(
        summary_dir,
        "{}_targets".format(task),
    )
    write_lines_to_file(targets[task], targets_filename)

    inputs = []
    for ex in examples[task]:
      if "inputs_plaintext" in ex:
        inputs.append(ex["inputs_plaintext"])
      else:
        inputs.append(ex["inputs"])

    inputs_filename = os.path.join(
        summary_dir,
        "{}_inputs".format(task))

    write_lines_to_file(inputs, inputs_filename)


def get_targets_and_examples(tasks, dataset_fn):
  """Get targets and examples.

  Args:
    tasks: list, contains tasks objects.
    dataset_fn: function, returns the dataset from the task object.
  Returns:
    Dict of plaintext examples for each task, list of plaintext targets for each
    task, a dict of datasets for each task, and a dict with max sequence lengths
    for inputs and targets.
  """
  # Pre-load in all of the targets once before entering continuous eval loop
  cached_targets = {}
  cached_examples = {}
  cached_datasets = {}

  max_sequence_length = {"inputs": 0, "targets": 0}

  for task in tasks:
    ds = dataset_fn(task)

    examples = []
    targets = []

    for ex in tfds.as_numpy(ds):
      max_sequence_length["inputs"] = max(
          max_sequence_length["inputs"], len(ex["inputs"]))
      max_sequence_length["targets"] = max(
          max_sequence_length["targets"], len(ex["targets"]))

      examples.append(ex)

      # Create list of postprocessed targets
      if "targets_plaintext" in ex:
        targets.append(task.postprocess_fn(
            tf.compat.as_text(ex["targets_plaintext"]),
            example=ex, is_target=True))
      else:
        targets.append(task.postprocess_fn(
            tf.compat.as_text(ex["targets"]), example=ex, is_target=True))

    cached_targets[task.name] = targets
    cached_examples[task.name] = examples
    cached_datasets[task.name] = ds

  return cached_examples, cached_targets, cached_datasets, max_sequence_length


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


