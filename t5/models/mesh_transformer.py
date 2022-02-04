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

"""Functions for providing data to Mesh TF transformer."""

import functools

from absl import logging
import gin
import mesh_tensorflow.transformer.dataset as transformer_dataset
import t5.data
from t5.models import utils as model_utils
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds


DEPRECATED_GIN_REFERENCES = (
    "configurable_vocabulary",
    "get_sentencepiece_model_path",
    "maybe_print_dataset",
    "num_parallel_calls",
    "SentencePieceVocabulary",
    "t5.data.sentencepiece_vocabulary.SentencePieceVocabulary",
    "t5.models.mesh_transformer.get_sentencepiece_model_path",
    "train_model",
    "vocabularies.Vocabulary",
    "Vocabulary",
)


@gin.configurable()
def mesh_train_dataset_fn(
    mixture_or_task_name,
    sequence_length,
    vocabulary=None,
    dataset_split=tfds.Split.TRAIN,
    shuffle=True,
    seed=None,
    use_cached=False,
    pack=True):
  """Returns the tf.data.Dataset for training on a given mixture.

  This uses the format required for utils.run's `train_dataset_fn` argument in
  the Mesh TF transformer standalone.

  Args:
    mixture_or_task_name: string, an identifier for a Mixture or Task in the
      appropriate registry. Must be specified via gin.
    sequence_length: dict mapping feature key to the int length for that feature
      the max sequence length.
    vocabulary: unused argument, maintains compatibility with other dataset_fns.
    dataset_split: string, which split of the dataset to load. In most cases
      this should be "train".
    shuffle: Whether or not to shuffle dataset.
    seed: tf.int64 scalar tf.Tensor (or None). Used for both the global seed and
      shuffle seed for tf.data
    use_cached: bool, whether to load the cached version of this dataset.
    pack: bool, whether to pack the dataset.

  Returns:
    A tf.data.Dataset of preprocessed, tokenized, and batched examples.
  """
  del vocabulary
  mixture_or_task = t5.data.get_mixture_or_task(mixture_or_task_name)

  ds = mixture_or_task.get_dataset(
      sequence_length, split=dataset_split, use_cached=use_cached,
      shuffle=shuffle, num_epochs=None, seed=seed)

  # Select just the output features which are present in the dataset.
  feature_keys = tuple(k for k in mixture_or_task.output_features
                       if k in tf.data.get_output_shapes(ds))

  # Filtering feature keys is done in pack_or_pad function. However, when
  # packing is turned off, input_features aren't filtered leading to training
  # problems due to strings showing up in the input example. Filtering features
  # ensures that we don't rely on pack_or_pad to filter features for training.
  def _filter_features(ex):
    return {k: ex[k] for k in feature_keys}

  ds = ds.map(
      _filter_features, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  eos_keys = set(
      k for k, f in mixture_or_task.output_features.items() if f.add_eos)
  ds = transformer_dataset.pack_or_pad(
      ds, sequence_length, pack=pack,
      feature_keys=feature_keys, ensure_eos=eos_keys)
  return ds


@gin.configurable()
def mesh_inference_dataset_fn(
    mixture_or_task_name,
    sequence_length,
    dataset_split,
    shuffle=False,
    seed=None,
    vocabulary=None,
    num_inference_examples=-1,
    use_cached=False,
    priming_sequence_length=None):
  """Returns all tf.data.Datasets for LM inference on a given mixture.

  For Tasks without inputs (such as language modeling), the first
  `priming_sequence_length` tokens in the target are used as the "inputs" for
  inference.

  Args:
    mixture_or_task_name: string, an identifier for a Mixture or Task in the
      appropriate registry. Must be specified via gin.
    sequence_length: dict mapping feature key to the int length for that feature
      the max sequence length. If set to None, packing and padding will be
      disabled.
    dataset_split: string, which split of the dataset to load. NOTE, this
      function does NOT receive the split specified in utils.run. It needs to be
      specified separately.
    shuffle: Whether or not to shuffle dataset.
    seed: tf.int64 scalar tf.Tensor (or None). Used as shuffle seed for tf.data.
    vocabulary: unused argument, maintains compatibility with other dataaset_fns
    num_inference_examples: maximum number of examples per task to do inference
      on. If None or less than 0, use all examples.
    use_cached: bool, whether to load the cached version of this dataset.
      evals but should not be used for iterative decoding.
    priming_sequence_length: If the Task only has "targets", select the first
      this many tokens from each target sequence to use as "inputs". This is
      useful for decoder-only language models where you would like to use a
      portion of the targets as a priming sequence for generation.

  Returns:
    A list of mesh_tensorflow.transformer.dataset.EvalDataset tuples.
  """
  del vocabulary
  mixture_or_task = t5.data.get_mixture_or_task(mixture_or_task_name)

  def _split_targets_for_primed_inference(ex):
    ex["inputs"] = ex["targets"][:priming_sequence_length]
    ex["targets"] = ex["targets"][priming_sequence_length:]
    ex["inputs"] = tf.pad(
        ex["inputs"],
        [[0, priming_sequence_length - tf.shape(ex["inputs"])[0]]], "CONSTANT")
    ex["inputs"] = tf.reshape(ex["inputs"], shape=(priming_sequence_length,))
    return ex

  def _prepare_for_unprimed_inference(ex):
    ex["inputs"] = tf.constant([], dtype=tf.int64)
    return ex

  def _get_dataset_for_single_task(task, sequence_length):
    """Get a tensorflow.data.Dataset for the provided task."""

    ds = task.get_dataset(
        sequence_length, split=dataset_split, use_cached=use_cached,
        shuffle=shuffle, seed=seed)
    if "inputs" not in ds.element_spec:
      if not priming_sequence_length or priming_sequence_length <= 0:
        logging.warning("Priming sequence length not specified so priming "
                        "with the empty string.")
        ds = ds.map(_prepare_for_unprimed_inference)
      else:
        logging.info("Using the first %d tokens of each target as input.",
                     priming_sequence_length)
        ds = ds.map(_split_targets_for_primed_inference)
    elif priming_sequence_length is not None:
      raise ValueError(
          "Setting a priming sequence length only makes sense for decoder-only "
          "Tasks, which have `targets` but no `inputs`.")

    eos_keys = set(
        k for k, f in mixture_or_task.output_features.items() if f.add_eos)

    logging.info(
        "Padding '%s' with sequence lengths: %s", task.name, sequence_length)
    ds = transformer_dataset.pack_or_pad(
        ds,
        sequence_length,
        pack=False,
        feature_keys=tuple(task.output_features),
        ensure_eos=eos_keys)

    if num_inference_examples is not None and num_inference_examples >= 0:
      ds = ds.take(num_inference_examples)

    return ds

  outputs = []

  for task in t5.data.get_subtasks(mixture_or_task):
    if dataset_split not in task.splits:
      logging.info("Task %s has no '%s' split, skipping inference.",
                   task.name, dataset_split)
      continue

    outputs.append(
        transformer_dataset.EvalDataset(
            task.name,
            functools.partial(
                _get_dataset_for_single_task,
                task=task,
                sequence_length=sequence_length),
            task.postprocess_fn,
            task.metric_fns,
        )
    )

  if not outputs:
    logging.warning("No %s data found for %s.",
                    dataset_split, mixture_or_task_name)

  return outputs


@gin.configurable()
def mesh_eval_dataset_fn(
    mixture_or_task_name,
    sequence_length,
    dataset_split,
    vocabulary=None,
    num_eval_examples=-1,
    use_cached=False,
    pack=False,
    shuffle_eval_examples=False,
    seed=None):
  """Returns all tf.data.Datasets for evaluation on a given mixture.

  This uses the format required for utils.run's `eval_dataset_fn` argument in
  the Mesh TF transformer standalone.

  Args:
    mixture_or_task_name: string, an identifier for a Mixture or Task in the
      appropriate registry. Must be specified via gin.
    sequence_length: dict mapping feature key to the int length for that feature
      the max sequence length. If set to None, packing and padding will be
      disabled.
    dataset_split: string, which split of the dataset to load.
    vocabulary: unused argument, maintains compatibility with other dataaset_fns
    num_eval_examples: maximum number of examples per task to use for continuous
      eval. If None or less than 0, use all examples.
    use_cached: bool, whether to load the cached version of this dataset.
    pack: a boolean, whether to pack examples. This is useful for perplexity
      evals but should not be used for iterative decoding.
    shuffle_eval_examples: boolean, whether to shuffle eval examples, applied
      only when num_eval_examples is not None. Intended to be able to eval on a
      different eval slice at every iteration.
    seed: tf.int64 scalar tf.Tensor (or None). Used for both the global seed and
      shuffle seed for tf.data

  Returns:
    A list of mesh_tensorflow.transformer.dataset.EvalDataset tuples.
  """
  del vocabulary

  mixture_or_task = t5.data.get_mixture_or_task(mixture_or_task_name)

  def _get_dataset_for_single_task(task, sequence_length):
    """Get a tensorflow.data.Dataset for the provided task."""
    if shuffle_eval_examples and seed is None:
      logging.warning(("shuffle_seed_examples is true but no seed was ",
                       "provided. Using a random seed."))

    ds = task.get_dataset(
        sequence_length, split=dataset_split,
        use_cached=use_cached, shuffle=shuffle_eval_examples, seed=seed,
    )
    eos_keys = set(
        k for k, f in mixture_or_task.output_features.items() if f.add_eos)
    if sequence_length is None:
      logging.info(
          "Skipping packing/padding for '%s' since sequence length is None.",
          task.name)
    else:
      logging.info(
          "%sing '%s' with sequence lengths: %s",
          "Pack" if pack else "Padd", task.name, sequence_length)
      ds = transformer_dataset.pack_or_pad(
          ds,
          sequence_length,
          pack=pack,
          feature_keys=tuple(task.output_features),
          ensure_eos=eos_keys)

    if num_eval_examples is not None and num_eval_examples >= 0:
      ds = ds.take(num_eval_examples)

    return ds

  outputs = []

  for task in t5.data.get_subtasks(mixture_or_task):
    if dataset_split not in task.splits:
      logging.info(
          "Task %s has no '%s' split, skipping eval.", task.name, dataset_split
      )
      continue

    outputs.append(
        transformer_dataset.EvalDataset(
            task.name,
            functools.partial(
                _get_dataset_for_single_task,
                task=task,
                sequence_length=sequence_length),
            task.postprocess_fn,
            task.metric_fns,
        )
    )

  if not outputs:
    logging.warning("No %s data found for %s.",
                    dataset_split, mixture_or_task_name)

  return outputs


@gin.configurable()
def tsv_dataset_fn(
    filename,
    sequence_length,
    dataset_split,
    vocabulary,
    shuffle_buffer_size=10000):
  r"""Returns a dataset based on a TSV file formatted as `<input>\t<target>`."""
  # Currently `tf.gfile.glob` is broken on GCS, so we only read a file or
  # list of files.
  return transformer_dataset.packed_parallel_tsv_dataset(
      dataset=tf.data.TextLineDataset(filename).shuffle(shuffle_buffer_size),
      sequence_length=sequence_length,
      vocabulary=vocabulary,
      dataset_split=dataset_split,
      append_eos=True,
      eos_id=1)


@gin.configurable()
def get_vocabulary(mixture_or_task_name=None):
  """Get the appropriate value for the utils.run.vocabulary argument.

  Args:
    mixture_or_task_name: string, an identifier for a Mixture or Task in the
      appropriate registry. Must be specified via gin.

  Returns:
    Either a single t5.data.vocabularies.Vocabulary or a tuple of
    t5.data.vocabularies.Vocabulary for inputs and targets.
  """
  return model_utils.get_vocabulary(mixture_or_task_name)
