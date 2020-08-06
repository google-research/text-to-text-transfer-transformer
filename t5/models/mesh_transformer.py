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

"""Functions for providing data to Mesh TF transformer."""

import functools
import warnings

from absl import logging
import gin
import mesh_tensorflow.transformer.dataset as transformer_dataset

import t5.data
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds


def valid_vocabulary(vocabulary):
  """Tests that a vocabulary is valid to pass to Mesh Tensorflow transformer."""
   # Mesh TF allows for a (inputs_vocab, targets_vocab) tuple
  if not isinstance(vocabulary, tuple):
    vocabulary = (vocabulary,)
  for v in vocabulary:
    if not isinstance(v, t5.data.vocabularies.Vocabulary):
      raise ValueError("vocabulary must be a t5.data.vocabularies.Vocabulary")


@gin.configurable()
def mesh_train_dataset_fn(
    mixture_or_task_name,
    sequence_length,
    vocabulary,
    dataset_split=tfds.Split.TRAIN,
    use_cached=False):
  """Returns the tf.data.Dataset for training on a given mixture.

  This uses the format required for utils.run's `train_dataset_fn` argument in
  the Mesh TF transformer standalone.

  Args:
    mixture_or_task_name: string, an identifier for a Mixture or Task in the
      appropriate registry. Must be specified via gin.
    sequence_length: dict mapping feature key to the int length for that feature
      the max sequence length.
    vocabulary: a t5.data.vocabularies.Vocabulary.
    dataset_split: string, which split of the dataset to load. In most cases
      this should be "train".
    use_cached: bool, whether to load the cached version of this dataset.

  Returns:
    A tf.data.Dataset of preprocessed, tokenized, and batched examples.
  """
  valid_vocabulary(vocabulary)

  mixture_or_task = t5.data.get_mixture_or_task(mixture_or_task_name)

  ds = mixture_or_task.get_dataset(
      sequence_length, split=dataset_split, use_cached=use_cached, shuffle=True)
  if any(not f.add_eos for f in mixture_or_task.output_features.values()):
    warnings.warn(
        "pack_or_pad is being called with ensure_eos=True, but EOS is not "
        "being added to all features."
    )

  # Select just the output features which are present in the dataset.
  feature_keys = tuple(k for k in mixture_or_task.output_features
                       if k in tf.data.get_output_shapes(ds))
  ds = transformer_dataset.pack_or_pad(
      ds, sequence_length, pack=True,
      feature_keys=feature_keys, ensure_eos=True)
  return ds


def maybe_shuffle_and_subsample_dataset(
    ds,
    num_eval_examples=None,
    shuffle_eval_examples=False,
    shuffle_buffer_size=t5.data.utils.SHUFFLE_BUFFER_SIZE):
  """Takes only `num_eval_examples` and shuffles examples if needed."""

  if num_eval_examples is None:
    return ds
  if shuffle_eval_examples:
    ds = ds.shuffle(shuffle_buffer_size, reshuffle_each_iteration=True)
  ds = ds.take(num_eval_examples)
  return ds


@gin.configurable()
def mesh_eval_dataset_fn(
    mixture_or_task_name,
    sequence_length,
    vocabulary,
    dataset_split,
    num_eval_examples=None,
    use_cached=False,
    pack=False,
    shuffle_eval_examples=False,
    shuffle_buffer_size=t5.data.utils.SHUFFLE_BUFFER_SIZE):
  """Returns all tf.data.Datasets for evaluation on a given mixture.

  This uses the format required for utils.run's `eval_dataset_fn` argument in
  the Mesh TF transformer standalone.

  Args:
    mixture_or_task_name: string, an identifier for a Mixture or Task in the
      appropriate registry. Must be specified via gin.
    sequence_length: dict mapping feature key to the int length for that feature
      the max sequence length.
    vocabulary: a t5.data.vocabularies.Vocabulary.
    dataset_split: string, which split of the dataset to load.
    num_eval_examples: maximum number of examples per task to use for continuous
      eval. If None, use all examples.
    use_cached: bool, whether to load the cached version of this dataset.
    pack: a boolean, whether to pack examples. This is useful for perplexity
      evals but should not be used for iterative decoding.
    shuffle_eval_examples: boolean, whether to shuffle eval examples, applied
      only when num_eval_examples is not None. Intended to be able to eval on a
      different eval slice at every iteration.
    shuffle_buffer_size: integer - the shuffle buffer size if we shuffle
      eval examples, ideally this should be some large multiple of
      `num_eval_examples` to ensure good mixing and random batches.

  Returns:
    A list of mesh_tensorflow.transformer.dataset.EvalDataset tuples.
  """
  valid_vocabulary(vocabulary)

  mixture_or_task = t5.data.get_mixture_or_task(mixture_or_task_name)

  def _get_dataset_for_single_task(task):
    """Get a tensorflow.data.Dataset for the provided task."""
    ds = task.get_dataset(
        sequence_length, split=dataset_split,
        use_cached=use_cached, shuffle=False
    )
    if any(not f.add_eos for f in task.output_features.values()):
      warnings.warn(
          "pack_or_pad is being called with ensure_eos=True, but EOS is not "
          "being added to all features."
      )
    ds = transformer_dataset.pack_or_pad(
        ds,
        sequence_length,
        pack=pack,
        feature_keys=tuple(task.output_features),
        ensure_eos=True)
    ds = maybe_shuffle_and_subsample_dataset(
        ds, num_eval_examples, shuffle_eval_examples, shuffle_buffer_size)
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
            functools.partial(_get_dataset_for_single_task, task),
            task.postprocess_fn,
            task.metric_fns,
        )
    )

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
def get_vocabulary(mixture_or_task_name):
  """Get the appropriate value for the utils.run.vocabulary argument.

  Args:
    mixture_or_task_name: string, an identifier for a Mixture or Task in the
      appropriate registry. Must be specified via gin.

  Returns:
    Either a single t5.data.vocabularies.Vocabulary or a tuple of
    t5.data.vocabularies.Vocabulary for inputs and targets.
  """
  provider = t5.data.get_mixture_or_task(mixture_or_task_name)
  features = provider.output_features
  if "inputs" in features and "targets" in features:
    return (features["inputs"].vocabulary, features["targets"].vocabulary)
  else:
    return provider.get_vocabulary()


@gin.configurable()
def get_sentencepiece_model_path(mixture_or_task_name):
  """Return the SentencePiece model path for a given mixture or task.

  DEPRECATED. Please pass the vocabulary directly to utils.run instead.

  Args:
    mixture_or_task_name: string, an identifier for a Mixture or Task in the
      appropriate registry. Must be specified via gin.

  Returns:
    Path to a SentencePiece model file.
  """
  warnings.warn(
      "get_sentencepiece_model_path is deprecated. Please pass the mixture or "
      "task vocabulary directly to the Mesh TensorFlow Transformer instead."
  )
  provider = t5.data.get_mixture_or_task(mixture_or_task_name)
  vocabulary = provider.get_vocabulary()
  if not isinstance(vocabulary,
                    t5.data.sentencepiece_vocabulary.SentencePieceVocabulary):
    raise ValueError(
        "get_sentencepiece_model_path was called for a provider that does not "
        "use a SentencePieceVocabulary."
    )
  return vocabulary.sentencepiece_model_file
