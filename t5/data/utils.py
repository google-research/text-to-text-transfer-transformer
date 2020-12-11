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

# Lint as: python3
"""Utilities for data loading and processing."""

import contextlib
import functools
import inspect
import os
from typing import Mapping

from absl import logging
import gin
import numpy as np
from t5.data import vocabularies
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

_INFO_FILENAME = "info.{split}.json"
_STATS_FILENAME = "stats.{split}.json"
_TFRECORD_PREFIX = "{split}.tfrecord"

_TFDS_DATA_DIR_OVERRIDE = None
_GLOBAL_CACHE_DIRECTORIES = []

DEFAULT_SPM_PATH = "gs://t5-data/vocabs/cc_all.32000/sentencepiece.model"  # GCS
DEFAULT_EXTRA_IDS = 100


def get_default_vocabulary():
  return vocabularies.SentencePieceVocabulary(
      DEFAULT_SPM_PATH, DEFAULT_EXTRA_IDS)


def set_tfds_data_dir_override(tfds_data_dir):
  global _TFDS_DATA_DIR_OVERRIDE
  _TFDS_DATA_DIR_OVERRIDE = tfds_data_dir


def get_global_cache_dirs():
  return _GLOBAL_CACHE_DIRECTORIES


def set_global_cache_dirs(global_cache_dirs):
  global _GLOBAL_CACHE_DIRECTORIES
  _GLOBAL_CACHE_DIRECTORIES = global_cache_dirs


def add_global_cache_dirs(global_cache_dirs):
  global _GLOBAL_CACHE_DIRECTORIES
  _GLOBAL_CACHE_DIRECTORIES += global_cache_dirs


class LazyTfdsLoader(object):
  """Wrapper for TFDS datasets with memoization and additional functionality.

  Lazily loads info from TFDS and provides memoization to avoid expensive hidden
  file operations. Also provides additional utility methods.
  """

  _MEMOIZED_BUILDERS = {}

  def __init__(self, name, data_dir=None, split_map=None):
    """LazyTfdsLoader constructor.

    Args:
      name: str, the name of the TFDS dataset.
      data_dir: str (optional), directory to read/write TFDS data.
      split_map: dict (optional), mapping from canonical splits
        (e.g., 'validation') to TFDS splits or slices
        (e.g., 'train[':1%']).
    """
    self._name = name
    self._data_dir = data_dir
    self._split_map = split_map

  @property
  def name(self):
    return self._name

  @property
  def data_dir(self):
    if _TFDS_DATA_DIR_OVERRIDE:
      if self._data_dir:
        logging.warning(
            "Overriding TFDS data directory '%s' with '%s' for dataset '%s'.",
            self._data_dir, _TFDS_DATA_DIR_OVERRIDE, self.name)
      return _TFDS_DATA_DIR_OVERRIDE
    return self._data_dir

  @property
  def builder(self):
    builder_key = (self.name, self.data_dir)
    if builder_key not in LazyTfdsLoader._MEMOIZED_BUILDERS:
      LazyTfdsLoader._MEMOIZED_BUILDERS[builder_key] = tfds.builder(
          self.name, data_dir=self.data_dir)
    return LazyTfdsLoader._MEMOIZED_BUILDERS[builder_key]

  @property
  def info(self):
    return self.builder.info

  def _map_split(self, split):
    return self._split_map[split] if self._split_map else split

  def files(self, split):
    """Returns set of instructions for reading TFDS files for the dataset."""
    split = self._map_split(split)

    if "/" not in self.name and self.builder.BUILDER_CONFIGS:
      # If builder has multiple configs, and no particular config was
      # requested, raise an error.
      raise ValueError("Dataset '%s' has multiple configs." % self.name)

    split_info = self.builder.info.splits[split]
    files = split_info.file_instructions

    if not files:
      logging.fatal("No TFRecord files found for dataset: %s", self.name)
    return files

  def load(self, split, shuffle_files, seed=None, shard_info=None):
    """Returns a tf.data.Dataset for the given split."""
    split = self._map_split(split)
    input_context = (
        tf.distribute.InputContext(
            num_input_pipelines=shard_info.num_shards,
            input_pipeline_id=shard_info.index) if shard_info else None)
    return tfds.load(
        self._name,
        split=split,
        data_dir=self.data_dir,
        shuffle_files=shuffle_files,
        download=True,
        try_gcs=True,
        read_config=tfds.ReadConfig(
            shuffle_seed=seed,
            skip_prefetch=True,
            input_context=input_context
        )
    )

  def load_shard(self, file_instruction, shuffle_files=False, seed=None):
    """Returns a dataset for a single shard of the TFDS TFRecord files."""
    # pytype:disable=attribute-error
    ds = self.builder._tfrecords_reader.read_files(  # pylint:disable=protected-access
        [file_instruction],
        read_config=tfds.ReadConfig(shuffle_seed=seed),
        shuffle_files=shuffle_files)
    # pytype:enable=attribute-error
    return ds

  def size(self, split):
    """Returns the number of examples in the split."""
    split = self._map_split(split)
    ds_splits = self.info.splits
    dataset_size = ds_splits[split].num_examples
    # Very large datasets have num_examples = 0; default instead to np.inf
    dataset_size = dataset_size if dataset_size > 0 else np.inf
    return dataset_size


def dict_to_tfexample(ex):
  """Convert example dictionary to tf.train.Example proto."""
  feature_dict = {}
  for k, v in ex.items():
    t = tf.constant(v)
    if len(t.shape) == 0:  # pylint:disable=g-explicit-length-test
      v = [v]
    elif len(t.shape) == 1:
      v = list(v)
    else:
      raise ValueError(
          "Unsupported shape (%s) for '%s' value: %s" %
          (t.shape, k, v))

    if t.dtype == tf.string and len(t.shape) <= 1:
      feature_dict[k] = tf.train.Feature(
          bytes_list=tf.train.BytesList(
              value=[tf.compat.as_bytes(t) for t in v]))
    elif t.dtype in (tf.int32, tf.int64) and len(t.shape) <= 1:
      feature_dict[k] = tf.train.Feature(
          int64_list=tf.train.Int64List(value=v))
    elif t.dtype in (tf.float32, tf.float64) and len(t.shape) <= 1:
      feature_dict[k] = tf.train.Feature(
          float_list=tf.train.FloatList(value=v))
    else:
      raise ValueError(
          "Unsupported type (%s) and shape (%s) for '%s' value: %s" %
          (t.dtype, t.shape, k, v))

  return tf.train.Example(features=tf.train.Features(feature=feature_dict))


# ================================ Tasks =======================================
def get_info_path(data_dir, split):
  return os.path.join(data_dir, _INFO_FILENAME.format(split=split))


def get_tfrecord_prefix(data_dir, split):
  return os.path.join(data_dir, _TFRECORD_PREFIX.format(split=split))


def get_stats_path(data_dir, split):
  return os.path.join(data_dir, _STATS_FILENAME.format(split=split))


def print_dataset(dataset):
  """tf.Print dataset fields for debugging purposes."""
  def my_fn(x):
    return {k: tf.Print(v, [v], k + ": ") for k, v in x.items()}
  return dataset.map(my_fn)


def stateless_shuffle(value, seed):
  """Randomly shuffles a tensor, statelessly."""
  flat_value = tf.reshape(value, [-1])
  indices = tf.argsort(
      tf.random.stateless_uniform(tf.shape(flat_value), seed=seed)
  )
  flat_shuffle = tf.gather(flat_value, indices)
  return tf.reshape(flat_shuffle, tf.shape(value))


def trim_and_pad_dataset(
    dataset: tf.data.Dataset,
    feature_lengths: Mapping[str, int]
) -> tf.data.Dataset:
  """Trim and pad first dimension of features to `feature_lengths`.

  Args:
    dataset: tf.data.Dataset, the dataset to trimp/pad examples in.
    feature_lengths: map from feature key to final length. Other features will
      be returned unchanged.
  Returns:
    Trimmed/padded tf.data.Dataset.
  """
  def _trim_and_pad(k: str, t: tf.Tensor) -> tf.Tensor:
    """Trim/pad to the first axis of `t` to be of size `length`."""
    if k not in feature_lengths:
      return t
    length_k = feature_lengths[k]
    t = t[:length_k]
    pad_amt = length_k - tf.shape(t)[0]
    padded_t = tf.pad(t, [(0, pad_amt)] + [(0, 0)] * (len(t.shape) - 1))
    padded_t.set_shape([length_k] + t.shape.as_list()[1:])
    return padded_t

  return dataset.map(
      lambda x: {k: _trim_and_pad(k, t) for k, t in x.items()},
      num_parallel_calls=tf.data.experimental.AUTOTUNE)


def _strip_packed_feature_key(key: str) -> str:
  strip_suffix = lambda k, s: k[:-len(s)] if k.endswith(s) else k
  return strip_suffix(strip_suffix(key, "_positions"), "_segment_ids")


def trim_and_pack_dataset(
    dataset: tf.data.Dataset,
    feature_lengths: Mapping[str, int],
    use_custom_ops: bool = False
) -> tf.data.Dataset:
  """Creates a 'packed' version of a dataset on-the-fly.

  Modified from the tensor2tensor library.

  This is meant to replace the irritation of having to create a separate
  "packed" version of a dataset to train efficiently on TPU.

  Each example in the output dataset represents several examples in the
  input dataset.

  For each key in the input dataset that also exists in `feature_lengths`, two
  additional keys are created:
    <key>_segment_ids: an int32 tensor identifying the parts
       representing the original example.
    <key>_positions: an int32 tensor identifying the position within the
       original example.

  Features that are not in `feature_lengths` will be removed.

  Example:
    Two input examples get combined to form an output example.
    The input examples are:
    {"inputs": [8, 7, 1, 0], "targets":[4, 1, 0], "idx": 0}
    {"inputs": [2, 3, 4, 1], "targets":[5, 6, 1], "idx": 1}
    The output example is:
    {
                   "inputs": [8, 7, 1, 2, 3, 4, 1, 0, 0, 0]
       "inputs_segment_ids": [1, 1, 1, 2, 2, 2, 2, 0, 0, 0]
         "inputs_positions": [0, 1, 2, 0, 1, 2, 3, 0, 0, 0]
                  "targets": [4, 1, 5, 6, 1, 0, 0, 0, 0, 0]
      "targets_segment_ids": [1, 1, 2, 2, 2, 0, 0, 0, 0, 0]
        "targets_positions": [0, 1, 0, 1, 2, 0, 0, 0, 0, 0]
    }

    0 represents padding in both the inputs and the outputs.

    Sequences in the incoming examples are truncated to length in
    `feature_lengths`, and the sequences in the output examples all have this
    fixed (padded) length. Features not in `features_length` (i.e, "idx") are
    removed.

  Args:
    dataset: a tf.data.Dataset
    feature_lengths: map from feature key to final length. Other features will
      be discarded.
    use_custom_ops: a boolean - custom ops are faster but require a custom-built
      binary, which is not currently possible on cloud-tpu.

  Returns:
    a tf.data.Dataset
  """
  element_spec = dataset.element_spec
  # Make sure that the dataset contains all keys in `feature_lengths`.
  for k in feature_lengths:
    if k not in element_spec:
      raise ValueError(
          f"Feature '{k}' not found in dataset. Available keys are "
          f"{list(element_spec.keys())}")
    if not element_spec[k].shape.is_compatible_with(tf.TensorShape([None])):
      raise ValueError(
          f"Features to be packed must be one-dimensional. '{k}' is not.'")
  # Warn if there are any additional keys that will be removed.
  additional_keys = set(element_spec) - set(feature_lengths)
  if additional_keys:
    logging.warn(
        "Features not in `features_length` will be removed during packing: %s",
        additional_keys)

  ds = dataset.map(
      lambda x: {k: x[k][:l] for k, l in feature_lengths.items()},
      num_parallel_calls=tf.data.experimental.AUTOTUNE)

  # Setting batch_size=length ensures that the concatenated sequences (if they
  # have length >=1) are sufficient to fill at least one packed example.
  batch_size = max(feature_lengths.values())
  ds = ds.padded_batch(
      batch_size, padded_shapes={k: [-1] for k in feature_lengths})

  if use_custom_ops and len(feature_lengths) <= 2:
    ds = _pack_with_custom_ops(ds, feature_lengths)
  else:
    ds = _pack_with_tf_ops(ds, feature_lengths)

  # Set the Tensor shapes correctly since they get lost in the process.
  def _set_shape(x):
    for k, v in x.items():
      v.set_shape([feature_lengths[_strip_packed_feature_key(k)]])
    return x
  return ds.map(_set_shape, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def _pack_with_tf_ops(
    dataset: tf.data.Dataset,
    feature_lengths: Mapping[str, int]
) -> tf.data.Dataset:
  """Helper-function for packing a dataset which has already been batched.

  See trim_and_pack_dataset()

  Uses tf.while_loop. Slow.

  Args:
    dataset: a dataset containing padded batches of examples.
    feature_lengths: mapping from feature key to packed length.

  Returns:
    a dataset.
  """
  empty_example = {}
  for k in feature_lengths:
    for suff in ("", "_positions"):
      empty_example[k + suff] = tf.zeros([0], dtype=tf.int32)
      empty_example[k + suff].set_shape([None])
  keys_etc = empty_example.keys()

  def _write_packed_example(partial, outputs):
    new_partial = empty_example.copy()
    new_outputs = {}
    for k in keys_etc:
      new_outputs[k] = outputs[k].write(
          outputs[k].size(),
          tf.pad(partial[k],
                 [[0, feature_lengths[_strip_packed_feature_key(k)] -
                   tf.size(partial[k])]]))
    return new_partial, new_outputs

  def pack_batch(x: Mapping[str, tf.Tensor]) -> Mapping[str, tf.Tensor]:
    """Internal function to map over.

    Consumes a batch of input examples and produces a variable number of output
    examples.

    Args:
      x: a single example
    Returns:
      a tf.data.Dataset
    """
    keys = list(feature_lengths)
    partial = empty_example.copy()
    first_key, *_ = keys
    dynamic_batch_size = tf.shape(x[first_key])[0]
    outputs = {}
    for k in keys:
      outputs[k] = tf.TensorArray(
          tf.int32, size=0, dynamic_size=True,
          element_shape=[feature_lengths[k]])
      outputs[k + "_positions"] = tf.TensorArray(
          tf.int32, size=0, dynamic_size=True,
          element_shape=[feature_lengths[k]])

    for i in tf.range(0, dynamic_batch_size):
      tf.autograph.experimental.set_loop_options(
          shape_invariants=[
              (partial, {k: tf.TensorShape([None]) for k in keys_etc}),
              (outputs, {k: tf.TensorShape(None) for k in keys_etc})]
      )

      can_append = True
      one_example = {}
      for k in keys:
        val = tf.cast(x[k][i], tf.int32)
        val = val[:tf.reduce_sum(tf.cast(tf.not_equal(val, 0), tf.int32))]
        one_example[k] = val
      for k in keys:
        can_append = tf.logical_and(
            can_append,
            tf.less_equal(
                tf.size(partial[k]) + tf.size(one_example[k]),
                feature_lengths[k]))

      if not can_append:
        partial, outputs = _write_packed_example(partial, outputs)

      new_partial = {}
      for k in keys:
        new_seq = one_example[k][:feature_lengths[k]]
        new_seq_len = tf.size(new_seq)
        new_partial[k] = tf.concat([partial[k], new_seq], 0)
        new_partial[k + "_positions"] = tf.concat(
            [partial[k + "_positions"],
             tf.range(new_seq_len, dtype=tf.int32)], 0)
      partial = new_partial

    partial, outputs = _write_packed_example(partial, outputs)
    packed = {k: outputs[k].stack() for k in keys_etc}
    for k in keys:
      packed[k + "_segment_ids"] = (
          tf.cumsum(
              tf.cast(tf.equal(packed[k + "_positions"], 0), tf.int32), axis=1)
          * tf.cast(tf.not_equal(packed[k], 0), tf.int32))
    return packed
  dataset = dataset.map(
      pack_batch, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  return dataset.unbatch()


def _pack_with_custom_ops(
    dataset: tf.data.Dataset,
    feature_lengths: Mapping[str, int]
) -> tf.data.Dataset:
  """Helper-function for packing a dataset which has already been batched.

  See trim_and_pack_dataset()

  Relies on custom ops which require a custom compiled binary.
  Faster than _pack_with_tf_ops(), and denser packing.

  Args:
    dataset: a dataset containing padded batches of examples.
    feature_lengths: mapping from feature key to packed length.

  Returns:
    a dataset.
  """
  # TODO(adarob): Move ops into this library and fix int64 issue.
  from tensor2tensor.data_generators.ops import pack_sequences_ops  # pylint: disable=g-import-not-at-top
  keys = list(feature_lengths)
  if len(keys) == 1:
    k1, = keys
    k2 = k1
  elif len(keys) == 2:
    k1, k2 = keys
  else:
    raise ValueError(f"Packing op requires 1 or 2 keys. Got {len(keys)}")

  def custom_pack_batch(x):
    """Map-function."""
    (k1_packed, k1_segment_ids, k1_positions,
     k2_packed, k2_segment_ids, k2_positions) = (
         pack_sequences_ops.pack_sequences2(
             # cast to int64 for compatibility with custom ops
             tf.cast(x[k1], tf.int64),
             tf.cast(x[k2], tf.int64),
             feature_lengths[k1],
             feature_lengths[k2]))
    packed = {
        k1: k1_packed,
        k1 + "_segment_ids": k1_segment_ids,
        k1 + "_positions": k1_positions,
    }
    if len(keys) == 2:
      packed.update({
          k2: k2_packed,
          k2 + "_segment_ids": k2_segment_ids,
          k2 + "_positions": k2_positions,
      })

    # cast back to int32
    for k, v in packed.items():
      packed[k] = tf.cast(v, tf.int32)

    return packed
  dataset = dataset.map(
      custom_pack_batch, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  dataset = dataset.unbatch()
  return dataset


# ========================= Mixing Rate Functions ==============================


@gin.configurable
def rate_num_examples(
    task, maximum=None, temperature=1.0, scale=1.0,
    fallback_to_num_input_examples=True):
  """Mixing rate equal to the number of examples for the task."""

  if task.cache_dir or not fallback_to_num_input_examples:
    ret = task.get_cached_stats("train")["examples"]
  else:
    logging.warning(
        "Task '%s' not cached so using number of input examples instead of "
        "preprocessed examples to compute rate.",
        task.name)
    ret = task.num_input_examples("train")

  ret *= scale
  if maximum:
    ret = min(ret, maximum)
  if temperature != 1.0:
    ret = ret ** (1.0 / temperature)
  return ret


@gin.configurable
def rate_unsupervised(task, value=1e6):
  """Gin-configurable mixing rate for the unsupervised co-training task."""
  del task
  return value


# ======================== Decorators =========================================


_NEXT_MAP_SEED = None


@contextlib.contextmanager
def map_seed_manager(initial_seed=None):
  """Contextmanager to control the initial seed used by `map_over_dataset`."""
  global _NEXT_MAP_SEED
  old_map_seed = _NEXT_MAP_SEED
  _NEXT_MAP_SEED = initial_seed
  yield
  _NEXT_MAP_SEED = old_map_seed


def map_over_dataset(fn=None, *, num_seeds=None):
  """Decorator to map decorated function over dataset.

  Many preprocessors map a function over a dataset. This decorator helps reduce
  boilerplate for this common pattern.

  If `num_seeds` is set to 1, a unique random seed (pair of int32) will be
  passed to the mapping function with keyword 'seed'.
  If `num_seeds` is greater than 1, unique random seeds (pairs of int32) will be
  passed to the mapping function with keyword 'seeds'.
  These seeds can be generated deterministically by using the `map_seed_manager`
  to set the seed for the process that generates the individual seeds for each
  mapping function. These seeds will be set sequentially from the initial seed
  for each call to `map_over_dataset` where `num_seeds > 0`.

  Args:
    fn: map function
    num_seeds: optional number of random seeds (pairs of int32) to pass to the
      mapping function.

  Returns:
    Function which takes dataset as first argument.
  """

  def map_without_seeds(fn):
    @functools.wraps(fn)
    def wrapped_fn(ds, *args, **kargs):
      return ds.map(
          lambda arg: fn(arg, *args, **kargs),
          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return wrapped_fn

  def map_with_seeds(fn):
    @functools.wraps(fn)
    def wrapped_fn(ds, *args, **kwargs):
      global _NEXT_MAP_SEED
      if _NEXT_MAP_SEED is None:
        random_ds_seeds = ((None, None),) * num_seeds
      else:
        random_ds_seeds = np.arange(
            _NEXT_MAP_SEED, _NEXT_MAP_SEED + 2 * num_seeds).reshape(-1, 2)
        random_ds_seeds = tuple(tuple(s) for s in random_ds_seeds)
        _NEXT_MAP_SEED += 2 * num_seeds
      seed_datasets = tf.nest.map_structure(
          tf.data.experimental.RandomDataset,
          random_ds_seeds)
      if num_seeds == 1:
        map_fn = lambda x, s: fn(x, seed=s[0], *args, **kwargs)
      else:
        map_fn = lambda x, s: fn(x, seeds=s, *args, **kwargs)
      return tf.data.Dataset.zip((ds, seed_datasets)).map(
          map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Remove seeds from function signature.
    sig = inspect.signature(wrapped_fn)
    wrapped_fn.__signature__ = sig.replace(
        parameters=tuple(
            p for p in sig.parameters.values() if p.name not in("seed", "seeds")
        )
    )
    return wrapped_fn

  if fn is None:
    return map_with_seeds
  else:
    return map_without_seeds(fn)
