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

import os

from absl import logging
import gin
import numpy as np
from t5.data import sentencepiece_vocabulary
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds

_INFO_FILENAME = "info.{split}.json"
_STATS_FILENAME = "stats.{split}.json"
_TFRECORD_PREFIX = "{split}.tfrecord"

_TFDS_DATA_DIR_OVERRIDE = None
_GLOBAL_CACHE_DIRECTORIES = []

DEFAULT_SPM_PATH = "gs://t5-data/vocabs/cc_all.32000/sentencepiece.model"  # GCS
DEFAULT_EXTRA_IDS = 100


def get_default_vocabulary():
  return sentencepiece_vocabulary.SentencePieceVocabulary(
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

  def load(self, split, shuffle_files, seed=None):
    """Returns a tf.data.Dataset for the given split."""
    split = self._map_split(split)
    return tfds.load(
        self._name,
        split=split,
        data_dir=self.data_dir,
        shuffle_files=shuffle_files,
        download=True,
        try_gcs=True,
        read_config=tfds.ReadConfig(shuffle_seed=seed))

  def load_shard(self, file_instruction, seed=None):
    """Returns a dataset for a single shard of the TFDS TFRecord files."""
    ds = self.builder._tfrecords_reader.read_files(  # pylint:disable=protected-access
        [file_instruction],
        read_config=tfds.ReadConfig(shuffle_seed=seed),
        shuffle_files=False)
    return ds

  def size(self, split):
    """Returns the number of examples in the split."""
    split = self._map_split(split)
    ds_splits = self.info.splits
    dataset_size = ds_splits[split].num_examples
    # Very large datasets have num_examples = 0; default instead to np.inf
    dataset_size = dataset_size if dataset_size > 0 else np.inf
    return dataset_size


def encode_string_features(
    dataset, output_features, keys, copy_plaintext=False):
  """Encode specified string features.

  Passes through non-string features unchanged. Optionally passes through copy
  of original string features with "_plaintext" suffix added to the key.

  Args:
    dataset: a tf.data.Dataset
    output_features: a dict of Feature objects; their vocabulary attribute will
      be used to tokenize the specified features.
    keys: list of strings, keys of features to encode.
    copy_plaintext: bool, whether to pass through copies of plaintext strings
      with a "_plaintext" suffix added to the key.
  Returns:
    a tf.data.Dataset
  """
  keys = set(keys)
  def my_fn(features):
    """Encode all specified feature that are strings and return a dictionary.

    Args:
      features: a dictionary
    Returns:
      a dictionary
    """
    ret = {}
    for k, v in features.items():
      if k in keys and v.dtype == tf.string:
        if copy_plaintext:
          ret["%s_plaintext" % k] = v
        v = tf.cast(output_features[k].vocabulary.encode_tf(v), tf.int64)
      ret[k] = v
    return ret
  return dataset.map(my_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)


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

