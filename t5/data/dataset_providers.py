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
"""Classes for data loading and processing.

Defines Tasks, TaskRegistry, Mixture, and MixtureRegistry
"""

import abc
import collections
import inspect
import json
import os
import re
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Tuple, Type, Union

from absl import logging
import gin
from t5.data import utils
from t5.data.preprocessors import tokenize as tokenize_preprocessor
from t5.data.vocabularies import Vocabulary
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


_DEFAULT_FEATURE_KEYS = ["inputs", "targets"]

_VALID_TASK_NAME_REGEX = re.compile(r"^[\w\d\._]+$")
_MAX_EXAMPLES_TO_MEM_CACHE = 10000
SHUFFLE_BUFFER_SIZE = 1000


class Feature(object):
  """A container for attributes of output features of data providers."""

  def __init__(
      self,
      vocabulary: Union[Callable[[], Vocabulary], Vocabulary],
      add_eos: bool = True,
      required: bool = True):
    """Create a Feature instance.

    Args:
      vocabulary: vocabularies.Vocabulary object to use for tokenization, or a
        callable function returning a vocabulary
      add_eos: bool, whether an EOS token should be added to this Feature.
      required: Whether or not this feature must exist in the final outputs of
        the Task.
    """
    self._vocabulary = vocabulary
    self.add_eos = add_eos
    self.required = required

  @property
  def vocabulary(self) -> Vocabulary:
    if callable(self._vocabulary):
      self._vocabulary = self._vocabulary()
    return self._vocabulary


class DatasetProviderBase(object):
  """Abstract base for classes that provide a tf.data.Dataset."""

  __metaclass__ = abc.ABCMeta

  @abc.abstractproperty
  def output_features(self) -> Mapping[str, Feature]:
    raise NotImplementedError

  @abc.abstractproperty
  def splits(self) -> Iterable[str]:
    raise NotImplementedError

  @abc.abstractmethod
  def get_dataset(
      self,
      sequence_length: int,
      split: str,
      use_cached: bool = False,
      shuffle: bool = True,
      seed: Optional[int] = None
  ) -> tf.data.Dataset:
    raise NotImplementedError

  @abc.abstractmethod
  def num_input_examples(self, split: str) -> int:
    raise NotImplementedError


class DatasetProviderRegistry(object):
  """Base for registry of data providers.

  Child classes must implement a _REGISTRY dict.
  """

  _PROVIDER_TYPE = DatasetProviderBase

  @classmethod
  def add(cls, name, provider_cls, *provider_args, **provider_kwargs):
    """Adds provider to the registry."""
    if name in cls._REGISTRY:
      raise ValueError("Attempting to register duplicate provider: %s" % name)
    provider = provider_cls(*provider_args, **provider_kwargs)
    if not isinstance(provider, cls._PROVIDER_TYPE):
      raise ValueError(
          "Attempting to register a class not of an invalid type. "
          "Expecting instance of %s, got %s" %
          (cls._PROVIDER_TYPE, provider_cls))

    cls._REGISTRY[name] = provider

  @classmethod
  def remove(cls, name):
    """Remove provider from the registry, if it exists."""
    if name in cls._REGISTRY:
      del cls._REGISTRY[name]

  @classmethod
  def get(cls, name):
    """Returns provider from the registry."""
    if name not in cls._REGISTRY:
      raise ValueError("Provider name not registered: %s" % name)
    return cls._REGISTRY[name]

  @classmethod
  def names(cls):
    """Returns all provider names in registry."""
    return cls._REGISTRY.keys()

  @classmethod
  def get_dataset(
      cls,
      name,
      sequence_length,
      split,
      use_cached=False,
      shuffle=True,
      seed=None):
    return cls.get(name).get_dataset(
        sequence_length=sequence_length, split=split, use_cached=use_cached,
        shuffle=shuffle, seed=seed)


# =============================== DataSources ==================================


class DataSource(DatasetProviderBase):
  """A `DatasetProvider` that provides raw data from an input source.

  Inherits all abstract methods and properties of `DatasetProviderBase` except
  those overidden below.
  """

  __metaclass__ = abc.ABCMeta

  def __init__(
      self,
      splits: Iterable[str],
      num_input_examples: Optional[Mapping[str, int]] = None):
    self._splits = tuple(splits)
    self._num_input_examples = (
        dict(num_input_examples) if num_input_examples is not None else None)

  @property
  def splits(self) -> Iterable[str]:
    return self._splits

  @property
  def output_features(self) -> Mapping[str, Feature]:
    """Override unused property of `DatasetProviderBase`."""
    raise NotImplementedError

  @abc.abstractmethod
  def list_shards(self, split: str) -> Iterable[str]:
    """Returns string identifiers of input shards."""
    raise NotImplementedError

  @abc.abstractmethod
  def get_dataset(
      self,
      split: str,
      shuffle: bool = True,
      seed: Optional[int] = None,
      shard: Optional[str] = None
    ) -> tf.data.Dataset:
    """Overrides base class to add shard identifier and remove use_cached.

    Args:
      split: string, the split to return.
      shuffle: bool, whether to shuffle the input source.
      seed: tf.int64 scalar tf.Tensor (or None) for shuffling input source.
      shard: string, optional identifier for loading a single shard of the
        split.
    """
    raise NotImplementedError

  def num_input_examples(self, split: str) -> int:
    if self._num_input_examples is None:
      return None
    return self._num_input_examples[split]


class FunctionSource(DataSource):
  """A `DataSource` that uses a function to provide the input data."""

  def __init__(
      self,
      dataset_fn: Callable[[str, bool, Optional[int]], tf.data.Dataset],
      splits: Iterable[str],
      num_input_examples: Optional[Mapping[str, int]] = None
  ):
    """FunctionSource constructor.

    Args:
      dataset_fn: a function with the signature `dataset_fn(split,
        shuffle_files)' (and optionally the variable `seed`) that returns a
        `tf.data.Dataset`.
      splits: an iterable of applicable string split names.
      num_input_examples: dict or None, an optional dictionary mapping split
          to its size in number of input examples (before preprocessing). The
          `num_input_examples` method will return None if not provided.
    """
    _validate_args(dataset_fn, ["split", "shuffle_files"])
    self._dataset_fn = dataset_fn
    super().__init__(splits=splits, num_input_examples=num_input_examples)

  def get_dataset(
      self,
      split: str,
      shuffle: bool = True,
      seed: Optional[int] = None,
      shard: Optional[str] = None
  ) -> tf.data.Dataset:
    if seed is None:
      ds = self._dataset_fn(split=split, shuffle_files=shuffle)
    else:
      _validate_args(self._dataset_fn, ["split", "shuffle_files", "seed"])
      ds = self._dataset_fn(split=split, shuffle_files=shuffle, seed=seed)
    return ds

  def list_shards(self, split: str) -> Iterable[str]:
    return [split]


class TfdsDataSource(DataSource):
  """A `DataSource` that uses TensorFlow Datasets to provide the input data."""

  def __init__(
      self,
      tfds_name: str,
      tfds_data_dir: Optional[str] = None,
      splits: Optional[Union[Iterable[str], Mapping[str, str]]] = None
    ):
    """TfdsTask constructor.

    Args:
      tfds_name: string, the name and version number of a TFDS dataset,
        optionally with a config.
      tfds_data_dir: string, an optional path to a specific TFDS data directory
        to use.
      splits: an iterable of allowable string split names, a dict mapping
        allowable canonical splits (e.g., 'validation') to TFDS splits or slices
        (e.g., 'train[':1%']), or None. The default, None, uses all available
          splits from the TFDS dataset info.
    """
    if ":" not in tfds_name:
      raise ValueError("TFDS name must contain a version number, got: %s" %
                       tfds_name)

    self._tfds_dataset = utils.LazyTfdsLoader(
        tfds_name,
        data_dir=tfds_data_dir,
        split_map=splits if isinstance(splits, dict) else None)

    # If splits are not provided, we pass an empty tuple and use the lazy
    # lookup in the `splits` property.
    super().__init__(splits=splits or ())

  @property
  def splits(self):
    """Overrides since we can't call `info.splits` until after init."""
    return self._splits or self._tfds_dataset.info.splits

  @property
  def tfds_dataset(self):
    return self._tfds_dataset

  def get_dataset(
      self,
      split: str,
      shuffle: bool = True,
      seed: Optional[int] = None,
      shard: Optional[str] = None
  ) -> tf.data.Dataset:
    if shard:
      return self.tfds_dataset.load_shard(
          shard, shuffle_files=shuffle, seed=seed)
    return self.tfds_dataset.load(split, shuffle_files=shuffle, seed=seed)

  def num_input_examples(self, split: str) -> int:
    """Overrides since we can't call `info.splits` until after init."""
    return self.tfds_dataset.size(split)

  def list_shards(self, split: str) -> Iterable[str]:
    return self.tfds_dataset.files(split)


class FileDataSource(DataSource):
  """A `DataSource` that reads a file to provide the input dataset."""

  def __init__(
      self,
      read_file_fn: Callable[[tf.data.Dataset], tf.data.Dataset],
      split_to_filepattern: Mapping[str, Union[str, Iterable[str]]],
      num_input_examples: Optional[Mapping[str, int]] = None,
  ):
    """FileDataSource constructor.

    Args:
      read_file_fn: a callable for creating a `tf.data.Dataset` from a
        `tf.data.Dataset` of file paths, e.g., `tf.data.TFRecordDataset`.
      split_to_filepattern: a mapping from split names to filepatterns to be
        expanded with glob.
      num_input_examples: dict or None, an optional dictionary mapping split
        to its size in number of input examples (before preprocessing). The
        `num_input_examples` method will return None if not provided.
    """
    self._split_to_filepattern = split_to_filepattern
    self._reader = read_file_fn
    super().__init__(
        splits=split_to_filepattern.keys(),
        num_input_examples=num_input_examples)

  def get_dataset(
      self,
      split: str,
      shuffle: bool = True,
      seed: Optional[int] = None,
      shard: Optional[str] = None
    ) -> tf.data.Dataset:
    filepattern = shard or self._split_to_filepattern[split]

    files = tf.data.Dataset.list_files(filepattern, shuffle=shuffle, seed=seed)

    return files.interleave(
        self._reader,
        cycle_length=16,
        block_length=16,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

  def list_shards(self, split: str) -> Iterable[str]:
    return tf.io.gfile.glob(self._split_to_filepattern[split])


class TextLineDataSource(FileDataSource):
  """A `FileDataSource` that reads lines of text from a file as input."""

  def __init__(
      self,
      split_to_filepattern: Mapping[str, Union[str, Iterable[str]]],
      skip_header_lines: int = 0,
      num_input_examples: Optional[Mapping[str, int]] = None,
  ):
    """TextLineDataSource constructor.

    Args:
      split_to_filepattern: a mapping from split names to filepatterns to be
        expanded with glob.
      skip_header_lines: int, number of header lines to skip in each source
        file.
      num_input_examples: dict or None, an optional dictionary mapping split to
        its size in number of input examples (before preprocessing). The
        `num_input_examples` method will return None if not provided.
    """
    # Used during caching.
    self._skip_header_lines = skip_header_lines

    def read_file_fn(filepattern):
      return tf.data.TextLineDataset(filepattern).skip(skip_header_lines)

    super().__init__(
        read_file_fn=read_file_fn,
        split_to_filepattern=split_to_filepattern,
        num_input_examples=num_input_examples)


class TFExampleDataSource(FileDataSource):
  """A `FileDataSource` that reads files of tf.train.Example protos as input."""

  def __init__(
      self,
      split_to_filepattern: Mapping[str, Union[str, Iterable[str]]],
      feature_description: Mapping[str, Union[tf.io.FixedLenFeature,
                                              tf.io.VarLenFeature]],
      reader_cls: Type[tf.data.Dataset] = tf.data.TFRecordDataset,
      num_input_examples: Optional[Mapping[str, int]] = None,
  ):
    """TFExampleDataSource constructor.

    Args:
      split_to_filepattern: dict of string (split name) to either string
        (filename or filepattern) or list of strings (filenames or
        filepatterns).
      feature_description: dict, a mapping of string feature keys to
        `tf.io.FixedLenFeature` or `tf.io.VarLenFeature` values.
      reader_cls: `tf.data.Dataset`, a dataset class to read the input files.
      num_input_examples: dict or None, an optional dictionary mapping split to
        its size in number of input examples (before preprocessing). The
        `num_input_examples` method will return None if not provided.
    """

    self._feature_description = feature_description

    super().__init__(
        read_file_fn=reader_cls,
        split_to_filepattern=split_to_filepattern,
        num_input_examples=num_input_examples)

  def get_dataset(
      self,
      split: str,
      shuffle: bool = True,
      seed: Optional[int] = None,
      shard: Optional[str] = None
    ) -> tf.data.Dataset:
    """Overrides to parse tf.train.Example proto after reading file."""
    ds = super().get_dataset(
        split=split, shuffle=shuffle, seed=seed, shard=shard)
    return ds.map(
        lambda pb: tf.io.parse_single_example(pb, self._feature_description),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)


# ================================ Tasks =======================================


def print_dataset(dataset):
  """tf.Print dataset fields for debugging purposes."""
  def my_fn(x):
    return {k: tf.print(v, [v], k + ": ") for k, v in x.items()}
  return dataset.map(my_fn)


@gin.configurable
def maybe_print_dataset(dataset, should_print=False):
  """tf.Print dataset for debugging purposes."""
  return print_dataset(dataset) if should_print else dataset


class CacheDatasetPlaceholder(object):
  """A placeholder to signal when in the pipeline offline caching will occur."""

  def __call__(self, dataset):
    raise RuntimeError("`CacheDatasetPlaceholder` should never be called.")


class TaskV3(DatasetProviderBase):
  """A class to manage a dataset and its related metrics.

  The V3 API is still under development and may change without warning.
  """

  def __init__(
      self,
      name: str,
      source: DataSource,
      output_features: Mapping[str, Feature],
      preprocessors: Optional[Sequence[Callable[..., tf.data.Dataset]]] = None,
      postprocess_fn: Optional[Callable[..., Any]] = None,
      metric_fns: Optional[Callable[..., Mapping[str, float]]] = None,
      shuffle_buffer_size: Optional[int] = SHUFFLE_BUFFER_SIZE):
    """Task V3 constructor.

    Args:
      name: a unique name for the Task.
      source: a `DataSource` that provides a raw `tf.data.Dataset`.
      output_features: dict(str, Feature), output features of the Task to be
        passed to the model. After preprocessing, examples will be validated to
        ensure they include features that match this specification. Note that
        additional features may be included (e.g., for evaluation), but they
        will not be passed to the model.
      preprocessors: list(callable), an optional list of functions that receive
        a tf.data.Dataset and return a tf.data.Dataset. These will be executed
        sequentually and the final dataset must include features matching
        `output_features`.
      postprocess_fn: callable, an optional function that receives decoded model
        outputs and converts them to a form that is ready for evaluation using
        the metric functions in `metric_fns`.
      metric_fns: list(callable), an optional list of metric functions with the
        signature `metric_fn(targets, predictions)` to use during evaluation. If
        undefined or empty, no evaluation will occur on the task.
      shuffle_buffer_size: an optional integer
    """
    if not _VALID_TASK_NAME_REGEX.match(name):
      raise ValueError(
          "Task name '%s' contains invalid characters. Must match regex: %s" % (
              name, _VALID_TASK_NAME_REGEX.pattern))
    metric_fns = metric_fns or []
    for metric_fn in metric_fns:
      _validate_args(metric_fn, ["targets", "predictions"])

    self._name = name
    self._source = source

    # Find optional CacheDatasetPlaceholder.
    preprocessors = tuple(preprocessors or [])
    cache_step_idxs = [
        i for i, p in enumerate(preprocessors)
        if isinstance(p, CacheDatasetPlaceholder)
    ]
    if len(cache_step_idxs) > 1:
      raise ValueError(
          "`CacheDatasetPlaceholder` can appear at most once in the "
          f"preprocessing pipeline. Found {len(cache_step_idxs)} in '{name}'.")
    cache_step_idx = cache_step_idxs[0] if cache_step_idxs else -1
    if cache_step_idx > -1:
      for prep in preprocessors[:cache_step_idx]:
        if "sequence_length" in inspect.signature(prep).parameters.keys():
          raise ValueError(
              f"'{prep.__name__}' has a `sequence_length` argument but occurs "
              f"before `CacheDatasetPlaceholder` in '{name}'. This is not "
              "allowed since the sequence length is specified at run time.")
    self._cache_step_idx = cache_step_idx
    self._preprocessors = preprocessors

    self._metric_fns = tuple(metric_fns)
    self._postprocess_fn = postprocess_fn

    self._cache_dir = None
    self._stats = {}
    self._shuffle_buffer_size = shuffle_buffer_size

    self._output_features = collections.OrderedDict(
        sorted(list(output_features.items()))
    )

  @property
  def name(self) -> str:
    return self._name

  @property
  def metric_fns(
      self) -> Optional[Iterable[Callable[..., Mapping[str, float]]]]:
    return self._metric_fns

  @property
  def output_features(self) -> Mapping[str, Feature]:
    return self._output_features

  @property
  def splits(self) -> Iterable[str]:
    return self.source.splits

  @property
  def source(self) -> DataSource:
    return self._source

  def num_input_examples(self, split: str) -> int:
    return self.source.num_input_examples(split)

  def _preprocess_dataset(
      self,
      dataset: tf.data.Dataset,
      preprocessors: Iterable[Callable[..., tf.data.Dataset]],
      sequence_length: Optional[Mapping[str, int]] = None) -> tf.data.Dataset:
    """Sequentially applies preprocessors."""
    for prep_fn in preprocessors:
      # prep_fn must not rely on variable length keyword args such as **kwargs.
      fn_args = set(inspect.signature(prep_fn).parameters.keys())
      kwargs = {}
      if "sequence_length" in fn_args:
        assert sequence_length is not None
        kwargs["sequence_length"] = sequence_length
      if "output_features" in fn_args:
        kwargs["output_features"] = self.output_features
      dataset = prep_fn(dataset, **kwargs)
    return dataset

  def _validate_dataset(self,
                        dataset: tf.data.Dataset,
                        expected_output_type: tf.DType,
                        expected_output_rank: int,
                        error_label: str,
                        ensure_no_eos: bool = False) -> tf.data.Dataset:
    """Validates properties of a tf.data.Dataset, raising Exceptions if needed.

    Args:
      dataset: a tf.data.Dataset to validate.
      expected_output_type: a tf.Dtype, the expected type of the model features.
      expected_output_rank: an int, the expected rank of the model features.
      error_label: a string, an identifier for the previous processing step to
        report in raised ValueErrors.
      ensure_no_eos: a bool, whether or not to verify that the model features
        contain no EOS tokens.

    Returns:
      a validated tf.data.Dataset.
    """
    element_spec = dataset.element_spec
    for feat in self.output_features:
      if feat not in element_spec:
        if self.output_features[feat].required:
          raise ValueError(
              "Task dataset is missing expected output feature after {label}: "
              "{feat}".format(label=error_label, feat=feat))
        else:
          # It's ok that this feature does not exist.
          continue
      if expected_output_type != element_spec[feat].dtype:
        raise ValueError(
            "Task dataset has incorrect type for feature '{feat}' after "
            "{label}: Got {actual}, expected {expected}".format(
                feat=feat, label=error_label,
                actual=element_spec[feat].dtype.name,
                expected=expected_output_type.name))
      if expected_output_rank != len(element_spec[feat].shape):
        raise ValueError(
            "Task dataset has incorrect rank for feature '{feat}' after "
            "{label}: Got {actual}, expected {expected}".format(
                feat=feat, label=error_label,
                actual=len(element_spec[feat].shape),
                expected=expected_output_rank))

    def _ensure_no_eos(feat, v):
      if feat not in self.output_features:
        return v
      with tf.control_dependencies([
          tf.debugging.assert_none_equal(
              v, tf.constant(1, tf.int64),
              message="Feature '{feat}' unexpectedly contains EOS=1 token "
              "after {label}.".format(feat=feat, label=error_label))
      ]):
        return v
    if ensure_no_eos:
      dataset = dataset.map(
          lambda ex: {k: _ensure_no_eos(k, v) for k, v in ex.items()},
          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset

  def _trim_and_ensure_eos(
      self,
      dataset: tf.data.Dataset,
      sequence_length: Mapping[str, int]
    ) -> tf.data.Dataset:
    """Trim and append EOS=1 token to model features."""
    def _trim_and_append_eos(feat, v):
      if feat not in self.output_features:
        return v
      if sequence_length and self.output_features[feat].add_eos:
        v = tf.concat([v[:sequence_length[feat]-1], [1]], axis=0)
      elif sequence_length:
        v = v[:sequence_length[feat]]
      elif self.output_features[feat].add_eos:
        v = tf.concat([v, [1]], axis=0)
      v.set_shape([sequence_length[feat]])
      return v

    return dataset.map(
        lambda ex: {k: _trim_and_append_eos(k, v) for k, v in ex.items()},
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

  def preprocess_precache(
      self,
      dataset: tf.data.Dataset,
      seed: Optional[int] = None
    ) -> tf.data.Dataset:
    """Runs preprocessing steps before the optional CacheDatasetPlaceholder."""
    if not self.supports_caching:
      return dataset

    with utils.map_seed_manager(seed):
      return self._preprocess_dataset(
          dataset,
          self._preprocessors[:self._cache_step_idx],
      )

  def preprocess_postcache(
      self,
      dataset: tf.data.Dataset,
      sequence_length: Mapping[str, int],
      seed: Optional[int] = None
    ) -> tf.data.Dataset:
    """Runs preprocessing steps after the optional CacheDatasetPlaceholder.

    Args:
      dataset: a tf.data.Dataset
      sequence_length: dict mapping feature key to int length for that feature.
        If None, the features will not be truncated.
      seed: an optional random seed for deterministic preprocessing.
    Returns:
      a tf.data.Dataset
    """
    # Skip a sufficient number of seeds to avoid duplicating any from pre-cache
    # preprocessing.
    seed = None if seed is None else 42 * self._cache_step_idx
    with utils.map_seed_manager(seed):
      dataset = self._preprocess_dataset(
          dataset,
          self._preprocessors[self._cache_step_idx + 1:],
          sequence_length=sequence_length,
      )
    dataset = self._validate_dataset(
        dataset,
        expected_output_type=tf.int64,
        expected_output_rank=1,
        error_label="preprocessing",
        ensure_no_eos=True)
    return dataset

  @property
  def cache_dir(self) -> Optional[str]:
    """Returns the cache directory (or None), initializing if needed."""
    if not self._cache_dir:
      # See if cached data exists in any of the cache directories.
      potential_cache_dirs = [
          os.path.join(d, self.name) for d in utils.get_global_cache_dirs()]
      for cache_dir in potential_cache_dirs:
        if tf.io.gfile.exists(os.path.join(cache_dir, "COMPLETED")):
          self._cache_dir = cache_dir
          logging.info("'%s' is cached at %s.", self.name, self.cache_dir)
          break

      if not self._cache_dir:
        logging.info(
            "'%s' does not exist in any task cache directories (searched %s).",
            self.name,
            potential_cache_dirs,
        )
    return self._cache_dir

  @property
  def supports_caching(self) -> bool:
    """Wether or not this task supports offline caching."""
    return self._cache_step_idx > -1

  def assert_cached(self) -> None:
    """Raises an assertion error if cached dataset does not exist."""
    assert self.cache_dir, (
        "'%s' does not exist in any of the task cache directories" % self.name)

  def get_cached_stats(self,
                       split: str = tfds.Split.TRAIN
                      ) -> Mapping[str, Union[int, float]]:
    """Returns basic statistics for cached dataset."""
    self.assert_cached()
    if split not in self._stats:
      stats_path = utils.get_stats_path(self.cache_dir, split)
      if not tf.io.gfile.exists(stats_path):
        raise ValueError(
            "Stats do not exist for '%s' split: %s" % (self.name, split))
      with tf.io.gfile.GFile(stats_path) as f:
        self._stats[split] = json.load(f)
    return self._stats[split]

  def get_dataset(
      self,
      sequence_length: Mapping[str, int],
      split: str = tfds.Split.TRAIN,
      use_cached: bool = False,
      shuffle: bool = True,
      shuffle_buffer_size: Optional[int] = None,
      seed: Optional[int] = None,
  ) -> tf.data.Dataset:
    """Returns a tf.data.Dataset from cache or generated on the fly.

    Args:
      sequence_length: dict mapping feature key to int length for that feature
      split: string, the split to return.
      use_cached: bool, whether to use the cached dataset instead of processing
        it on the fly. Defaults to False.
      shuffle: bool, whether to shuffle the dataset. Only used when generating
        on the fly (use_cached=False).
      shuffle_buffer_size: an integer or None to use task-specific buffer size.
      seed: tf.int64 scalar tf.Tensor (or None) for shuffling tf.data.
    Returns:
      A mixed tf.data.Dataset.
    """
    if seed is not None:
      logging.warning(("Global random seed is now set to %d. All TF operations "
                       "are now deterministic with respect to that seed."),
                      seed)
      tf.random.set_seed(seed)

    if use_cached and not self.supports_caching:
      logging.warning(
          "Task '%s' does not support caching. Switching to on-the-fly "
          "preprocessing.", self.name)
      use_cached = False
    if use_cached:
      ds = self._get_cached_dataset(split, shuffle)
    else:
      ds = self.source.get_dataset(split=split, shuffle=shuffle, seed=seed)
      ds = self.preprocess_precache(ds, seed)

    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    if (not use_cached and self.num_input_examples(split) and
        self.num_input_examples(split) < _MAX_EXAMPLES_TO_MEM_CACHE):
      ds = ds.cache()

    # Post tokenization processing.
    ds = self.preprocess_postcache(ds, sequence_length=sequence_length)
    ds = self._trim_and_ensure_eos(ds, sequence_length=sequence_length)
    ds = maybe_print_dataset(ds)

    if shuffle:
      # Shuffle before mixing since preprocessor can output multiple
      # (correlated) examples per input.
      ds = ds.shuffle(shuffle_buffer_size or self._shuffle_buffer_size,
                      seed=seed)

    return ds.prefetch(tf.data.experimental.AUTOTUNE)

  def _get_cached_dataset(self,
                          split: str = tfds.Split.TRAIN,
                          shuffle: bool = True,
                          seed: Optional[int] = None) -> tf.data.Dataset:
    """Returns a tf.data.Dataset read from cached files."""
    self.assert_cached()
    with tf.io.gfile.GFile(utils.get_info_path(self.cache_dir, split)) as f:
      split_info = json.load(f)

    # Use `FixedLenSequenceFeature` for sequences with variable length.
    def _feature_config(shape, dtype):
      if shape and shape[0] is None:
        return tf.io.FixedLenSequenceFeature(
            shape[1:], dtype, allow_missing=True)
      return tf.io.FixedLenFeature(shape, dtype)
    feature_desc = {
        feat: _feature_config(**desc)
        for feat, desc in split_info["features"].items()}

    ds = tf.data.Dataset.list_files(
        "%s-*-of-*%d" % (
            utils.get_tfrecord_prefix(self.cache_dir, split),
            split_info["num_shards"]),
        shuffle=shuffle,
        seed=seed)
    ds = ds.interleave(
        tf.data.TFRecordDataset,
        cycle_length=16, block_length=16,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.map(lambda ex: tf.io.parse_single_example(ex, feature_desc),
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if self.get_cached_stats(split)["examples"] <= _MAX_EXAMPLES_TO_MEM_CACHE:
      ds = ds.cache()
    return ds

  def postprocess_fn(self, decoded_model_output: Any,
                     **postprocess_kwargs) -> Any:
    """Returns the model output after applying the postprocess function."""
    if self._postprocess_fn:
      return self._postprocess_fn(decoded_model_output, **postprocess_kwargs)
    return decoded_model_output


class Task(TaskV3):
  """A wrapper for a `tf.data.Dataset` along with preprocessing information.

  Tasks handle preprocessing (via arbitrary TF function) and tokenization.
  Non-train splits also pass through the original plaintext strings with a
  "_plaintext" suffix added to the key.
  """

  def __init__(self,
               name,
               dataset_fn,
               splits,
               text_preprocessor,
               metric_fns=None,
               postprocess_fn=None,
               token_preprocessor=None,
               output_features=None,
               num_input_examples=None,
               supports_caching=True,
               shuffle_buffer_size=SHUFFLE_BUFFER_SIZE,
               source=None):

    if (dataset_fn, source).count(None) != 1:
      raise ValueError(
          "Exactly one of either `dataset_fn` or `source` must be provided.")

    if source and (splits or num_input_examples):
      raise ValueError(
          "If `source` is provided, `splits` and `num_input_examples` should "
          "not also be provided to the Task.")
    source = source or FunctionSource(
        dataset_fn=dataset_fn,
        splits=splits,
        num_input_examples=num_input_examples)

    if text_preprocessor and not hasattr(text_preprocessor, "__iter__"):
      text_preprocessor = [text_preprocessor]
    if token_preprocessor and not hasattr(token_preprocessor, "__iter__"):
      token_preprocessor = [token_preprocessor]

    preprocessors = list(text_preprocessor or [])
    preprocessors.append(tokenize_preprocessor)
    if supports_caching:
      preprocessors.append(CacheDatasetPlaceholder())
    preprocessors.extend(token_preprocessor or [])

    if hasattr(output_features, "__len__") and not output_features:
      raise ValueError("output_features must be non-empty.")
    if output_features is None:
      output_features = Feature(utils.get_default_vocabulary())
    if isinstance(output_features, dict):
      pass
    elif isinstance(output_features, Feature):
      output_features = {k: output_features for k in _DEFAULT_FEATURE_KEYS}
    elif isinstance(output_features, list) and all(
        isinstance(f, str) for f in output_features):
      output_features = {
          k: Feature(utils.get_default_vocabulary()) for k in output_features
      }
    else:
      raise ValueError(
          "output_features must be a dict, Feature, list of str, or None")

    if hasattr(postprocess_fn, "__iter__"):
      postprocess_fns = postprocess_fn

      def postprocess_fn(x, **postprocess_kwargs):  # pylint:disable=function-redefined
        for post_fn in postprocess_fns:
          x = post_fn(x, **postprocess_kwargs)
        return x

    super().__init__(
        name=name,
        source=source,
        output_features=output_features,
        preprocessors=preprocessors,
        postprocess_fn=postprocess_fn,
        metric_fns=metric_fns,
        shuffle_buffer_size=shuffle_buffer_size)


class TfdsTask(Task):
  """A `Task` that uses TensorFlow Datasets to provide the input dataset."""

  def __init__(
      self,
      name,
      tfds_name,
      text_preprocessor,
      metric_fns,
      tfds_data_dir=None,
      splits=None,
      **task_kwargs):
    """TfdsTask constructor.

    Args:
      name: string, a unique name for the Task. A ValueError will be raised if
        another task with this name is already registered.
      tfds_name: string, the name and version number of a TFDS dataset,
        optionally with a config.
      text_preprocessor: a function (or list of functions) that (each) takes in
        a tf.data.Dataset of string features and returns a tf.data.Dataset of
        string features. Can be set to None as a no-op. If a list is given, they
        will be executed sequentially.
      metric_fns: list(callable), list of metric functions with the signature
        metric_fn(targets, predictions) to use during evaluation.
      tfds_data_dir: string, an optional path to a specific TFDS data directory
        to use.
      splits: a list(string) of allowable splits to load, a dict mapping
        allowable canonical splits (e.g., 'validation') to TFDS splits or slices
        (e.g., 'train[':1%']), or None. The default, None, uses all available
          splits from the TFDS dataset info.
      **task_kwargs: dict, additional keyword arguments for the parent `Task`
        class.
    """
    super().__init__(
        name,
        source=TfdsDataSource(
            tfds_name=tfds_name, tfds_data_dir=tfds_data_dir, splits=splits),
        text_preprocessor=text_preprocessor,
        metric_fns=metric_fns,
        dataset_fn=None,
        splits=None,
        **task_kwargs)


class TextLineTask(Task):
  """A `Task` that reads text lines as input.

  Requires a text_processor to be passed that takes a tf.data.Dataset of
  strings and returns a tf.data.Dataset of feature dictionaries.
  e.g. preprocessors.preprocess_tsv()
  """

  def __init__(
      self,
      name,
      split_to_filepattern,
      text_preprocessor,
      metric_fns,
      skip_header_lines=0,
      **task_kwargs):
    """TextLineTask constructor.

    Args:
      name: string, a unique name for the Task. A ValueError will be raised if
        another task with this name is already registered.
      split_to_filepattern: dict of string (split name) to string (filename or
        filepattern).
      text_preprocessor: a function (or list of functions) that (each) takes in
        a tf.data.Dataset of string features and returns a tf.data.Dataset of
        string features. Can be set to None as a no-op. If a list is given, they
        will be executed sequentially.
      metric_fns: list(callable), list of metric functions with the signature
        metric_fn(targets, predictions) to use during evaluation.
      skip_header_lines: int, number of header lines to skip in each source
        file.
      **task_kwargs: dict, additional keyword arguments for the parent `Task`
        class.
    """

    super().__init__(
        name,
        source=TextLineDataSource(
            split_to_filepattern=split_to_filepattern,
            skip_header_lines=skip_header_lines,
            num_input_examples=task_kwargs.pop("num_input_examples", None)),
        text_preprocessor=text_preprocessor,
        metric_fns=metric_fns,
        dataset_fn=None,
        splits=None,
        **task_kwargs)


class TFExampleTask(Task):
  """A `Task` that reads a file of tf.train.Example protos as input."""

  def __init__(
      self,
      name,
      split_to_filepattern,
      feature_description,
      text_preprocessor,
      metric_fns,
      reader=tf.data.TFRecordDataset,
      **task_kwargs):
    """TextLineTask constructor.

    Args:
      name: string, a unique name for the Task. A ValueError will be raised if
        another task with this name is already registered.
      split_to_filepattern: dict of string (split name) to string (filename or
        filepattern).
      feature_description: dict, a mapping of string feature keys to
        `tf.io.FixedLenFeature` or `tf.io.VarLenFeature` values.
      text_preprocessor: a function (or list of functions) that (each) takes in
        a tf.data.Dataset of string features and returns a tf.data.Dataset of
        string features. Can be set to None as a no-op. If a list is given, they
        will be executed sequentially.
      metric_fns: list(callable), list of metric functions with the signature
        metric_fn(targets, predictions) to use during evaluation.
      reader: `tf.data.Dataset`, a dataset class to read the input files.
      **task_kwargs: dict, additional keyword arguments for the parent `Task`
        class.
    """

    super().__init__(
        name,
        source=TFExampleDataSource(
            split_to_filepattern=split_to_filepattern,
            feature_description=feature_description,
            reader_cls=reader,
            num_input_examples=task_kwargs.pop("num_input_examples", None)),
        text_preprocessor=text_preprocessor,
        metric_fns=metric_fns,
        dataset_fn=None,
        splits=None,
        **task_kwargs)


class TaskRegistry(DatasetProviderRegistry):
  _REGISTRY = {}
  _PROVIDER_TYPE = TaskV3

  @classmethod
  def add(cls, name, task_cls=Task, **kwargs):
    super(TaskRegistry, cls).add(name, task_cls, name, **kwargs)


# ================================ Mixtures ====================================
class Mixture(DatasetProviderBase):
  """Class for mixing multiple tasks."""

  def __init__(
      self,
      name: str,
      tasks: Union[Iterable[str], Iterable[Tuple[str, int]]],
      default_rate: Union[float, Callable[[TaskV3], float]] = None):
    """Mixture constructor.

    A mixture specifies a set of tasks with associated mixing rates.

    Mixing happens on preprocessed tokenized examples.

    The mixing rates represent relative numbers of examples to use from their
    associated tasks.  Setting the mixing rates to be equal to the numbers of
    examples in the tasks will result in each task going through an epoch in
    about the same amount of time - i.e. all examples are sampled equally across
    all tasks.

    Rates can be expressed either as absolute numbers or as functions that
    receive the Task as an argument.

    Args:
      name: string, a unique name for the Mixture.
      tasks: a list where each element is either a string (task name) or a
        pair whose first element is the task name and whose second element
        is either a float (rate) or a function from Task to float.
      default_rate: a float or a function from Task to float. This specifies the
        default rate if rates are not provided in the `tasks` argument.
    """
    self._task_to_rate = {}
    self._tasks = []
    self._sub_mixtures = []
    self._name = name
    for t in tasks:
      if isinstance(t, str):
        task_name = t
        rate = default_rate
        if default_rate is None:
          raise ValueError("need a rate for each task")
      else:
        task_name, rate = t

      if task_name in TaskRegistry.names():
        self._tasks.append(TaskRegistry.get(task_name))
        self._task_to_rate[task_name] = rate
      else:
        self._sub_mixtures.append(MixtureRegistry.get(task_name))
        self._task_to_rate[task_name] = rate

    if len(set(tuple(t.output_features) for t in self.tasks)) != 1:
      raise ValueError(
          "All Tasks in a Mixture must have the same output features."
      )

  @property
  def name(self) -> str:
    return self._name

  @property
  def tasks(self) -> Iterable[TaskV3]:
    sub_tasks = (mix.tasks for mix in self._sub_mixtures)
    return list(sorted(set(sum(sub_tasks, self._tasks)), key=lambda t: t.name))

  @property
  def total_rate(self) -> float:
    return sum(float(rate(TaskRegistry.get(name)) if callable(rate) else rate)
               for name, rate in self._task_to_rate.items())

  def get_rate(self, task: TaskV3) -> float:
    """Computes the mixing rate for the given task."""
    value = 0.0

    for mix in self._sub_mixtures:
      if task in mix.tasks:
        rate = self._task_to_rate[mix.name]
        value += rate * mix.get_rate(task) / mix.total_rate

    if task.name in self._task_to_rate:
      rate = self._task_to_rate[task.name]
      value += float(rate(task) if callable(rate) else rate)

    return value

  def num_input_examples(self, split: str) -> int:
    return sum(t.num_input_examples(split) for t in self.tasks)

  @property
  def splits(self) -> Iterable[str]:
    splits = set()
    for task in self.tasks:
      splits.update(task.splits)
    return splits

  @property
  def output_features(self) -> Mapping[str, Feature]:
    # We require all tasks to have the same output_features in __init__
    # so we can just get the output_features for the 0th task
    return self.tasks[0].output_features

  def _check_same_vocabularies(self) -> None:
    """Throw an Exception if features across tasks have different vocabs."""
    for name, feature in self.tasks[0].output_features.items():
      for task in self.tasks[1:]:
        if task.output_features[name].vocabulary != feature.vocabulary:
          raise ValueError(
              "Features across tasks in a mixture must use the same vocabulary."
          )

  def get_dataset(
      self,
      sequence_length: Mapping[str, int],
      split: str = tfds.Split.TRAIN,
      use_cached: bool = False,
      shuffle: bool = True,
      seed: Optional[int] = None,
      copy_plaintext: bool = False,
      compute_stats_empirically: bool = False,
  ) -> tf.data.Dataset:
    """Returns the dataset of mixed tasks using the object-specified rates.

    Args:
      sequence_length: dict mapping feature key to int length for that feature
      split: string, the split to return for all tasks.
      use_cached: bool, whether to use the cached dataset instead of processing
        it on the fly. Defaults to False.
      shuffle: bool, whether to shuffle the dataset.  Only used when generating
        on the fly (use_cached=False).
      seed: tf.int64 scalar tf.Tensor (or None) for shuffling tf.data.
      copy_plaintext: bool, whether to pass through copies of plaintext strings
        with a "_plaintext" suffix added to the key.
      compute_stats_empirically: a boolean - does not work on TPU
    """
    self._check_same_vocabularies()
    tasks = []
    for task in self.tasks:
      if split not in task.splits:
        logging.info(
            "Task %s has no '%s' split, skipping.", task.name, split
        )
        continue
      tasks.append(task)
    if not tasks:
      raise ValueError("No datasets have a '{}' split".format(split))

    output_feature_keys = set(self.output_features.keys())
    if copy_plaintext:
      output_feature_keys.update(
          {f + "_plaintext" for f in output_feature_keys})

    def filter_features(ex):
      return {k: v for k, v in ex.items() if k in output_feature_keys}
    datasets = [
        task.get_dataset(  # pylint:disable=g-complex-comprehension
            sequence_length,
            split=split,
            use_cached=use_cached,
            shuffle=shuffle,
            seed=seed)
        .repeat()
        .map(filter_features, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        for task in tasks]
    rates = [self.get_rate(task) for task in tasks]
    # Sample from the dataset with the rates rates
    seed = None if shuffle else 42
    dataset = tf.data.experimental.sample_from_datasets(
        datasets, rates, seed)
    if (split == "train" and use_cached and
        all(t.supports_caching for t in tasks)):
      _log_mixing_proportions(tasks, datasets, rates, dataset, sequence_length,
                              compute_stats_empirically)
    return dataset


def _log_padding_fractions(dataset, sequence_length, num_examples=100):
  """Empirically compute the fraction of padding - log the results.

  Args:
    dataset: a tf.data.Dataset
    sequence_length: dict from string to int (packed lengths)
    num_examples: an integer
  """
  logging.info("computing padding fractions")
  keys = sequence_length.keys()
  padding_frac = {k: 0 for k in keys}
  for ex in tfds.as_numpy(dataset.take(num_examples)):
    for k in keys:
      padding_frac[k] += 1 - (sequence_length[k] / len(ex[k]))
  for k in keys:
    logging.info("%s padding fraction = %g", k, padding_frac[k])


def _log_mixing_proportions(
    tasks, datasets, rates, mixed_dataset,
    sequence_length, compute_stats_empirically):
  """Log information about the mixing proportions.

  Called from Mixture.get_dataset.

  Args:
    tasks: a list of Task
    datasets: a list of tf.data.Dataset
    rates: a list of floats
    mixed_dataset: a tf.data.Dataset
    sequence_length: dict from string to int (packed lengths)
    compute_stats_empirically: a boolean - does not work on TPU
  """
  def _normalize(l):
    denom = sum(l)
    if not denom:
      return l
    return [x / denom for x in l]
  # compute some stats about the mixture
  examples_fraction = _normalize(rates)
  if compute_stats_empirically:
    stats_examples = 100
    mean_inputs_length = []
    mean_targets_length = []
    for dataset in datasets:
      inputs_sum = 0
      targets_sum = 0
      for ex in tfds.as_numpy(dataset.take(stats_examples)):
        # Some tasks, like LMs, don't have inputs.
        if "inputs" in ex:
          inputs_sum += ex["inputs"].size
        targets_sum += ex["targets"].size
      mean_inputs_length.append(inputs_sum / float(stats_examples))
      mean_targets_length.append(targets_sum / float(stats_examples))
  else:
    def _estimated_mean_length(task, key):
      if task._cache_step_idx < len(task._preprocessors) - 1:  # pylint:disable=protected-access
        # There is processing after caching, so we can't rely on the stats.
        return sequence_length[key]
      # Some tasks, like LMs, don't have inputs.
      if key + "_tokens" in task.get_cached_stats("train"):
        return min(sequence_length[key],
                   (task.get_cached_stats("train")[key + "_tokens"] /
                    task.get_cached_stats("train")["examples"]))
      else:
        return 0

    mean_inputs_length = [_estimated_mean_length(task, "inputs")
                          for task in tasks]
    mean_targets_length = [_estimated_mean_length(task, "targets")
                           for task in tasks]
  inputs_fraction = _normalize(
      [l * r for l, r in zip(mean_inputs_length, rates)])
  targets_fraction = _normalize(
      [l * r for l, r in zip(mean_targets_length, rates)])
  logging.info("%12s %12s %12s %12s %12s %12s %s",
               "rate", "ex.frac.", "inp.frac.", "tgt.frac.",
               "inp.len.", "tgt.len", "task")
  for i in range(len(rates)):
    logging.info("%12g %12g %12g %12g %12g %12g %s",
                 rates[i], examples_fraction[i],
                 inputs_fraction[i], targets_fraction[i],
                 mean_inputs_length[i], mean_targets_length[i],
                 tasks[i].name)
  if compute_stats_empirically:
    _log_padding_fractions(mixed_dataset, sequence_length)


class MixtureRegistry(DatasetProviderRegistry):
  _REGISTRY = {}
  _PROVIDER_TYPE = Mixture

  @classmethod
  def add(cls, name, tasks, default_rate=None):
    super(MixtureRegistry, cls).add(name, Mixture, name, tasks, default_rate)


def get_mixture_or_task(task_or_mixture_name):
  """Return the Task or Mixture from the appropriate registry."""
  mixtures = MixtureRegistry.names()
  tasks = TaskRegistry.names()
  if task_or_mixture_name in mixtures:
    if task_or_mixture_name in tasks:
      logging.warning("%s is both a Task and a Mixture, returning Mixture",
                      task_or_mixture_name)
    return MixtureRegistry.get(task_or_mixture_name)
  if task_or_mixture_name in tasks:
    return TaskRegistry.get(task_or_mixture_name)
  else:
    raise ValueError("No Task or Mixture found with name: %s" %
                     task_or_mixture_name)


def get_subtasks(task_or_mixture):
  """Returns all the Tasks in a Mixture as a list or the Task itself."""
  if isinstance(task_or_mixture, TaskV3):
    return [task_or_mixture]
  else:
    return task_or_mixture.tasks


def _validate_args(fn, expected_pos_args):
  """Ensure function has exactly expected positional args."""
  argspec = inspect.getfullargspec(fn)
  expected_pos_args = tuple(expected_pos_args)
  actual_args = tuple(argspec.args)
  if actual_args[:len(expected_pos_args)] != expected_pos_args:
    raise ValueError(
        "'%s' must have positional args %s, got: %s" % (
            fn.__name__, expected_pos_args, actual_args))
  actual_pos_args = tuple(
      argspec.args[:-len(argspec.defaults)]
      if argspec.defaults else argspec.args)
  if actual_pos_args != expected_pos_args[:len(actual_pos_args)]:
    raise ValueError(
        "'%s' may only have positional args %s, got: %s" % (
            fn.__name__, expected_pos_args, actual_pos_args))
