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

"""Classes for data loading and processing.

Defines legacy Tasks and TaskRegistry. It is recommended that you use
seqio.Task for new projects.

For backward compatibility with the original T5 task format, these Tasks
separate preprocessing into text and token stages, with optional caching in
between.
"""

import re

import seqio
from t5.data import utils
import tensorflow.compat.v2 as tf

_DEFAULT_FEATURE_KEYS = ["inputs", "targets"]

_VALID_TASK_NAME_REGEX = re.compile(r"^[\w\d\._]+$")
_MAX_EXAMPLES_TO_MEM_CACHE = 10000
SHUFFLE_BUFFER_SIZE = 1000

# ================================ Tasks =======================================


class FunctionTask(seqio.Task):
  """A wrapper for `seqio.Task` using a `seqio.FunctionDataSource`.

  For backward compatibility with the original T5 task format, this task
  separates preprocessing into text and token stages, with optional caching in
  between.
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
    source = source or seqio.FunctionDataSource(
        dataset_fn=dataset_fn,
        splits=splits,
        num_input_examples=num_input_examples)

    if text_preprocessor and not hasattr(text_preprocessor, "__iter__"):
      text_preprocessor = [text_preprocessor]
    if token_preprocessor and not hasattr(token_preprocessor, "__iter__"):
      token_preprocessor = [token_preprocessor]

    preprocessors = list(text_preprocessor or [])
    preprocessors.append(seqio.preprocessors.tokenize)
    if supports_caching:
      preprocessors.append(seqio.CacheDatasetPlaceholder())
    preprocessors.extend(token_preprocessor or [])
    preprocessors.append(seqio.preprocessors.append_eos_after_trim)

    if hasattr(output_features, "__len__") and not output_features:
      raise ValueError("output_features must be non-empty.")
    if output_features is None:
      output_features = seqio.Feature(utils.get_default_vocabulary())
    if isinstance(output_features, dict):
      pass
    elif isinstance(output_features, seqio.Feature):
      output_features = {k: output_features for k in _DEFAULT_FEATURE_KEYS}
    elif isinstance(output_features, list) and all(
        isinstance(f, str) for f in output_features):
      output_features = {
          k: seqio.Feature(utils.get_default_vocabulary())
          for k in output_features
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


class TfdsTask(FunctionTask):
  """A wrapper for `seqio.Task` using a `seqio.TfdsDataSource`.

  Reads raw data from TensorFlow Datasets.

  For backward compatibility with the original T5 task format, this task
  separates preprocessing into text and token stages, with optional caching in
  between.
  """

  def __init__(self,
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
        source=seqio.TfdsDataSource(
            tfds_name=tfds_name, tfds_data_dir=tfds_data_dir, splits=splits),
        text_preprocessor=text_preprocessor,
        metric_fns=metric_fns,
        dataset_fn=None,
        splits=None,
        **task_kwargs)


class TextLineTask(FunctionTask):
  """A wrapper for `seqio.Task` using a `seqio.TextLineDataSource`.

  Reads raw data from a text file.

  For backward compatibility with the original T5 task format, this task
  separates preprocessing into text and token stages, with optional caching in
  between.

  Requires a text_processor to be passed that takes a tf.data.Dataset of
  strings and returns a tf.data.Dataset of feature dictionaries.
  e.g. preprocessors.preprocess_tsv()
  """

  def __init__(self,
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
        source=seqio.TextLineDataSource(
            split_to_filepattern=split_to_filepattern,
            skip_header_lines=skip_header_lines,
            num_input_examples=task_kwargs.pop("num_input_examples", None)),
        text_preprocessor=text_preprocessor,
        metric_fns=metric_fns,
        dataset_fn=None,
        splits=None,
        **task_kwargs)


class TFExampleTask(FunctionTask):
  """A wrapper for `seqio.Task` using a `seqio.TFExampleDataSource`.

  Reads raw data from a file of tf.train.Example protos.

  For backward compatibility with the original T5 task format, this task
  separates preprocessing into text and token stages, with optional caching in
  between.
  """

  def __init__(self,
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
        source=seqio.TFExampleDataSource(
            split_to_filepattern=split_to_filepattern,
            feature_description=feature_description,
            reader_cls=reader,
            num_input_examples=task_kwargs.pop("num_input_examples", None)),
        text_preprocessor=text_preprocessor,
        metric_fns=metric_fns,
        dataset_fn=None,
        splits=None,
        **task_kwargs)


class TaskRegistry(seqio.TaskRegistry):
  """Wrapper for seqio.TaskRegistry for backward-compatibility.

  Shares a registry with seqio.TaskRegistry.
  """

  @classmethod
  def add(cls, name: str, task_cls=FunctionTask, **kwargs) -> seqio.Task:
    provider = task_cls(name, **kwargs)
    super().add_provider(name, provider)
    return provider

  @classmethod
  def reset(cls) -> None:
    """Calls seqio.TaskRegistry.reset() to keep functionality consistent.

    Without this, calling t5.data.TaskRegistry.reset() initializes an empty
    registry (dictionary) of tasks. Afterwards, any task added to
    t5.data.TaskRegistry gets added to this newly initialized registry
    dictionary, instead of seqio.TaskRegistry.
    """
    seqio.TaskRegistry.reset()
