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

"""Feature converters for common architectures.

In short, feature converters provide additional data processings to the
tf.data.Dataset out of the Task API. They convert the features of the input
dataset into more descriptive features (e.g., "decoder_target_token" instead of
"targets") as well as pad and/or pack them.

We provide feature converters for the following three architectures:

  - encoder-decoder
  - decoder-only
  - encoder-only

Each of these feature converters inherit the base class FeatureConverter and
override two methods `_convert_features` and `get_output_lengths` to
define how input features are mapped to the output features including the length
relationships. Other model architectures can be supported by subclassing the
FeatureConverter class in a similar manner.


Definition: standard_features

Throughout this module, we refer to the following 10 fields as
standard_features. Depending on the model architecture, a subset of them will be
returned by the feature converter.

  - encoder_input_token
  - encoder_target_token
  - encoder_loss_weight
  - encoder_position
  - encoder_segment_id
  - decoder_input_token
  - decoder_target_token
  - decoder_loss_weight
  - decoder_position
  - decoder_segment_id

  *_segment_id and *_position are only relevant for packed dataset.

  *_segment_id is a tf.Tensor of integer which is aligned with
  *_input_token. Positive integers represent the sequence membership in
  the packed examples. 0 represents padding. For example, encoder_segment_id =
  [1, 1, 2, 2, 2, 0] means that the first two positions belong to the first
  sequence, the next three to the second sequence and the last position is a
  padding.

  *_position is a tf.Tensor of integer representing the position index in the
  original sequence before packing. For example, consider
  encoder_position = [0, 1, 0, 1, 2, 0]. The first two tokens were the 0th and
  1st tokens of the first sequence and next three tokens are the 0th, 1st and
  2nd tokens of the second sequence before packing.

  *_loss_weight is used to indicate which positions should be used for the loss
  calculation.


Underlying assumptions

The feature converters implemented in this module assume the following about the
input dataset.

  - If EOS tokens are required, they are already appended in the input dataset.
  - The input dataset is not batched.
"""
import abc
import functools
from typing import Mapping, Sequence
import tensorflow.compat.v2 as tf


STANDARD_INPUTS = ["inputs", "targets"]
STANDARD_FEATURES = [
    "encoder_input_token",
    "encoder_target_token",
    "encoder_loss_weight",
    "encoder_position",
    "encoder_segment_id",
    "decoder_input_token",
    "decoder_target_token",
    "decoder_loss_weight",
    "decoder_position",
    "decoder_segment_id",
]


def _check_lengths(ds: tf.data.Dataset, expected_lengths: Mapping[str, int],
                   strict: bool, error_label: str) -> tf.data.Dataset:
  """Check the length of each feature in `ds` against `expected_lengths`.

  There are two checking criteria controlled by `strict` arg.

  If strict = True,
  for each feature in ds, check len(feature) == expected_lengths[feature].

  If strict = False,
  for each feature in ds, check len(feature) <= expected_lengths[feature].

  Features of the input dataset may have [None] shape. The assertion is run at
  the graph execution time when the length is determined.

  Args:
    ds: a tf.data.Dataset to be checked.
    expected_lengths: a mapping from a feature name to an expected length.
    strict: if true, the length of each feature should exactly match the
      expected length whereas false condition allows the length to be less
      than or equal to the expected length.
    error_label: a label used to indicate the validation stage

  Returns:
    ds: the same dataset as but with the assertion ops attached.
  """

  def _check_length(feat, v):
    if feat not in expected_lengths:
      return v

    if strict:
      error_message = (
          f"Feature '{feat}' has length not equal to the expected length of "
          f"{expected_lengths[feat]} during {error_label} validation")
      assertion_op = functools.partial(
          tf.debugging.assert_equal, message=error_message)
    else:
      error_message = (
          f"Feature '{feat}' has length not less than or equal to the expected "
          f"length of {expected_lengths[feat]} during {error_label} validation")
      assertion_op = functools.partial(
          tf.debugging.assert_less_equal, message=error_message)

    expected_length = tf.constant(expected_lengths[feat], dtype=tf.int64)
    # Assumes that v has rank of 1.
    actual_length = tf.size(v, out_type=tf.int64)
    assertion_op(actual_length, expected_length)
    return v

  ds = ds.map(
      lambda ex: {k: _check_length(k, v) for k, v in ex.items()},
      num_parallel_calls=tf.data.experimental.AUTOTUNE)

  return ds


class FeatureConverter(abc.ABC):
  """Abstract base class for feature converters.

  Subclasses of FeatureConverter are used to convert the tf.data.Dataset
  instance from the Task API to features that are passed to the
  model implementation. Note that Task API has an attribute "output_features",
  which is considered an "input features" in the context of FeatureConverter. To
  minimize confusion, output features always refer to the features of the
  dataset output by the FeatureConverter unless otherwise noted.

  Typically the input features contain keys: "inputs" and "targets". The output
  features are constructed based on what is consumed by the model architecture.
  For custom model architectures that require additional output features, one
  needs to subclass FeatureConverter.

  This conversion is fully specified by

    1. defining the mapping of the features in `_convert_features` method and
    2. defining the relationship between sequence lengths of input and output
       features in `get_output_lengths` which is a function of input_lengths.

  Therefore, a subclass of FeatureConverter should override `_convert_features`
  and `get_output_lengths` methods.

  The actual feature conversion is done in the `__call__` method, which
  wraps around the `_convert_features` method in order to provide useful checks
  and ensure compatibilities. See `_validate_dataset` and `__call__` methods
  for more details.

  Other notes:

    If pack = True, each feature in the input_features should be packable,
    i.e., 1-dimensional.

  Attributes:
    pack: whether to pack the dataset.
    input_dtypes: a mapping from a feature name to its data type.
    output_dtypes: a mapping from a feature name to its data type.
  """

  def __init__(self,
               pack: bool = True,
               input_dtypes: Mapping[str, tf.dtypes.DType] = None,
               output_dtypes: Mapping[str, tf.dtypes.DType] = None):
    self._pack = pack

    if input_dtypes is None:
      input_dtypes = {feat: tf.int32 for feat in STANDARD_INPUTS}
    self._input_dtypes = input_dtypes

    if output_dtypes is None:
      output_dtypes = {feat: tf.int32 for feat in STANDARD_FEATURES}
    self._output_dtypes = output_dtypes

  def _validate_dataset(self,
                        ds: tf.data.Dataset,
                        expected_features: Sequence[str],
                        expected_dtypes: Mapping[str, tf.dtypes.DType],
                        expected_lengths: Mapping[str, int],
                        strict: bool,
                        error_label: str,
                        expected_rank: int = 1) -> tf.data.Dataset:
    """Validate properties of the dataset, raising Exceptions if needed.

    Expanded from `t5.data.dataset_providers.TaskV3._validate_dataset`.

    This method is used to validate whether the input dataset is compatiable
    with the desired specifications. In particular, the following aspects are
    checked.

    Each feature in `expected_features`
      - is also in `ds`
      - has dtype = self.output_dtypes[feature]
      - has rank of 1
      - is also in expected_lengths
      - is compatible with the expected lengths

    The compatibility of the length is controlled by strict. If true, the length
    of each feature should exactly match the expected length whereas false
    condition allows the length to be less than or equal to the expected length.

    Args:
      ds: a tf.data.Dataset to be validated
      expected_features: expected features either in Mapping or List format.
      expected_dtypes: expected data type of each feature
      expected_lengths: a mapping from feature to its length
      strict: whether the lengths should be strictly equal or a length less than
        or equal to expected length is allowed.
      error_label: a label used to indicate the validation stage
      expected_rank: expected rank of each feature

    Returns:
      ds: the same dataset as but with the assertion ops attached.
    """
    element_spec = ds.element_spec
    for feat in expected_features:
      if feat not in element_spec:
        raise ValueError("Dataset is missing an expected feature during "
                         f"{error_label} validation: '{feat}'")

      if feat not in expected_dtypes:
        raise ValueError(f"A feature {feat} is missing in the expected_dtypes "
                         f"during {error_label} validation")

      if expected_dtypes[feat] != element_spec[feat].dtype:
        actual_dtype = element_spec[feat].dtype.name
        raise ValueError(
            f"Dataset has incorrect type for feature '{feat}' during "
            f"{error_label} validation: "
            f"Got {actual_dtype}, expected {expected_dtypes[feat].name}")

      if expected_rank != len(element_spec[feat].shape):
        actual_rank = len(element_spec[feat].shape)
        raise ValueError(
            f"Dataset has incorrect rank for feature '{feat}' during "
            f"{error_label} validation: "
            f"Got {actual_rank}, expected {expected_rank}")

      if feat not in expected_lengths:
        raise ValueError(f"Sequence length for feature '{feat}' is missing "
                         f"during {error_label} validation")

    ds = _check_lengths(ds, expected_lengths, strict, error_label)
    return ds

  def __call__(
      self, ds: tf.data.Dataset,
      input_lengths: Mapping[str, int]) -> tf.data.Dataset:
    """Convert the features of `ds` into output features.

    This method should not be overridden by subclasses.

    There are three major steps.

    1. The input dataset is validated. See _validate_dataset for what aspects
       are checked. At this stage, the length of each feature can be less than
       or equal to the specified length in `input_lengths`.

    2. Input features are converted to output features.

    3. The output dataset is validated. The length has to be exactly
       equal to the length specified in `output_lengths`.

    Args:
      ds: a tf.data.Dataset to be validated
      input_lengths: a mapping from a task feature to its length

    Returns:
      ds: the converted dataset.
    """
    # Input dataset validation stage
    ds = self._validate_dataset(
        ds,
        expected_features=input_lengths.keys(),
        expected_dtypes=self.input_dtypes,
        expected_lengths=input_lengths,
        # Before pack/pad, check feature (of ds) length <= task feature length
        strict=False,
        error_label="input_validation")

    # Input to output feature conversion, implemented by subclasses
    ds = self._convert_features(ds, input_lengths)

    # Output dataset validation stage
    output_lengths = self.get_output_lengths(input_lengths)
    ds = self._validate_dataset(
        ds,
        expected_features=output_lengths.keys(),
        expected_dtypes=self.output_dtypes,
        expected_lengths=output_lengths,
        # After pack/pad, check feature (of ds) length == model feature length
        strict=True,
        error_label="output_validation")

    return ds

  @abc.abstractmethod
  def _convert_features(
      self, ds: tf.data.Dataset,
      input_lengths: Mapping[str, int]) -> tf.data.Dataset:
    """Main feature conversion method to be overridden.."""
    raise NotImplementedError

  @abc.abstractmethod
  def get_output_lengths(self,
                         input_lengths: Mapping[str, int]) -> Mapping[str, int]:
    """Define the length relationship between input and output features."""
    raise NotImplementedError

  @property
  def pack(self) -> bool:
    return self._pack

  @property
  def input_dtypes(self) -> Mapping[str, tf.dtypes.DType]:
    return self._input_dtypes

  @property
  def output_dtypes(self) -> Mapping[str, tf.dtypes.DType]:
    return self._output_dtypes
