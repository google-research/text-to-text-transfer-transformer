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
dataset into more descriptive features (e.g., "decoder_target_tokens" instead of
"targets") as well as pad and/or pack them. The features of the input dataset
are referred to as "task_features" because they are the output of the Task API.
Those of the output dataset are referred to as "model_features" as they are the
features directly fed to the model implementation.

We provide feature converters for the following three architectures:

  - encoder-decoder
  - decoder-only
  - encoder-only

Each of these feature converters inherit the base class FeatureConverter and
override two methods `_convert_features` and `get_model_feature_lengths` to
define how task features are mapped to the model features including the length
relationships. Other model architectures can be supported by subclassing the
FeatureConverter class in a similar manner.


Definition: standard_features

Throughout this module, we refer to the following 10 fields as
standard_features. Depending on the model architecture, a subset of them will
be returned by the feature converter.

  - encoder_input_tokens
  - encoder_target_tokens
  - encoder_loss_weight
  - encoder_positions
  - encoder_segment_ids
  - decoder_input_tokens
  - decoder_target_tokens
  - decoder_loss_weights
  - decoder_positions
  - decoder_segment_ids

  *_segment_id and *_position fields are only relevant for packed dataset.

  *_segment_id is a tf.Tensor of integer which is aligned with
  *_input_token. Positive integers represent the sequence membership in
  the packed examples. 0 represents padding. For example, encoder_segment_ids =
  [1, 1, 2, 2, 2, 0] means that the first two positions belong to the first
  sequence, the next three to the second sequence and the last position is a
  padding.

  *_position is a tf.Tensor of integer representing the position index in the
  original sequence before packing. For example, consider
  encoder_positions = [0, 1, 0, 1, 2, 0]. The first two tokens were the 0th and
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
from absl import logging
from t5.data import dataset_providers
from t5.data import utils
import tensorflow.compat.v2 as tf


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


def non_padding_position(tensor: tf.Tensor,
                         dtype: tf.dtypes.DType = tf.int32,
                         pad_id: int = 0) -> tf.Tensor:
  """Return a tensor with 1 on non-padding and 0 on padding positions."""
  return tf.cast(tf.not_equal(tensor, pad_id), dtype=dtype)


def _shift_right_by_one(tensor: tf.Tensor, axis: int = -1) -> tf.Tensor:
  """Shift the 1d input tensor to the right by one position without wrapping."""

  if not tensor.dtype.is_integer:
    raise ValueError("Only integer types are supported.")

  # tf.roll wraps around the axis.
  rolled = tf.roll(tensor, shift=1, axis=axis)

  # Zero out the first position by multiplying with [0, 1, 1, ..., 1].
  reverse_onehot = tf.one_hot(0,
                              depth=tensor.shape[axis],
                              on_value=0,
                              off_value=1,
                              dtype=tensor.dtype)

  return rolled * reverse_onehot


def autoregressive_inputs(
    targets: tf.Tensor,
    sequence_id: tf.Tensor = None,
    output_dtype: tf.dtypes.DType = tf.int32) -> tf.Tensor:
  """Generate inputs for an autoregressive model, by shifting the targets.

  Modified from mesh_tensorflow.transformer.transformer.autoregressive_inputs.

  For the first element of each sequence, the returned input id is 0.

  For a "packed" dataset, also pass the sequence_id tensor, which aligns
  with the targets tensor and contains different values for different
  concatenated examples.

  Example for a packed dataset:

  ```
        targets = [3, 8, 1, 9, 1, 5, 4, 1, 0, 0]
    sequence_id = [1, 1, 1, 2, 2, 3, 3, 3, 0, 0]
         inputs = [0, 3, 8, 0, 9, 0, 5, 4, 0, 0]
                            |     |        |
                            These positions are set to 0 if sequence_id is not
                            None.
  ```

  Args:
    targets: a tf.int32 tensor with shape [length].
    sequence_id: an optional tensor with the same shape as targets.
    output_dtype: an optional output data type.

  Returns:
    a tensor with dtype tf.int32 and the same shape as targets.
  """
  if not targets.dtype.is_integer:
    raise ValueError("The targets should be integer-valued tensors.")

  if sequence_id is not None and not sequence_id.dtype.is_integer:
    raise ValueError(
        "The sequence_id should be integer-valued tensors for a packed dataset."
    )

  inputs = _shift_right_by_one(targets)
  if inputs.dtype != output_dtype:
    inputs = tf.cast(inputs, output_dtype)

  # We should have a 0 at the beginning of each sequence rather than the
  # shifted EOS (e.g. 1) from the previous sequence.
  if sequence_id is not None:
    not_first_in_sequence = tf.equal(
        sequence_id,
        _shift_right_by_one(sequence_id))
    inputs *= tf.cast(not_first_in_sequence, output_dtype)
  return inputs


def _check_exact_match(expected_features: Sequence[str],
                       actual_features: Sequence[str],
                       expected_feature_source: str,
                       actual_feature_source: str) -> None:
  """Check whether expected and actual features match one-to-one."""
  expected_features = set(expected_features)
  actual_features = set(actual_features)

  if expected_features != actual_features:
    if actual_features - expected_features:
      extra_features = actual_features - expected_features
      raise ValueError(
          f"The {actual_feature_source} contains extra features not specified "
          f"in the {expected_feature_source}: {extra_features}")
    else:
      missing_features = expected_features - actual_features
      raise ValueError(
          f"The {actual_feature_source} is missing features specified "
          f"in the {expected_feature_source}: {missing_features}")


class FeatureConverter(abc.ABC):
  """Abstract base class for feature converters.

  Subclasses of FeatureConverter are used to convert the tf.data.Dataset
  instance from the Task API to features that are passed to the
  model implementation. Note that Task API has an attribute "output_features",
  which is referred to as "model features" in the context of FeatureConverter.

  Typically the task features contain keys: "inputs" and "targets". The model
  features are constructed based on what is consumed by the model architecture.
  For custom model architectures that require additional model features, one
  needs to subclass FeatureConverter.

  This conversion is fully specified by

    1. defining the mapping of the features in `_convert_features` method and
    2. defining the relationship between sequence lengths of input and output
       features in `get_model_feature_lengths` which is a function of
       task_feature_lengths.

  Therefore, a subclass of FeatureConverter should override `_convert_features`
  and `get_model_feature_lengths` methods.

  The actual feature conversion is done in the `__call__` method, which
  wraps around the `_convert_features` method in order to provide useful checks
  and ensure compatibilities. See `_validate_dataset` and `__call__` methods
  for more details.

  Other notes:

    If pack = True, each feature in the task features should be packable,
    i.e., 1-dimensional.

    Subclasses must override TASK_FEATURE_DTYPES and MODEL_FEATURE_DTYPES. If
    packing is used, they must override PACKING_FEATURE_DTYPES as well. These
    are the packing-specific features such as "*_segment_ids".

  Attributes:
    pack: whether to pack the dataset.
    use_custom_packing_ops: whether to use custom ops for packing.
  """

  TASK_FEATURE_DTYPES: Mapping[str, tf.dtypes.DType]
  MODEL_FEATURE_DTYPES: Mapping[str, tf.dtypes.DType]
  PACKING_FEATURE_DTYPES: Mapping[str, tf.dtypes.DType]

  def __init__(self,
               pack: bool = True,
               use_custom_packing_ops: bool = False):
    self._pack = pack
    self._use_custom_packing_ops = use_custom_packing_ops

    if self.TASK_FEATURE_DTYPES is None:
      raise ValueError("TASK_FEATURE_DTYPES must be defined in the subclass.")

    if self.MODEL_FEATURE_DTYPES is None:
      raise ValueError("MODEL_FEATURE_DTYPES must be defined in the subclass.")

    if self.pack and self.PACKING_FEATURE_DTYPES is None:
      raise ValueError(
          "PACKING_FEATURE_DTYPES must be defined in the subclass if pack=True."
      )

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
      - has dtype = self.MODEL_FEATURE_DTYPES[feature]
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

    ds = _check_lengths(ds, expected_lengths, strict, error_label)
    return ds

  def __call__(self, ds: tf.data.Dataset,
               task_feature_lengths: Mapping[str, int]) -> tf.data.Dataset:
    r"""Convert the features of `ds` into output features.

    This method should not be overridden by subclasses.

    There are two conversion steps and five validation steps.

    Conversion 1: task features are converted to model features in
                  `_convert_features

    Conversion 2: task_feature_lengths are converted to model_feature_lengths in
                  `get_model_feature_lengths`

    Validation 1: verifies that the user input `task_feature_lengths` only
                  contains the required features.

    Validation 2: verifies whether the input dataset has same or more features,
                  same dtype, and length that is less than or equal compared to
                  input_ds.

    Validation 3: partially verifies the behavior of overriden
                  `get_model_feature_lengths`.

    Validation 4: check whether the ouput dataset has expected features (extra
                  features are allowed), dtype, rank and lengths (exact mactch).

    Validation 5: check one-to-one match between the output dataset and
                  `expected_dtypes`. Extra features are not allowed.

    The following diagram describes the validation and conversion processes. We
    treat features in the TASK_FEATURE_DTYPES and MODEL_FEATURE_DTYPES specified
    as class variables as the ground-truth. For validations 3, 4 and 5, we
    define `expected_dtypes`.

    There are 5 validation steps. features (<=) means that features of the
    variable on the left is a subset of those of the variable on the right. For
    example, validation 2 guarantees that TASK_FEATURE_DTYPES have features that
    are subset of the features of input_ds. Validation 4 has length (==), which
    means that it ensures that each feature in MODEL_FEATURE_DTYPES has the same
    length as the corresponding feature in output_ds.

    Overall, these 5 validations ensures that the output_ds has the expected
    features with exact length, dtype and rank. Again, these validations assume
    that TASK_FEATURE_DTYPES and MODEL_FEATURE_DTYPES are correct.


                        Validation 1                     Validation 2
    task_feature_lengths <-------> TASK_FEATURE_DTYPES <--------------> input_ds
    |                   features (==)                    features (<=)        |
    |                                                    dtype (==)           |
    |                                                    length (<=)          |
    |                                                    rank (==1)           |
    |                                                                         |
    |   Conversion 2                                           Conversion 1   |
    | get_model_feature_lengths                             _convert_features |
    |                                                                         |
    |                                              Validation 5               |
    |                                           <-------------------->        |
    |                                                 features (==)           |
    |                                                                         |
    \/                    Validation 3                    Validation 4        \/
    model_feature_lengths <-------> expected_dtypes <----------------> output_ds
                        features (==)                     features (<=)
                                                          dtype (==)
                                                          length (==)
                                                          rank (==1)

    Args:
      ds: a tf.data.Dataset to be validated
      task_feature_lengths: a mapping from a task feature to its length

    Returns:
      ds: the converted dataset.
    """
    # Validation 1
    _check_exact_match(expected_features=list(self.TASK_FEATURE_DTYPES),
                       actual_features=list(task_feature_lengths),
                       expected_feature_source="TASK_FEATURE_DTYPES",
                       actual_feature_source="task_feature_lengths")

    # Validation 2
    ds = self._validate_dataset(
        ds,
        expected_features=list(self.TASK_FEATURE_DTYPES),
        expected_dtypes=self.TASK_FEATURE_DTYPES,
        expected_lengths=task_feature_lengths,
        # Before pack/pad, check feature (of ds) length <= task feature length
        strict=False,
        error_label="input_validation")

    # Conversion 1: implemented by subclass
    ds = self._convert_features(ds, task_feature_lengths)

    expected_dtypes = dict(self.MODEL_FEATURE_DTYPES)
    if self.pack:
      expected_dtypes = {**expected_dtypes, **self.PACKING_FEATURE_DTYPES}

    # Conversion 2: implemented by subclasses
    model_feature_lengths = self.get_model_feature_lengths(task_feature_lengths)

    # Validation 3
    _check_exact_match(expected_features=list(expected_dtypes),
                       actual_features=list(model_feature_lengths),
                       expected_feature_source="model_feature_names",
                       actual_feature_source="model_feature_lengths")

    # Validation 4
    ds = self._validate_dataset(
        ds,
        expected_features=list(expected_dtypes),
        expected_dtypes=expected_dtypes,
        expected_lengths=model_feature_lengths,
        # After pack/pad, check feature (of ds) length == model feature length
        strict=True,
        error_label="output_validation")

    # Validation 5
    _check_exact_match(expected_features=list(expected_dtypes),
                       actual_features=list(ds.element_spec),
                       expected_feature_source="model_feature_names",
                       actual_feature_source="output_dataset")
    return ds

  def _pack_or_pad(self,
                   ds: tf.data.Dataset,
                   packed_lengths: Mapping[str, int]) -> tf.data.Dataset:
    """Trim/pad to packed_lengths and optionally pack the input dataset."""
    if self.pack:
      ds = utils.trim_and_pack_dataset(ds, packed_lengths,
                                       self._use_custom_packing_ops)
    else:
      ds = utils.trim_and_pad_dataset(ds, packed_lengths)
    return ds

  @abc.abstractmethod
  def _convert_features(
      self, ds: tf.data.Dataset,
      task_feature_lengths: Mapping[str, int]) -> tf.data.Dataset:
    """Main feature conversion method to be overridden.."""
    raise NotImplementedError

  @abc.abstractmethod
  def get_model_feature_lengths(
      self, task_feature_lengths: Mapping[str, int]) -> Mapping[str, int]:
    """Define the length relationship between task and model features."""
    raise NotImplementedError

  @property
  def pack(self) -> bool:
    return self._pack


class EncDecFeatureConverter(FeatureConverter):
  """Feature converter for an encoder-decoder architecture.

  The input dataset has "inputs" and "targets" field. These will be converted
  to a subset of standard features.

  To use packing, pass pack = True argument to the FeatureConverter's
  constructor. When packing is done, two additional fields are added for each of
  "inputs" and "targets" fields.

  Example for a packed dataset:

  The input dataset has two examples each with "inputs" and "targets".

  ds = [{"inputs": [7, 8, 5, 1], "targets": [3, 9, 1]},
        {"inputs": [8, 4, 9, 3, 1], "targets": [4, 1]}]

  task_feature_lengths = {"inputs": 10, "targets": 7}

  First, the `inputs` are packed together, padded to length 10 and assigned to
  "encoder_input_tokens" field. The `targets` are processed similarly.

  The "*_segment_id" fields are generated from the packing operation. For the
  explanation of these fields, see the module docstring.

  The "decoder_loss_weights" is a binary mask indicating where non-padding
  positions are, i.e., value of 1 indicates non-padding and 0 for padding. This
  class assumes that the loss is taken only on the decoder side.

  converted_ds = [{
       "encoder_input_tokens": [7, 8, 5, 1, 8, 4, 9, 3, 1, 0],
        "encoder_segment_ids": [1, 1, 1, 1, 2, 2, 2, 2, 2, 0],
          "encoder_positions": [0, 1, 2, 3, 0, 1, 2, 3, 4, 0],
      "decoder_target_tokens": [3, 9, 1, 4, 1, 0, 0],
       "decoder_input_tokens": [0, 3, 9, 0, 4, 0, 0],
       "decoder_loss_weights": [1, 1, 1, 1, 1, 0, 0],
        "decoder_segment_ids": [1, 1, 1, 2, 2, 0, 0],
          "decoder_positions": [0, 1, 2, 0, 1, 0, 0],
  }]

  Note that two examples are packed together into one example.
  """

  TASK_FEATURE_DTYPES = {"inputs": tf.int32, "targets": tf.int32}
  MODEL_FEATURE_DTYPES = {
      "encoder_input_tokens": tf.int32,
      "decoder_target_tokens": tf.int32,
      "decoder_input_tokens": tf.int32,
      "decoder_loss_weights": tf.int32,
  }
  PACKING_FEATURE_DTYPES = {
      "encoder_segment_ids": tf.int32,
      "decoder_segment_ids": tf.int32,
      "encoder_positions": tf.int32,
      "decoder_positions": tf.int32
  }

  def _convert_features(
      self, ds: tf.data.Dataset,
      task_feature_lengths: Mapping[str, int]) -> tf.data.Dataset:
    """Convert the dataset to be fed to the encoder-decoder model.

    The conversion process involves three steps

    1. Each feature in the `task_feature_lengths` is trimmed/padded and
       optionally packed depending on the value of self.pack.
    2. "inputs" fields are mapped to the encoder input and "targets" are mapped
       to decoder input (after being shifted) and target.

    All the keys in the `task_feature_lengths` should be present in the input
    dataset, which may contain some extra features that are not in the
    `task_feature_lengths`. They will not be included in the output dataset.
    One common scenario is the "inputs_plaintext" and "targets_plaintext"
    fields.

    Args:
      ds: an input tf.data.Dataset to be converted.
      task_feature_lengths: a mapping from feature to its length.

    Returns:
      ds: the converted dataset.
    """

    def convert_example(
        features: Mapping[str, tf.Tensor]) -> Mapping[str, tf.Tensor]:
      # targets_segment_id is present only for a packed dataset.
      decoder_input_tokens = autoregressive_inputs(
          features["targets"],
          sequence_id=features.get("targets_segment_ids", None))

      d = {"encoder_input_tokens": features["inputs"],
           "decoder_target_tokens": features["targets"],
           "decoder_input_tokens": decoder_input_tokens,
           # Loss is computed for all but the padding positions.
           "decoder_loss_weights": non_padding_position(features["targets"])}

      if self.pack:
        d["encoder_segment_ids"] = features["inputs_segment_ids"]
        d["decoder_segment_ids"] = features["targets_segment_ids"]
        d["encoder_positions"] = features["inputs_positions"]
        d["decoder_positions"] = features["targets_positions"]

      return d

    ds = self._pack_or_pad(ds, task_feature_lengths)
    return ds.map(
        convert_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  def get_model_feature_lengths(
      self, task_feature_lengths: Mapping[str, int]) -> Mapping[str, int]:
    """Define the length relationship between input and output features."""
    encoder_length = task_feature_lengths["inputs"]
    decoder_length = task_feature_lengths["targets"]

    model_feature_lengths = {
        "encoder_input_tokens": encoder_length,
        "decoder_target_tokens": decoder_length,
        "decoder_input_tokens": decoder_length,
        "decoder_loss_weights": decoder_length
    }
    if self.pack:
      model_feature_lengths["encoder_segment_ids"] = encoder_length
      model_feature_lengths["decoder_segment_ids"] = decoder_length
      model_feature_lengths["encoder_positions"] = encoder_length
      model_feature_lengths["decoder_positions"] = decoder_length

    return model_feature_lengths


def get_dataset(
    mixture_or_task_name: str,
    task_feature_lengths: Mapping[str, int],
    feature_converter: FeatureConverter,
    dataset_split: str = "train",
    use_cached: bool = False,
    shuffle: bool = True,
    shard_info: dataset_providers.ShardInfo = None,
    verbose: bool = True
) -> tf.data.Dataset:
  """Get processed dataset with the model features.

  In order to use options specific to a feature converter, e.g., packing,
  `feature_converter` instance should be instantiated with those options before
  being pased to this function.

  Getting sharded datasets is supported. To use this feature, pass in
  `shard_info`, with shard_index and num_shards information. Sharding is done
  before the feature converter stage. Therefore, if packing is used it will be
  done on the sharded dataset.

  Args:
    mixture_or_task_name: mixture or task name for the Task API.
    task_feature_lengths: dict mapping task feature key to its sequence length.
      This specifies the sequence length of the dataset from the Task API.
    feature_converter: a feature converter object to use to convert the task
      features to model features.
      Must be a subclass of FeatureConverter.
    dataset_split: the split to use.
    use_cached: whether to use the cached dataset instead of processing it on
      the fly.
    shuffle: whether to shuffle the dataset.
    shard_info: number of shards and shard index information.
    verbose: if true, log the feature shapes.

  Returns:
    ds: the processed dataset.
  """
  if not isinstance(feature_converter, FeatureConverter):
    raise TypeError(
        "feature_converter should be an instance of FeatureConverter.")

  mixture_or_task = dataset_providers.get_mixture_or_task(mixture_or_task_name)

  ds = mixture_or_task.get_dataset(
      task_feature_lengths,
      split=dataset_split,
      use_cached=use_cached,
      shuffle=shuffle,
      shard_info=shard_info)

  ds = feature_converter(ds, task_feature_lengths=task_feature_lengths)

  if verbose:
    logging.info(
        "The output dataset from t5.get_dataset has the following features")
    for feature_name, tensor_spec in ds.element_spec.items():
      logging.info("feature: %s \t shape: %s \t dtype: %s", feature_name,
                   tensor_spec.shape.as_list(), tensor_spec.dtype.name)
  return ds
