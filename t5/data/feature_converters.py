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

In short, feature converters are mapping from "task features" to "model
features" where the former refers to the features output by the Task API
(typically "inputs" and "outputs") and the latter refers to the features
specific to the model architecture (typically standard features defined below).

We provide feature converters for the following three architectures:

  - encoder-decoder
  - decoder-only
  - encoder-only

Each of these feature converters inherit the base class FeatureConverter and
override two methods _convert_features and get_model_feature_sequence_length to
define how task features are mapped to the model features including the length
relationships.


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

  *_segment_id is an tf.Tensor of integer which is aligned with
  encoder_input_token. Positive integers represent the sequence membership in
  the packed examples. 0 represents padding. For example, encoder_segment_id =
  [1, 1, 2, 2, 2, 0] means that the first two positions belong to the first
  sequence, the next three to the second sequence and the last position is a
  padding.

  *_loss_weight is used to indicate which positions should be used for the loss
  calculation.


Underlying assumptions

The feature converters implemented in this module assume the following about the
input dataset.

  - EOS tokens are not appended (i.e., the output of the Task API).
  - The input dataset is not batched.

For use-cases not covered by these standard cases, users need to define their
own feature-converter.
"""
import abc
import collections
from typing import Mapping, Union, List, Iterable
# TODO(hwchung): transformer_dataset does not depend on mesh_tensorflow at all.
# move/modify/adapt the pack_or_pad to t5
from mesh_tensorflow.transformer import dataset as transformer_dataset
from t5.data import dataset_providers
import tensorflow.compat.v2 as tf


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


def autoregressive_inputs(targets: tf.Tensor,
                          sequence_id: tf.Tensor = None) -> tf.Tensor:
  """Generate inputs for an autoregressive model, by shifting the targets.

  Modified from mesh_tensorflow.transformer.transformer.autoregressive_inputs.

  For the first element of each sequence, the returned input id is 0.

  For a "packed" dataset, also pass the sequence_id tensor, which aligns
  with the targets tensor and contains different values for different
  concatenated examples.

  Example for a packed dataset:

        targets = [3, 8, 1, 9, 1, 5, 4, 1, 0, 0]
    sequence_id = [1, 1, 1, 2, 2, 3, 3, 3, 0, 0]
         inputs = [0, 3, 8, 0, 9, 0, 5, 4, 0, 0]
                            |     |        |
                            These positions are set to 0 in the if statement.

  Args:
    targets: a tf.int32 tensor with shape [length].
    sequence_id: an optional tensor with the same shape as targets.

  Returns:
    a tensor with dtype tf.int32 and the same shape as targets.
  """
  if targets.dtype != tf.int32:
    raise ValueError("The targets should be tf.int32 tensors.")

  if sequence_id is not None and sequence_id.dtype != tf.int32:
    raise ValueError(
        "The sequence_id should be tf.int32 tensors for a packed dataset.")

  inputs = _shift_right_by_one(targets)

  # We should have a 0 at the beginning of each sequence rather than the
  # shifted EOS (e.g. 1) from the previous sequence.
  if sequence_id is not None:
    not_first_in_sequence = tf.equal(
        sequence_id,
        _shift_right_by_one(sequence_id))
    inputs *= tf.cast(not_first_in_sequence, tf.int32)
  return inputs


class FeatureConverter(abc.ABC):
  """Abstract base class for feature converters.

  Subclasses of FeatureConverter are used to convert from "task features" to
  "model features". The former refers to the features output from the Task API.
  Typically they are "inputs" and "outputs". The model features are more
  descriptive features that are passed to the model implementation. Typically,
  they are standard features, which are defined in the module docstring.

  This conversion is fully specified by

    1. defining the mapping of the features in `_convert_features` method and
    2. defining the relationship between sequence lengths of task features and
      model features in `get_model_feature_sequence_length` which is a function
      of sequence_length (i.e., the sequence length of the task features).

  Therefore, a subclass of FeatureConverter should override _convert_features
  and get_model_feature_sequence_length methods.

  The actual feature conversion is done in the `convert_features` method, which
  wraps around the `_convert_features` method in order to provide useful checks
  and ensure compatibilities. See validate_dataset and convert_features methods
  for more details.

  Other notes:

    The input dataset to the feature converter should not have the EOS tokens
    appended. If EOS tokens are desired, this should be done in the
    `_convert_features` of each derived class. We provide a helper method
    `trim_and_ensure_eos` for that purpose.

    `output_features` is a terminology used in the Task API. This can be
    confusing because output_features is an input to the FeatureConverter.
    Therefore, we interchangeably use the term task features to mean
    output_features.

    This class also provides helper methods such as validating the dataset
    properties as well as useful properties.

    If pack = True, each feature in the output_features should be packable,
    i.e., 1-dimensional.

  Attributes:
    output_features: features of the input dataset to be converted. Also
      corresponds output_features from the Task API.
    pack: whether to pack the dataset.
    input_dtype: input data type, typically int64, for compatiblility with
      tf.Example proto.
    output_dtype: input data type, typically int32, for compatibility with TPUs.
  """

  def __init__(self,
               output_features: Mapping[str, dataset_providers.Feature],
               pack: bool = True,
               input_dtype: tf.dtypes.DType = tf.int64,
               output_dtype: tf.dtypes.DType = tf.int32):
    self._output_features = collections.OrderedDict(
        sorted(list(output_features.items()))
    )
    self._pack = pack
    self._input_dtype = input_dtype
    self._output_dtype = output_dtype

  def validate_dataset(
      self,
      ds: tf.data.Dataset,
      expected_features: Union[Mapping[str, dataset_providers.Feature],
                               List[str]],
      expected_type: tf.DType,
      error_label: str,
      ensure_no_eos: bool = False,
      check_length: bool = False,
      expected_rank: int = 1,
      sequence_length: Mapping[str, int] = None) -> tf.data.Dataset:
    """Generate inputs for an autoregressive model, by shifting the targets.

    Expanded from t5.data.dataset_providers.TaskV3._validate_dataset.

    This method is used to validate whether the input dataset is compatiable
    with the desired specifications. In particular, the following aspects are
    checked.

    Each feature in `expected_features`
      - is also in `ds`
      - has self.output_dtype
      - has rank of 1
      - is also in model_feature_sequence_length
      - has matching length in sequence_length

    The last two are optional, controlled by `check_length` arg. The last one
    only works if the sequence has a length dimension defined. For example, the
    output dataset of the Task API's get_dataset method typically has [None]
    shape, i.e., the length is not defined yet. In this case, the last check is
    skipped.

    Args:
      ds: a tf.data.Dataset to be validated
      expected_features: expected features either in Mapping or List format.
      expected_type: expected data type of each feature
      error_label: a label used to indicate the validation stage
      ensure_no_eos: whether to ensure that each feature does not contain the
        EOS id anywhere.
      check_length: whether to check the length of each feature
      expected_rank: expected rank of each feature
      sequence_length: a mapping from feature to its length

    Returns:
      ds: the same dataset as the inpu as the input. Internally, the TensorFlow
        graph may contain additional nodes if ensure_no_eos is set to True.
    """
    element_spec = ds.element_spec
    for feat in expected_features:
      if feat not in element_spec:
        raise ValueError("Dataset is missing an expected feature during "
                         f"{error_label} validation: '{feat}'")

      if expected_type != element_spec[feat].dtype:
        actual_dtype = element_spec[feat].dtype.name
        raise ValueError(
            f"Dataset has incorrect type for feature '{feat}' during "
            f"{error_label} validation: "
            f"Got {actual_dtype}, expected {expected_type.name}")

      if expected_rank != len(element_spec[feat].shape):
        actual_rank = len(element_spec[feat].shape)
        raise ValueError(
            f"Dataset has incorrect rank for feature '{feat}' during "
            f"{error_label} validation: "
            f"Got {actual_rank}, expected {expected_rank}")

      if check_length:
        if expected_rank == 0:
          raise ValueError(
              "If check_length is True, expected rank should be greater than 0."
          )

        if sequence_length is None:
          raise ValueError(
              "If check_length is True, sequence_length should be specified.")

        if feat not in sequence_length:
          raise ValueError(f"Sequence length for feature '{feat}' is missing "
                           f"during {error_label} validation")

        # At this point, the rank of each feature is strictly greater than 0.
        actual_length = element_spec[feat].shape.as_list()[0]
        # Prior to padding, the length is None. Skip the check.
        if actual_length is not None and sequence_length[feat] != actual_length:
          raise ValueError(
              f"The sequence length of feature '{feat}' does not match the "
              f"requested sequence length during {error_label} validation: "
              f"Got {actual_length}, expected {sequence_length[feat]}")

    def _ensure_no_eos(feat, v):
      if feat not in expected_features:
        return v
      error_message = (f"Feature '{feat}' unexpectedly contains EOS=1 token "
                       f"during {error_label} validation")
      with tf.control_dependencies([
          tf.debugging.assert_none_equal(
              v, tf.constant(1, tf.int64), message=error_message)
      ]):
        return v

    if ensure_no_eos:
      ds = ds.map(
          lambda ex: {k: _ensure_no_eos(k, v) for k, v in ex.items()})

    return ds

  def trim_and_ensure_eos(
      self,
      ds: tf.data.Dataset,
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
      return v

    return ds.map(
        lambda ex: {k: _trim_and_append_eos(k, v) for k, v in ex.items()},
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

  def convert_features(self, ds: tf.data.Dataset,
                       sequence_length: Mapping[str, int]) -> tf.data.Dataset:
    """This method should not be overridden by subclasses.

    In the initial validation
      - Each feature in self.output_features also in `ds`
      - Each feature in self.output_features has self.input_dtype
      - Each feature in self.output_features has rank of 1
      - Each feature in self.output_features also in sequence_length

    In the final validation
      - Each feature in model_features also in (output) ds
      - Each feature in model_features has self.output_dtype
      - Each feature in model_features has rank of 1
      - Each feature in model_features also in model_feature_sequence_length
      - Each feature in model_features has matching length as in
        model_feature_sequence_length.

    Therefore, the input dataset and the output dataset are compatible in terms
    of expected fields and length.

    Args:
      ds: a tf.data.Dataset to be validated
      sequence_length: a mapping from feature to its length

    Returns:
      ds: the converted dataset.
    """
    # Initial validation stage
    ds = self.validate_dataset(
        ds,
        expected_features=self.output_features,
        expected_type=self.input_dtype,
        error_label="initial",
        check_length=True,
        sequence_length=sequence_length,
        ensure_no_eos=True)

    # Main feature conversion, implemented by subclasses
    ds = self._convert_features(ds, sequence_length)

    model_feature_sequence_length = self.get_model_feature_sequence_length(
        sequence_length)
    # Final validation stage
    ds = self.validate_dataset(
        ds,
        expected_features=model_feature_sequence_length.keys(),
        expected_type=self.output_dtype,
        error_label="final",
        check_length=True,
        sequence_length=model_feature_sequence_length)

    return ds

  @abc.abstractmethod
  def _convert_features(self, ds: tf.data.Dataset,
                        sequence_length: Mapping[str, int]) -> tf.data.Dataset:
    """Main feature conversion method to be overridden.."""
    raise NotImplementedError

  @abc.abstractmethod
  def get_model_feature_sequence_length(
      self, sequence_length: Mapping[str, int]) -> Mapping[str, int]:
    """Define the length relationship between task and model features."""
    raise NotImplementedError

  @property
  def output_features(self) -> Mapping[str, dataset_providers.Feature]:
    return self._output_features

  @property
  def pack(self) -> bool:
    return self._pack

  @property
  def input_dtype(self) -> tf.dtypes.DType:
    return self._input_dtype

  @property
  def output_dtype(self) -> tf.dtypes.DType:
    return self._output_dtype


class EncoderDecoder(FeatureConverter):
  """Feature converter for an encoder-decoder architecture.

  The input dataset has "inputs" and "targets" field. These will be converted
  to a subset of standard features. If pack = True, two additional fields are
  added for each of "inputs" and "targets".

  Example for a packed dataset:

    The input dataset has two examples each with "inputs" and "targets".

    ds = [{"inputs": [7, 8, 5], "targets": [3, 9]},
          {"inputs": [8, 4, 9, 3], "targets": [4]}]

    sequence_length = {"inputs": 10, "targets": 7}

    First, the `inputs` are packed together, padded to length 10 and assigned to
    "encoder_input_token" field. The `targets` are processed similarly.

    The "*_segment_id" and *_position" fields are generated from the packing
    operation. For the explanation of these fields, see the module docstring.

    converted_ds = [{
         "encoder_input_token": [7, 8, 5, 1, 8, 4, 9, 3, 1, 0],
          "encoder_segment_id": [1, 1, 1, 1, 2, 2, 2, 2, 2, 0],
            "encoder_position": [0, 1, 2, 3, 0, 1, 2, 3, 4, 0],
        "decoder_target_token": [3, 9, 1, 4, 1, 0, 0],
         "decoder_input_token": [0, 3, 9, 0, 4, 0, 0],
         "decoder_loss_weight": [1, 1, 1, 1, 1, 0, 0],
            "decoder_position": [0, 1, 2, 0, 1, 0, 0],
          "decoder_segment_id": [1, 1, 1, 2, 2, 0, 0],
    }]

    Note that two examples are packed together into one example.


  If the input dataset contains features that are not in the `output_features`,
  they will not be included in the output dataset. One common scenario is the
  "inputs_plaintext" and "targets_plaintext" fields included in the old Task
  API. As long as they are not in the output_features, it is ok.

  Attributes:
    output_features: features of the input dataset to be converted. Also
      corresponds output_features from the Task API.
    pack: whether to pack the dataset.
  """

  def __init__(self,
               output_features: Mapping[str, dataset_providers.Feature],
               pack: bool = True,
               **kwargs):
    if "inputs" not in output_features or "targets" not in output_features:
      raise ValueError(
          "EncoderDecoder feature converter requires the output features "
          "to contain inputs and targets")
    super(EncoderDecoder, self).__init__(output_features, pack, **kwargs)

  def _convert_features(self, ds: tf.data.Dataset,
                        sequence_length: Mapping[str, int]) -> tf.data.Dataset:
    """Convert from task features to the model features.

    The conversion process involves three steps

    1. Each feature is trimmed and the EOS token is appended.
    2. Each feature in the self.output_features is packed.
    3. "inputs" fields are mapped to the encoder input and "targets" are mapped
      to decoder input (after being shifted) and target.

    Args:
      ds: an input tf.data.Dataset to be converted.
      sequence_length: a mapping from feature to its length.

    Returns:
      ds: an output dataset with the model features.
    """

    def map_fn(features):
      # targets_segmentation is present only for a packed dataset.
      decoder_input_token = autoregressive_inputs(
          features["targets"],
          sequence_id=features.get("targets_segmentation", None))

      d = {"encoder_input_token": features["inputs"],
           "decoder_target_token": features["targets"],
           "decoder_input_token": decoder_input_token,
           # Loss is computed for all but the padding positions.
           "decoder_loss_weight": non_padding_position(features["targets"])}

      if self.pack:
        d["encoder_segment_id"] = features["inputs_segmentation"]
        d["encoder_position"] = features["inputs_position"]
        d["decoder_segment_id"] = features["targets_segmentation"]
        d["decoder_position"] = features["targets_position"]

      return d

    ds = self.trim_and_ensure_eos(ds, sequence_length)
    ds = transformer_dataset.pack_or_pad(
        ds,
        sequence_length,
        pack=self.pack,
        # Assume that all features of output_features packable.
        feature_keys=self.output_features.keys())
    ds = ds.map(map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return ds

  def get_model_feature_sequence_length(
      self, sequence_length: Mapping[str, int]) -> Mapping[str, int]:
    """Define the length relationship between task and model features."""
    encoder_length = sequence_length["inputs"]
    decoder_length = sequence_length["targets"]

    model_sequence_length = {
        "encoder_input_token": encoder_length,
        "decoder_target_token": decoder_length,
        "decoder_input_token": decoder_length,
        "decoder_loss_weight": decoder_length
    }
    if self.pack:
      model_sequence_length["encoder_segment_id"] = encoder_length
      model_sequence_length["encoder_position"] = encoder_length
      model_sequence_length["decoder_segment_id"] = decoder_length
      model_sequence_length["decoder_position"] = decoder_length

    return model_sequence_length


def lm_inputs_mask(ds: tf.data.Dataset,
                   additional_position: bool = True,
                   eos_id: int = 1) -> tf.data.Dataset:
  """Binary mask indicating which parts of the ids represent the inputs.

  Expanded from mtf.transformer.transformer.delimited_lm_inputs_mask.

  When a decoder-only architecture is used for a seq2seq problem, i.e., with
  "inputs" and "targets" concatenated, the self attention for the "inputs" part
  can be fully visible. This was referred to as "prefix language model" in
  Raffel et al. (2020), https://arxiv.org/abs/1910.10683

  This function works for both packed and unpacked dataset.

  Assumes that the ids consist of packed sequences where each example is
  represented by two eos-terminated sequences, i.e.
  [<inputs0>, EOS, <targets0>, EOS, <inputs1>, EOS, <targets1>, EOS ...]

  As such, the inputs are the parts where the number of previous EOS tokens
  is even.

  The returned dataset has an additional field "inputs_attention_mask".

  Moreover, the position corresponding to the EOS token in the decoder input
  whose target is the first token of the next sequence (if present), can be
  added to the attention region. This option can be turned on with
  `additional_position = True`.

  Example:

    ds = {"targets": [7, 8, 5, 1, 3, 9, 1, 8, 4, 9, 3, 1, 4, 1, 0, 0]}
                      |  |  |  |           |  |  |  |  |
                        inputs                inputs

    The autoregressive input (i.e., the shifted target) is
                    inputs = [0, 7, 8, 5, 1, 3, 9, 0, 8, 4, 9, 3, 1, 4, 0, 0]

    1. additional_position = True
    output_ds = {
                   "targets: [7, 8, 5, 1, 3, 9, 1, 8, 4, 9, 3, 1, 4, 1, 0, 0],
    "inputs_attention_mask": [1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    }


    2. additional_position = False
    output_ds = {
                   "targets: [7, 8, 5, 1, 3, 9, 1, 8, 4, 9, 3, 1, 4, 1, 0, 0],
    "inputs_attention_mask": [1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    }

  Args:
    ds: a tf.data.Dataset.
    additional_position: Whether to include EOS token into the attention region.
    eos_id: EOS token id.

  Returns:
    a tf.data.Dataset with an additional field "inputs_attention_mask".
  """
  def map_fn(features):
    targets = features["decoder_target_token"]

    # Inicator of eos tokens.
    eos_tokens = tf.cast(tf.equal(targets, eos_id), tf.int32)

    # Number of eos tokens in the preceding positions.
    preceding_num_eos = tf.math.cumsum(eos_tokens, exclusive=True)

    # zero out positions with even number of preceding eos tokens.
    targets_portion = tf.math.floormod(preceding_num_eos, 2)

    # Positions corresponding to the "inputs" portion.
    inputs_portion = tf.equal(targets_portion, 0)

    if additional_position:
      shifted = tf.cast(
          _shift_right_by_one(tf.cast(inputs_portion, tf.int32)), tf.bool)
      inputs_portion = tf.math.logical_or(inputs_portion, shifted)

    inputs_portion = tf.cast(inputs_portion, tf.int32)

    # Zero out the padding part.
    non_padding = non_padding_position(targets)
    inputs_portion *= non_padding

    features["inputs_attention_mask"] = inputs_portion
    return features

  return ds.map(map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)


class SingleStack(FeatureConverter):
  """SingleStack serves as both encoder and decoder."""

  # TODO(hwchung): Add type annotation.
  def __init__(self,
               output_features: Mapping[str, dataset_providers.Feature],
               pack: bool,
               mask_id: int = None,
               autoregressive: bool = True,
               stack_name: str = "decoder",
               input_full_attention: bool = False,
               additional_position: bool = False,
               **kwargs):
    super().__init__(output_features, pack, **kwargs)
    required_features = {"targets"}
    if not autoregressive:
      required_features.add("inputs")

    for feat in required_features:
      if feat not in output_features:
        raise ValueError(
            f"The required features '{', '.join(required_features)}' do not "
            f"match the output_features: {', '.join(output_features.keys())}.")

    self._required_features = required_features

    self._autoregressive = autoregressive
    if not self.autoregressive and mask_id is None:
      raise ValueError("Non-autoregressive stacks require mask_id.")
    self.mask_id = mask_id
    self.input_full_attention = input_full_attention
    self.additional_position = additional_position

  def _convert_features(self, ds: tf.data.Dataset,
                        sequence_length: Mapping[str, int]) -> tf.data.Dataset:

    def map_fn(
        features: Mapping[str, tf.Tensor]) -> Mapping[str, tf.Tensor]:

      if self.autoregressive:
        input_token = autoregressive_inputs(
            features["targets"],
            sequence_id=features.get("targets_segmentation", None))
        loss_weight = non_padding_position(features["targets"])
      else:
        input_token = features["inputs"]
        # can we use non-padding-position somehow?
        loss_weight = tf.cast(
            tf.equal(features["inputs"], self.mask_id), tf.int32)

      d = {f"{self.stack_name}_target_token": features["targets"],
           f"{self.stack_name}_input_token": input_token,
           f"{self.stack_name}_loss_weight": loss_weight}

      if self.pack:
        d[f"{self.stack_name}_segment_id"] = features["targets_segmentation"]
        d[f"{self.stack_name}_position"] = features["targets_position"]

      return d

    ds = self.trim_and_ensure_eos(ds, sequence_length=sequence_length)

    # For text2self data format, we define a new sequence_length dict for
    # packing purpose.
    if self.autoregressive and "inputs" in sequence_length:
      def concat(features):
        return {
            "targets":
                tf.concat([features["inputs"], features["targets"]], axis=-1)
        }

      ds = ds.map(concat, num_parallel_calls=tf.data.experimental.AUTOTUNE)
      pack_sequence_length = {
          "targets": sequence_length["inputs"] + sequence_length["targets"]
      }
    else:
      pack_sequence_length = sequence_length

    # At this point, the dataset should only contain required features, all of
    # which can be packed. E.g., for decoder, whether or not "inputs" is
    # present, only "targets" is present at this point, which is stored in
    # required_features.
    ds = transformer_dataset.pack_or_pad(
        ds,
        pack_sequence_length,
        pack=self.pack,
        ensure_eos=True,
        feature_keys=self.required_features)
    ds = ds.map(map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if self.input_full_attention:
      ds = lm_inputs_mask(ds, self.additional_position, eos_id=1)

    return ds

  def get_model_feature_sequence_length(self, sequence_length):

    length = sequence_length["targets"]
    if self.autoregressive and "inputs" in sequence_length:
      length += sequence_length["inputs"]

    model_sequence_length = {
        f"{self.stack_name}_target_token": length,
        f"{self.stack_name}_input_token": length,
        f"{self.stack_name}_loss_weight": length
    }
    if self.pack:
      model_sequence_length[f"{self.stack_name}_segment_id"] = length
      model_sequence_length[f"{self.stack_name}_position"] = length

    if self.input_full_attention:
      model_sequence_length["inputs_attention_mask"] = length

    return model_sequence_length

  @property
  def autoregressive(self):
    return self._autoregressive

  @property
  def fully_autoregressive(self):
    return self.autoregressive and not self.input_full_attention

  @property
  def stack_name(self) -> str:
    if self.autoregressive:
      return "decoder"
    else:
      return "encoder"

  @property
  def required_features(self) -> Iterable[str]:
    return self._required_features
