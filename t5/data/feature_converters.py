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

We provide feature converters for the following three architectures:

  - encoder-decoder
  - decoder-only
  - encoder-only

The feature converters are composable mapping from tf.data.Dataset to
tf.data.Dataset. In other words, one or more compatible feature converters can
be composed together just like t5.data.preprocessors.

The input dataset has the fields or keys that are defined by the
`output_features` of the Task or Mixture under consideration. Often, these
fields are "inputs" and "targets" or just "targets" for the decoder-only models.

The role of feature converters is to convert these fields into a more
descriptive set of features to the model.


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

  - If a field requires an EOS token, it is already added prior to the
    converter.
  - Each field has length less than or equal to the lenghth specified by the
    input `sequence_length` argument. In particular, it is not padded yet.
  - The input dataset is not batched.
  - Each input tensor is one-dimensional (i.e., the length dimension).


For use-cases not covered by these standard cases, users need to define their
own feature-converter.
"""
from typing import MutableMapping, Mapping
# TODO(hwchung): transformer_dataset does not depend on mesh_tensorflow at all.
# move/modify/adapt the pack_or_pad to t5
from mesh_tensorflow.transformer import dataset as transformer_dataset
import tensorflow.compat.v2 as tf


def get_non_padding_position(tensor: tf.Tensor,
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


def encoder_decoder_feature_converter(
    ds: tf.data.Dataset,
    sequence_length: Mapping[str, int],
    pack: bool) -> tf.data.Dataset:
  """Feature converter for an encoder-decoder architecture.

  The input dataset has "inputs" and "targets" field. These will be converted
  to a subset of standard features. If pack = True, two additional fields are
  added for each of "inputs" and "targets".

  Example for a packed dataset:

    The input dataset has two examples each with "inputs" and "targets".

    ds = [{"inputs": [7, 8, 5, 1], "targets": [3, 9, 1]},
          {"inputs": [8, 4, 9, 3, 1], "targets": [4, 1]}]

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

  Args:
    ds: a tf.data.Dataset.
    sequence_length: dict mapping feature key to int length for that feature
    pack: whether to pack the dataset.

  Returns:
    a converted tf.data.Dataset with a subset of standard features.
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
         "decoder_loss_weight": get_non_padding_position(features["targets"])}

    if pack:
      d["encoder_segment_id"] = features["inputs_segmentation"]
      d["encoder_position"] = features["inputs_position"]
      d["decoder_segment_id"] = features["targets_segmentation"]
      d["decoder_position"] = features["targets_position"]

    return d

  ds = transformer_dataset.pack_or_pad(ds, sequence_length, pack=pack)
  ds = ds.map(map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  return ds


def to_text2self(features: Mapping[str, tf.Tensor]) -> Mapping[str, tf.Tensor]:
  """Convert an unpacked features from text2text to text2self format.

  Note that this function only works with an unpacked/unpadded dataset. Packing
  and padding should be applied after this function.

  Keys other than "inputs" and "targets" are not retaind.

  Example:
    {"inputs": [3, 5, 6, 9, 1], "targets": [4, 6, 8, 1]}
    -> {"targets": [3, 5, 6, 9, 1, 4, 6, 8, 1]}

  Args:
    features: a dict with "inputs" and "targets" keys.

  Returns:
    a dict with a concatenated a "targets" key.
  """
  text2self_features = {
      "targets": tf.concat([features["inputs"], features["targets"]], axis=-1)
  }
  return text2self_features


def decoder_feature_converter(ds: tf.data.Dataset,
                              sequence_length: Mapping[str, int],
                              pack: bool) -> tf.data.Dataset:
  """Feature converter for a decoder-only architecture.

  The input dataset has "targets" and an optional "inputs" field. If "inputs"
  field is present, these will be concatenated to form the new "targets".

  A decoder is a network which autoregressively produces an output sequence.
  Therefore, it can be used as a standard language model if an input data has
  "targets" field only, i.e., unsupervised. If the input data also has "inputs"
  field, e.g., machine translation, the decoder can still be used by
  concatenating the inputs and targets fields. See Raffel et al. (2020),
  https://arxiv.org/abs/1910.10683, Section 3.2.1 for more detailed take on
  this topic.

  If pack = True, "decoder_segment_id" and "decoder_position" fields are
  generated from the packing operation.

  Example 1: a packed dataset with "targets" only

    The input dataset has 2 examples, each is represented as a dict.

    ds = [{"targets": [3, 9, 1]}, {"targets": [4, 1]}]

    sequence_length = {"targets": 6}

    converted_ds = {
        "decoder_target_token": [3, 9, 1, 4, 1, 0],
         "decoder_input_token": [0, 3, 9, 0, 4, 0],
         "decoder_loss_weight": [1, 1, 1, 1, 1, 0],
            "decoder_position": [0, 1, 2, 0, 1, 0],
          "decoder_segment_id": [1, 1, 1, 2, 2, 0]
    }


  Exapmle 2: a packed dataset with "inputs" and "targets"
    ds = [{"inputs": [7, 8, 5, 1], "targets": [3, 9, 1]},
          {"inputs": [8, 4, 9, 3, 1], "targets": [4, 1]}]

    sequence_length = {"targets": 15}

    converted_ds = {
        "decoder_target_token": [7, 8, 5, 1, 3, 9, 1, 8, 4, 9, 3, 1, 4, 1, 0],
         "decoder_input_token": [0, 7, 8, 5, 1, 3, 9, 0, 8, 4, 9, 3, 1, 4, 0],
         "decoder_loss_weight": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            "decoder_position": [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0],
          "decoder_segment_id": [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 0],
    }

  Note that two examples are packed together into one example.

  Args:
    ds: a tf.data.Dataset.
    sequence_length: dict mapping feature key to int length for that feature
    pack: whether to pack the dataset.

  Returns:
    a converted tf.data.Dataset with a subset of standard features.
  """

  if "inputs" in ds.element_spec:
    # Concatenate inputs and outputs to form the text2self data format.
    ds = ds.map(to_text2self, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  ds = transformer_dataset.pack_or_pad(ds, sequence_length, pack=pack)

  def map_fn(features):
    # targets_segmentation is present only for a packed dataset.
    decoder_input_token = autoregressive_inputs(
        features["targets"],
        sequence_id=features.get("targets_segmentation", None))

    d = {"decoder_target_token": features["targets"],
         "decoder_input_token": decoder_input_token,
         "decoder_loss_weight": get_non_padding_position(features["targets"])}

    if pack:
      d["decoder_segment_id"] = features["targets_segmentation"]
      d["decoder_position"] = features["targets_position"]

    return d

  ds = ds.map(map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  return ds


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
    non_padding = get_non_padding_position(targets)
    inputs_portion *= non_padding

    features["inputs_attention_mask"] = inputs_portion
    return features

  return ds.map(map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
