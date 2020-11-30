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

"""Tests for t5.data.feature_converters."""

from typing import Dict, Sequence
from unittest import mock
from t5.data import dataset_providers
from t5.data import feature_converters
from t5.data import test_utils
import tensorflow.compat.v2 as tf

tf.compat.v1.enable_eager_execution()

assert_dataset = test_utils.assert_dataset


class HelperFunctionsTest(tf.test.TestCase):

  def test_non_padding_position(self):
    x = tf.constant([3, 8, 5, 0, 0, 2, 0])
    non_padding_position = feature_converters.non_padding_position(x)
    expected = [1, 1, 1, 0, 0, 1, 0]
    actual = self.evaluate(non_padding_position)
    self.assertAllEqual(actual, expected)

  def test_shift_right_by_one(self):
    x = tf.constant([3, 8, 1, 0, 0])
    shifted = feature_converters._shift_right_by_one(x)
    expected = [0, 3, 8, 1, 0]
    actual = self.evaluate(shifted)
    self.assertAllEqual(actual, expected)

  def test_shift_right_by_one_nonzero_last_position(self):
    x = tf.constant([3, 8, 8, 9, 4])
    shifted = feature_converters._shift_right_by_one(x)
    expected = [0, 3, 8, 8, 9]
    actual = self.evaluate(shifted)
    self.assertAllEqual(actual, expected)

  def test_autoregressive_inputs_unpacked(self):
    x = tf.constant([3, 8, 9, 5, 1, 0, 0])
    autoreg_inputs = feature_converters.autoregressive_inputs(x)
    actual = self.evaluate(autoreg_inputs)
    expected = [0, 3, 8, 9, 5, 1, 0]
    self.assertAllEqual(actual, expected)

  def test_autoregressive_inputs_packed(self):
    x = tf.constant([3, 8, 1, 9, 1, 5, 4, 1, 0, 0])
    sequence_id = tf.constant([1, 1, 1, 2, 2, 3, 3, 3, 0, 0])
    autoreg_inputs = feature_converters.autoregressive_inputs(
        x, sequence_id=sequence_id)
    actual = self.evaluate(autoreg_inputs)
    expected = [0, 3, 8, 0, 9, 0, 5, 4, 0, 0]
    self.assertAllEqual(actual, expected)

  def test_autoregressive_inputs_packed_non_eos(self):
    # In the correct input format, x[4] should have been 1 (EOS).
    x = tf.constant([3, 8, 1, 9, 6, 5, 4, 1, 0, 0])
    # sequence_id is correctly formated.
    sequence_id = tf.constant([1, 1, 1, 2, 2, 3, 3, 3, 0, 0])
    autoreg_inputs = feature_converters.autoregressive_inputs(
        x, sequence_id=sequence_id)
    actual = self.evaluate(autoreg_inputs)
    # The incorrect x[4] should not affect the output as long as the sequence_id
    # is correct.
    expected = [0, 3, 8, 0, 9, 0, 5, 4, 0, 0]
    self.assertAllEqual(actual, expected)

  def test_lm_inputs_mask(self):
    decoder_target_token = [7, 8, 5, 1, 3, 9, 1, 8, 4, 9, 3, 1, 4, 1, 0, 0]
    x = {"decoder_target_token": decoder_target_token}
    ds = tf.data.Dataset.from_tensors(x)

    expected = x.copy()
    inputs_attention_mask = [1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]
    expected["inputs_attention_mask"] = inputs_attention_mask

    actual = feature_converters.lm_inputs_mask(ds, additional_position=False)
    assert_dataset(actual, expected)

  def test_lm_inputs_mask_additional_position(self):
    decoder_target_token = [7, 8, 5, 1, 3, 9, 1, 8, 4, 9, 3, 1, 4, 1, 0, 0]
    x = {"decoder_target_token": decoder_target_token}
    ds = tf.data.Dataset.from_tensors(x)

    expected = x.copy()
    inputs_attention_mask = [1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]
    expected["inputs_attention_mask"] = inputs_attention_mask

    actual = feature_converters.lm_inputs_mask(ds, additional_position=True)
    assert_dataset(actual, expected)


def create_default_dataset(
    x: Sequence[Dict[str, int]],
    feature_names: Sequence[str] = ("inputs", "targets")) -> tf.data.Dataset:
  output_types = {feature_name: tf.int64 for feature_name in feature_names}
  output_shapes = {feature_name: [None] for feature_name in feature_names}

  ds = tf.data.Dataset.from_generator(
      lambda: x, output_types=output_types, output_shapes=output_shapes)
  return ds


class FeatureConvertersTest(tf.test.TestCase):

  def setUp(self):
    super(FeatureConvertersTest, self).setUp()
    default_vocab = test_utils.sentencepiece_vocab()
    self.output_features = {
        "inputs": dataset_providers.Feature(vocabulary=default_vocab),
        "targets": dataset_providers.Feature(vocabulary=default_vocab)
    }

  def test_validate_dataset_missing_feature(self):
    x = [{"targets": [3, 9, 4, 5]}]
    ds = create_default_dataset(x, feature_names=["targets"])

    with mock.patch.object(feature_converters.FeatureConverter,
                           "__abstractmethods__", set()):
      converter = feature_converters.FeatureConverter(self.output_features)
      expected_msg = ("Dataset is missing an expected feature during "
                      "initial validation: 'inputs'")
      with self.assertRaisesRegex(ValueError, expected_msg):
        converter.validate_dataset(
            ds,
            expected_features=self.output_features,
            expected_type=tf.int64,
            error_label="initial")

  def test_validate_dataset_incorrect_dtype(self):
    x = [{"inputs": [9, 4, 3, 8, 6], "targets": [3, 9, 4, 5]}]
    ds = tf.data.Dataset.from_generator(
        lambda: x, output_types={"inputs": tf.int32, "targets": tf.int64},
        output_shapes={"inputs": [None], "targets": [None]})

    with mock.patch.object(feature_converters.FeatureConverter,
                           "__abstractmethods__", set()):
      converter = feature_converters.FeatureConverter(self.output_features)
      expected_msg = ("Dataset has incorrect type for feature 'inputs' during "
                      "initial validation: Got int32, expected int64")
      with self.assertRaisesRegex(ValueError, expected_msg):
        converter.validate_dataset(
            ds,
            expected_features=self.output_features,
            expected_type=tf.int64,
            error_label="initial")

  def test_validate_dataset_incorrect_rank(self):
    x = [{"inputs": [[9, 4, 3, 8, 6]], "targets": [[3, 9, 4, 5]]}]
    ds = tf.data.Dataset.from_generator(
        lambda: x, output_types={"inputs": tf.int32, "targets": tf.int64},
        output_shapes={"inputs": [None, 1], "targets": [None, 1]})

    with mock.patch.object(feature_converters.FeatureConverter,
                           "__abstractmethods__", set()):
      converter = feature_converters.FeatureConverter(self.output_features)
      expected_msg = ("Dataset has incorrect type for feature 'inputs' during "
                      "initial validation: Got int32, expected int64")
      with self.assertRaisesRegex(ValueError, expected_msg):
        converter.validate_dataset(
            ds,
            expected_features=self.output_features,
            expected_type=tf.int64,
            expected_rank=1,
            error_label="initial")

  def test_validate_dataset_missing_length(self):
    x = [{"inputs": [9, 4, 3, 8, 6], "targets": [3, 9, 4, 5]}]
    ds = tf.data.Dataset.from_generator(
        lambda: x, output_types={"inputs": tf.int64, "targets": tf.int64},
        output_shapes={"inputs": [5], "targets": [5]})

    with mock.patch.object(feature_converters.FeatureConverter,
                           "__abstractmethods__", set()):
      converter = feature_converters.FeatureConverter(self.output_features)
      expected_msg = ("Sequence length for feature 'targets' is missing "
                      "during final validation")
      with self.assertRaisesRegex(ValueError, expected_msg):
        converter.validate_dataset(
            ds,
            expected_features=self.output_features,
            expected_type=tf.int64,
            check_length=True,
            sequence_length={"inputs": 5},
            error_label="final")

  def test_validate_dataset_incorrect_length(self):
    x = [{"inputs": [9, 4, 3, 8, 6], "targets": [3, 9, 4, 5]}]
    ds = tf.data.Dataset.from_generator(
        lambda: x, output_types={"inputs": tf.int64, "targets": tf.int64},
        output_shapes={"inputs": [5], "targets": [5]})

    with mock.patch.object(feature_converters.FeatureConverter,
                           "__abstractmethods__", set()):
      converter = feature_converters.FeatureConverter(self.output_features)
      expected_msg = (
          "The sequence length of feature 'inputs' does not match the requested"
          " sequence length during final validation: Got 5, expected 7")
      with self.assertRaisesRegex(ValueError, expected_msg):
        converter.validate_dataset(
            ds,
            expected_features=self.output_features,
            expected_type=tf.int64,
            check_length=True,
            sequence_length={"inputs": 7, "targets": 5},
            error_label="final")

  def test_validate_dataset_ensure_no_eos(self):
    x = [{"inputs": [9, 4, 3, 8, 1], "targets": [3, 9, 4, 5]}]
    ds = tf.data.Dataset.from_generator(
        lambda: x, output_types={"inputs": tf.int64, "targets": tf.int64},
        output_shapes={"inputs": [5], "targets": [4]})

    with mock.patch.object(feature_converters.FeatureConverter,
                           "__abstractmethods__", set()):
      expected_msg = (r".*Feature \\'inputs\\' unexpectedly contains EOS=1 "
                      r"token during initial validation.*")
      with self.assertRaisesRegex(
          tf.errors.InvalidArgumentError,
          expected_msg):
        converter = feature_converters.FeatureConverter(self.output_features)
        ds = converter.validate_dataset(
            ds,
            expected_features=self.output_features,
            expected_type=tf.int64,
            ensure_no_eos=True,
            error_label="initial")
        list(ds.as_numpy_iterator())

  def test_validate_dataset_plaintext_field(self):
    x = [{"targets": [3, 9, 4, 5], "targets_plaintext": "some text"}]
    output_types = {"targets": tf.int64, "targets_plaintext": tf.string}
    output_shapes = {"targets": [4], "targets_plaintext": []}
    ds = tf.data.Dataset.from_generator(
        lambda: x, output_types=output_types, output_shapes=output_shapes)

    # ds has targets and targets_plaintext but output_features only has targets
    default_vocab = test_utils.sentencepiece_vocab()
    output_features = {
        "targets": dataset_providers.Feature(vocabulary=default_vocab)
    }

    with mock.patch.object(feature_converters.FeatureConverter,
                           "__abstractmethods__", set()):
      converter = feature_converters.FeatureConverter(output_features)
      ds = converter.validate_dataset(
          ds,
          expected_features=output_features,
          expected_type=tf.int64,
          ensure_no_eos=True,
          error_label="initial")


class EncoderDecoderTest(FeatureConvertersTest):

  def test_encoder_decoder_unpacked(self):
    x = [{"inputs": [9, 4, 3, 8], "targets": [3, 9, 4]}]
    ds = create_default_dataset(x)
    sequence_length = {"inputs": 7, "targets": 5}

    converter = feature_converters.EncoderDecoder(
        self.output_features, pack=False)
    converted_ds = converter.convert_features(ds, sequence_length)

    expected = {
        "encoder_input_token": [9, 4, 3, 8, 1, 0, 0],
        "decoder_target_token": [3, 9, 4, 1, 0],
        # mtf.transformer.autoregressive_inputs does not zero out the last eos
        # when the data is not packed. This test mimic the behavior.
        "decoder_input_token": [0, 3, 9, 4, 1],
        "decoder_loss_weight": [1, 1, 1, 1, 0],
    }
    assert_dataset(converted_ds, expected)

  def test_encoder_decoder_targets_max_length(self):
    x = [{"inputs": [9, 4, 3, 8], "targets": [3, 9, 4, 5]}]
    ds = create_default_dataset(x)
    sequence_length = {"inputs": 5, "targets": 5}

    converter = feature_converters.EncoderDecoder(
        self.output_features, pack=False)
    converted_ds = converter.convert_features(ds, sequence_length)

    expected = {
        "encoder_input_token": [9, 4, 3, 8, 1],
        "decoder_target_token": [3, 9, 4, 5, 1],
        "decoder_input_token": [0, 3, 9, 4, 5],
        "decoder_loss_weight": [1, 1, 1, 1, 1],
    }
    assert_dataset(converted_ds, expected)

  def test_encoder_decoder_extra_long_inputs(self):
    x = [{"inputs": [9, 4, 3, 8, 4, 5], "targets": [3, 9, 4, 7, 8]}]
    ds = create_default_dataset(x)
    sequence_length = {"inputs": 5, "targets": 4}

    converter = feature_converters.EncoderDecoder(
        self.output_features, pack=False)
    converted_ds = converter.convert_features(ds, sequence_length)

    # Check that the sequences are trimmed and the EOS token is added correctly.
    expected = {
        "encoder_input_token": [9, 4, 3, 8, 1],
        "decoder_target_token": [3, 9, 4, 1],
        "decoder_input_token": [0, 3, 9, 4],
        "decoder_loss_weight": [1, 1, 1, 1],
    }
    assert_dataset(converted_ds, expected)

  def test_encoder_decoder_packed(self):
    x = [{"inputs": [7, 8, 5], "targets": [3, 9]},
         {"inputs": [8, 4, 9, 3], "targets": [4]}]
    ds = create_default_dataset(x)
    sequence_length = {"inputs": 10, "targets": 7}

    converter = feature_converters.EncoderDecoder(
        self.output_features, pack=True)
    converted_ds = converter.convert_features(ds, sequence_length)

    expected = {
        "encoder_input_token": [7, 8, 5, 1, 8, 4, 9, 3, 1, 0],
        "encoder_segment_id": [1, 1, 1, 1, 2, 2, 2, 2, 2, 0],
        "encoder_position": [0, 1, 2, 3, 0, 1, 2, 3, 4, 0],
        "decoder_target_token": [3, 9, 1, 4, 1, 0, 0],
        "decoder_input_token": [0, 3, 9, 0, 4, 0, 0],
        "decoder_loss_weight": [1, 1, 1, 1, 1, 0, 0],
        "decoder_position": [0, 1, 2, 0, 1, 0, 0],
        "decoder_segment_id": [1, 1, 1, 2, 2, 0, 0],
    }
    assert_dataset(converted_ds, expected)

  def test_encoder_decoder_packed_long_sequences(self):
    x = [{"inputs": [7, 8, 5, 6, 9, 4], "targets": [3, 9]},
         {"inputs": [8, 4, 9, 3, 5, 7, 9], "targets": [4]}]
    ds = create_default_dataset(x)
    sequence_length = {"inputs": 7, "targets": 3}

    converter = feature_converters.EncoderDecoder(
        self.output_features, pack=True)
    converted_ds = converter.convert_features(ds, sequence_length)

    # Corner case: packing is true but sequence length too long for packing to
    # happen (truncation happens for the second example). We should still get
    # the *_segment_id, *_position fields.
    expected = [{
        "encoder_input_token": [7, 8, 5, 6, 9, 4, 1],
        "encoder_segment_id": [1, 1, 1, 1, 1, 1, 1],
        "encoder_position": [0, 1, 2, 3, 4, 5, 6],
        "decoder_target_token": [3, 9, 1],
        "decoder_input_token": [0, 3, 9],
        "decoder_loss_weight": [1, 1, 1],
        "decoder_position": [0, 1, 2],
        "decoder_segment_id": [1, 1, 1],
    }, {
        "encoder_input_token": [8, 4, 9, 3, 5, 7, 1],
        "encoder_segment_id": [1, 1, 1, 1, 1, 1, 1],
        "encoder_position": [0, 1, 2, 3, 4, 5, 6],
        "decoder_target_token": [4, 1, 0],
        "decoder_input_token": [0, 4, 0],
        "decoder_loss_weight": [1, 1, 0],
        "decoder_position": [0, 1, 0],
        "decoder_segment_id": [1, 1, 0],
    }]
    assert_dataset(converted_ds, expected)

  def test_encoder_decoder_plaintext_field(self):
    x = [{"inputs": [7, 8, 5], "targets": [3, 9], "targets_plaintext": "abc"},
         {"inputs": [8, 4, 9, 3], "targets": [4], "targets_plaintext": "def"}]
    types = {
        "inputs": tf.int64,
        "targets": tf.int64,
        "targets_plaintext": tf.string
    }
    shapes = {"inputs": [None], "targets": [None], "targets_plaintext": []}
    ds = tf.data.Dataset.from_generator(
        lambda: x, output_types=types, output_shapes=shapes)

    sequence_length = {"inputs": 10, "targets": 7}
    converter = feature_converters.EncoderDecoder(
        self.output_features, pack=True)
    # Check whether convert_features raise error because targets_plaintext is
    # present in the ds but not in the output_features
    converter.convert_features(ds, sequence_length)


class SingleStackTest(FeatureConvertersTest):

  def test_decoder_targets_only_unpacked(self):
    x = [{"targets": [3, 9]}]
    ds = create_default_dataset(x, feature_names=["targets"])
    sequence_length = {"targets": 5}

    del self.output_features["inputs"]
    converter = feature_converters.SingleStack(
        self.output_features, autoregressive=True, pack=False)
    converted_ds = converter.convert_features(ds, sequence_length)

    expected = {
        "decoder_target_token": [3, 9, 1, 0, 0],
        "decoder_input_token": [0, 3, 9, 1, 0],
        "decoder_loss_weight": [1, 1, 1, 0, 0],
    }
    assert_dataset(converted_ds, expected)

  def test_decoder_targets_only_packed(self):
    x = [{"targets": [3, 9]}, {"targets": [4]}]
    ds = create_default_dataset(x, feature_names=["targets"])
    sequence_length = {"targets": 6}

    del self.output_features["inputs"]
    converter = feature_converters.SingleStack(
        self.output_features, autoregressive=True, pack=True)
    converted_ds = converter.convert_features(ds, sequence_length)

    expected = {
        "decoder_target_token": [3, 9, 1, 4, 1, 0],
        "decoder_input_token": [0, 3, 9, 0, 4, 0],
        "decoder_loss_weight": [1, 1, 1, 1, 1, 0],
        "decoder_position": [0, 1, 2, 0, 1, 0],
        "decoder_segment_id": [1, 1, 1, 2, 2, 0]
    }
    assert_dataset(converted_ds, expected)

  def test_decoder_targets_only_pack_long_sequences(self):
    x = [{"targets": [3, 9, 4, 5]}, {"targets": [4, 3, 2, 7, 8, 6]}]
    ds = create_default_dataset(x, feature_names=["targets"])
    sequence_length = {"targets": 5}

    del self.output_features["inputs"]
    converter = feature_converters.SingleStack(
        self.output_features, autoregressive=True, pack=True)
    converted_ds = converter.convert_features(ds, sequence_length)

    expected = [{
        "decoder_target_token": [3, 9, 4, 5, 1],
        "decoder_input_token": [0, 3, 9, 4, 5],
        "decoder_loss_weight": [1, 1, 1, 1, 1],
        "decoder_position": [0, 1, 2, 3, 4],
        "decoder_segment_id": [1, 1, 1, 1, 1]
    }, {
        "decoder_target_token": [4, 3, 2, 7, 1],
        "decoder_input_token": [0, 4, 3, 2, 7],
        "decoder_loss_weight": [1, 1, 1, 1, 1],
        "decoder_position": [0, 1, 2, 3, 4],
        "decoder_segment_id": [1, 1, 1, 1, 1]
    }]
    assert_dataset(converted_ds, expected)

  def test_decoder_inputs_targets_unpacked(self):
    x = [{"inputs": [9, 4, 6], "targets": [3, 9]}]
    ds = create_default_dataset(x)

    sequence_length = {"inputs": 5, "targets": 4}
    converter = feature_converters.SingleStack(
        self.output_features, autoregressive=True, pack=False)
    converted_ds = converter.convert_features(ds, sequence_length)

    expected = {
        "decoder_target_token": [9, 4, 6, 1, 3, 9, 1, 0, 0],
        # The last EOS token is kept if unpacked.
        "decoder_input_token": [0, 9, 4, 6, 1, 3, 9, 1, 0],
        "decoder_loss_weight": [1, 1, 1, 1, 1, 1, 1, 0, 0],
    }
    assert_dataset(converted_ds, expected)

  def test_decoder_inputs_targets_packed(self):
    x = [{"inputs": [7, 8, 5], "targets": [3, 9]},
         {"inputs": [8, 4, 9, 3], "targets": [4]}]
    ds = create_default_dataset(x)

    sequence_length = {"inputs": 8, "targets": 7}
    converter = feature_converters.SingleStack(
        self.output_features, autoregressive=True, pack=True)
    converted_ds = converter.convert_features(ds, sequence_length)

    expected = {
        "decoder_target_token": [7, 8, 5, 1, 3, 9, 1, 8, 4, 9, 3, 1, 4, 1, 0],
        "decoder_input_token": [0, 7, 8, 5, 1, 3, 9, 0, 8, 4, 9, 3, 1, 4, 0],
        "decoder_loss_weight": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        "decoder_position": [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0],
        "decoder_segment_id": [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 0],
    }
    assert_dataset(converted_ds, expected)

  def test_decoder_inputs_targets_pack_long_sequences(self):
    x = [{"inputs": [7, 8, 5, 6], "targets": [3, 9, 7]},
         {"inputs": [8, 4, 9, 3, 8], "targets": [4]}]
    ds = create_default_dataset(x)

    sequence_length = {"inputs": 4, "targets": 3}
    converter = feature_converters.SingleStack(
        self.output_features, autoregressive=True, pack=True)
    converted_ds = converter.convert_features(ds, sequence_length)

    # Inputs and targets should be trimmed separately.
    expected = [{
        "decoder_target_token": [7, 8, 5, 1, 3, 9, 1],
        "decoder_input_token": [0, 7, 8, 5, 1, 3, 9],
        "decoder_loss_weight": [1, 1, 1, 1, 1, 1, 1],
        "decoder_position": [0, 1, 2, 3, 4, 5, 6],
        "decoder_segment_id": [1, 1, 1, 1, 1, 1, 1],
    }, {
        "decoder_target_token": [8, 4, 9, 1, 4, 1, 0],
        "decoder_input_token": [0, 8, 4, 9, 1, 4, 0],
        "decoder_loss_weight": [1, 1, 1, 1, 1, 1, 0],
        "decoder_position": [0, 1, 2, 3, 4, 5, 0],
        "decoder_segment_id": [1, 1, 1, 1, 1, 1, 0],
    }]
    assert_dataset(converted_ds, expected)

  def test_decoder_input_full_attention(self):
    x = [{"inputs": [7, 8, 5], "targets": [3, 9]},
         {"inputs": [8, 4, 9, 3], "targets": [4]}]
    ds = create_default_dataset(x)

    sequence_length = {"inputs": 7, "targets": 8}
    converter = feature_converters.SingleStack(
        self.output_features,
        pack=True,
        autoregressive=True,
        input_full_attention=True,
        additional_position=True)
    converted_ds = converter.convert_features(ds, sequence_length)

    expected = {
        "decoder_target_token": [7, 8, 5, 1, 3, 9, 1, 8, 4, 9, 3, 1, 4, 1, 0],
        "decoder_input_token": [0, 7, 8, 5, 1, 3, 9, 0, 8, 4, 9, 3, 1, 4, 0],
        "decoder_loss_weight": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        "decoder_position": [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0],
        "decoder_segment_id": [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 0],
        "inputs_attention_mask": [1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0]
    }
    assert_dataset(converted_ds, expected)

  def test_encoder_unpacked(self):
    x = [{
        # Assume 9 is the sentinel used to indicate prediction-tokens (e.g., for
        # MLM this would be [MASK] token).
        "inputs": [8, 9, 4, 9],
        "targets": [8, 7, 4, 6]
    }]

    ds = create_default_dataset(x)
    sequence_length = {"inputs": 6, "targets": 6}
    converter = feature_converters.SingleStack(
        self.output_features, pack=False, mask_id=9, autoregressive=False)
    converted_ds = converter.convert_features(ds, sequence_length)

    # Determine the loss weight by tf.equal(inputs == mask_sentinel)
    # Let 8 be the index of the sentinel used for classification. For BERT this
    # corresponds to [CLS] token.
    expected = {
        "encoder_input_token": [8, 9, 4, 9, 1, 0],
        "encoder_target_token": [8, 7, 4, 6, 1, 0],
        "encoder_loss_weight": [0, 1, 0, 1, 0, 0],
    }
    assert_dataset(converted_ds, expected)

  def test_encoder_packed(self):
    x = [{"inputs": [8, 9, 9, 3, 4], "targets": [8, 7, 4, 3, 4]},
         {"inputs": [8, 3, 9], "targets": [8, 3, 6]}]

    ds = create_default_dataset(x)
    sequence_length = {"inputs": 11, "targets": 11}
    converter = feature_converters.SingleStack(
        self.output_features, pack=True, mask_id=9, autoregressive=False)
    converted_ds = converter.convert_features(ds, sequence_length)

    expected = {
        "encoder_input_token": [8, 9, 9, 3, 4, 1, 8, 3, 9, 1, 0],
        "encoder_target_token": [8, 7, 4, 3, 4, 1, 8, 3, 6, 1, 0],
        "encoder_segment_id": [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 0],
        "encoder_position": [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 0],
        "encoder_loss_weight": [0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0],
    }
    assert_dataset(converted_ds, expected)

  def test_encoder_pack_long_sequences(self):
    x = [{"inputs": [8, 9, 9, 3, 4], "targets": [8, 7, 4, 3, 4]},
         {"inputs": [8, 3, 9], "targets": [8, 3, 6]}]

    ds = create_default_dataset(x)
    sequence_length = {"inputs": 5, "targets": 5}
    converter = feature_converters.SingleStack(
        self.output_features, pack=True, mask_id=9, autoregressive=False)
    converted_ds = converter.convert_features(ds, sequence_length)

    expected = [{
        "encoder_input_token": [8, 9, 9, 3, 1],
        "encoder_target_token": [8, 7, 4, 3, 1],
        "encoder_segment_id": [1, 1, 1, 1, 1],
        "encoder_position": [0, 1, 2, 3, 4],
        "encoder_loss_weight": [0, 1, 1, 0, 0],
    }, {
        "encoder_input_token": [8, 3, 9, 1, 0],
        "encoder_target_token": [8, 3, 6, 1, 0],
        "encoder_segment_id": [1, 1, 1, 1, 0],
        "encoder_position": [0, 1, 2, 3, 0],
        "encoder_loss_weight": [0, 0, 1, 0, 0],
    }]
    assert_dataset(converted_ds, expected)

  def test_encoder_plaintext_field(self):
    x = [{
        "inputs": [8, 9, 9, 3, 4],
        "targets": [8, 7, 4, 3, 4],
        "targets_plaintext": "abc"
    }, {
        "inputs": [8, 3, 9],
        "targets": [8, 3, 6],
        "targets_plaintext": "def"
    }]
    types = {
        "inputs": tf.int64,
        "targets": tf.int64,
        "targets_plaintext": tf.string
    }
    shapes = {"inputs": [None], "targets": [None], "targets_plaintext": []}
    ds = tf.data.Dataset.from_generator(
        lambda: x, output_types=types, output_shapes=shapes)

    sequence_length = {"inputs": 7, "targets": 7}
    converter = feature_converters.SingleStack(
        self.output_features, pack=True, mask_id=9, autoregressive=False)
    # Check whether convert_features raise error because targets_plaintext is
    # present in the ds but not in the output_features
    converter.convert_features(ds, sequence_length)

  def test_decoder_targets_plaintext_field(self):
    x = [{"targets": [3, 9], "targets_plaintext": "abc"},
         {"targets": [4], "targets_plaintext": "abc"}]
    types = {"targets": tf.int64, "targets_plaintext": tf.string}
    shapes = {"targets": [None], "targets_plaintext": []}
    ds = tf.data.Dataset.from_generator(
        lambda: x, output_types=types, output_shapes=shapes)
    sequence_length = {"targets": 6}

    del self.output_features["inputs"]
    converter = feature_converters.SingleStack(
        self.output_features, autoregressive=True, pack=True)
    converter.convert_features(ds, sequence_length)


if __name__ == "__main__":
  tf.test.main()
