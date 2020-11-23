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

from t5.data import feature_converters
from t5.data import test_utils
import tensorflow.compat.v2 as tf

assert_dataset = test_utils.assert_dataset


class FeatureConvertersTest(tf.test.TestCase):

  def test_get_non_padding_position(self):
    x = tf.constant([3, 8, 5, 0, 0, 2, 0])
    non_padding_position = feature_converters.get_non_padding_position(x)
    expected = [1, 1, 1, 0, 0, 1, 0]
    actual = self.evaluate(non_padding_position)
    self.assertAllEqual(actual, expected)

  def test_shift_right_by_one(self):
    x = tf.constant([3, 8, 1, 0, 0])
    shifted = feature_converters._shift_right_by_one(x)
    expected = [0, 3, 8, 1, 0]
    actual = self.evaluate(shifted)
    self.assertAllEqual(actual, expected)

  def test_autoregressive_inputs_packed(self):
    x = tf.constant([3, 8, 1, 9, 1, 5, 4, 1, 0, 0])
    sequence_id = tf.constant([1, 1, 1, 2, 2, 3, 3, 3, 0, 0])
    autoreg_inputs = feature_converters.autoregressive_inputs(
        x, sequence_id=sequence_id)
    actual = self.evaluate(autoreg_inputs)
    expected = [0, 3, 8, 0, 9, 0, 5, 4, 0, 0]
    self.assertAllEqual(actual, expected)

  def test_encoder_decoder_feature_converter_unpacked(self):
    x = {"inputs": [9, 4, 3, 8, 1], "targets": [3, 9, 4, 1]}
    ds = tf.data.Dataset.from_tensors(x)
    sequence_length = {"inputs": 7, "targets": 5}
    converted_ds = feature_converters.encoder_decoder_feature_converter(
        ds, sequence_length=sequence_length, pack=False)
    expected = {
        "encoder_input_token": [9, 4, 3, 8, 1, 0, 0],
        "decoder_target_token": [3, 9, 4, 1, 0],
        # mtf.transformer.autoregressive_inputs does not zero out the last eos
        # when the data is not packed. This test mimic the behavior.
        "decoder_input_token": [0, 3, 9, 4, 1],
        "decoder_loss_weight": [1, 1, 1, 1, 0],
    }
    assert_dataset(converted_ds, expected)

  def test_encoder_decoder_feature_converter_packed(self):
    x = [{"inputs": [7, 8, 5, 1], "targets": [3, 9, 1]},
         {"inputs": [8, 4, 9, 3, 1], "targets": [4, 1]}]
    ds = tf.data.Dataset.from_generator(
        lambda: x, output_types={"inputs": tf.int32, "targets": tf.int32})
    sequence_length = {"inputs": 10, "targets": 7}
    converted_ds = feature_converters.encoder_decoder_feature_converter(
        ds, sequence_length=sequence_length, pack=True)

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

  def test_to_text2self(self):
    features = {"inputs": tf.constant([3, 5, 6, 9, 1]),
                "targets": tf.constant([4, 6, 8, 1])}
    expected = {"targets": tf.constant([3, 5, 6, 9, 1, 4, 6, 8, 1])}
    actual = feature_converters.to_text2self(features)
    self.assertAllClose(actual, expected)

  def test_decoder_feature_converter_targets_only_unpacked(self):
    x = {"targets": [3, 9, 1]}
    ds = tf.data.Dataset.from_tensors(x)

    sequence_length = {"targets": 5}
    converted_ds = feature_converters.decoder_feature_converter(
        ds, sequence_length=sequence_length, pack=False)

    expected = {
        "decoder_target_token": [3, 9, 1, 0, 0],
        "decoder_input_token": [0, 3, 9, 1, 0],
        "decoder_loss_weight": [1, 1, 1, 0, 0],
    }
    assert_dataset(converted_ds, expected)

  def test_decoder_feature_converter_targets_only_packed(self):
    x = [{"targets": [3, 9, 1]}, {"targets": [4, 1]}]
    ds = tf.data.Dataset.from_generator(
        lambda: x, output_types={"targets": tf.int32})

    sequence_length = {"targets": 6}
    converted_ds = feature_converters.decoder_feature_converter(
        ds, sequence_length=sequence_length, pack=True)

    expected = {
        "decoder_target_token": [3, 9, 1, 4, 1, 0],
        "decoder_input_token": [0, 3, 9, 0, 4, 0],
        "decoder_loss_weight": [1, 1, 1, 1, 1, 0],
        "decoder_position": [0, 1, 2, 0, 1, 0],
        "decoder_segment_id": [1, 1, 1, 2, 2, 0]
    }
    assert_dataset(converted_ds, expected)

  def test_decoder_feature_converter_inputs_targets_unpacked(self):
    x = {"inputs": [9, 4, 6, 1], "targets": [3, 9, 1]}
    ds = tf.data.Dataset.from_tensors(x)

    sequence_length = {"targets": 9}
    converted_ds = feature_converters.decoder_feature_converter(
        ds, sequence_length=sequence_length, pack=False)

    expected = {
        "decoder_target_token": [9, 4, 6, 1, 3, 9, 1, 0, 0],
        # The last EOS token is kept if unpacked.
        "decoder_input_token": [0, 9, 4, 6, 1, 3, 9, 1, 0],
        "decoder_loss_weight": [1, 1, 1, 1, 1, 1, 1, 0, 0],
    }
    assert_dataset(converted_ds, expected)

  def test_decoder_feature_converter_inputs_targets_packed(self):
    x = [{"inputs": [7, 8, 5, 1], "targets": [3, 9, 1]},
         {"inputs": [8, 4, 9, 3, 1], "targets": [4, 1]}]
    ds = tf.data.Dataset.from_generator(
        lambda: x, output_types={"inputs": tf.int32, "targets": tf.int32})

    sequence_length = {"targets": 15}
    converted_ds = feature_converters.decoder_feature_converter(
        ds, sequence_length=sequence_length, pack=True)

    expected = {
        "decoder_target_token": [7, 8, 5, 1, 3, 9, 1, 8, 4, 9, 3, 1, 4, 1, 0],
        "decoder_input_token": [0, 7, 8, 5, 1, 3, 9, 0, 8, 4, 9, 3, 1, 4, 0],
        "decoder_loss_weight": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        "decoder_position": [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0],
        "decoder_segment_id": [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 0],
    }
    assert_dataset(converted_ds, expected)

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

  def test_decoder_lm_inputs_mask_composition(self):
    x = [{"inputs": [7, 8, 5, 1], "targets": [3, 9, 1]},
         {"inputs": [8, 4, 9, 3, 1], "targets": [4, 1]}]

    ds = tf.data.Dataset.from_generator(
        lambda: x, output_types={"inputs": tf.int32, "targets": tf.int32})

    sequence_length = {"targets": 15}
    converted_ds = feature_converters.decoder_feature_converter(
        ds, sequence_length=sequence_length, pack=True)
    actual = feature_converters.lm_inputs_mask(
        converted_ds, additional_position=True)

    expected = {
        "decoder_target_token": [7, 8, 5, 1, 3, 9, 1, 8, 4, 9, 3, 1, 4, 1, 0],
        "decoder_input_token": [0, 7, 8, 5, 1, 3, 9, 0, 8, 4, 9, 3, 1, 4, 0],
        "decoder_loss_weight": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        "decoder_position": [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0],
        "decoder_segment_id": [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 0],
        "inputs_attention_mask": [1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0]
    }
    assert_dataset(actual, expected)


if __name__ == "__main__":
  tf.test.main()
