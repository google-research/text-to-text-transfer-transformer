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

from typing import Mapping, Sequence
from unittest import mock
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

  def test_autoregressive_inputs_different_dtypes(self):
    x = tf.constant([3, 8, 1, 9, 1, 5, 4, 1, 0, 0])
    sequence_id = tf.constant([1, 1, 1, 2, 2, 3, 3, 3, 0, 0], tf.int32)
    autoreg_inputs = feature_converters.autoregressive_inputs(
        x, sequence_id=sequence_id, output_dtype=tf.int64)
    self.assertEqual(autoreg_inputs.dtype, tf.int64)

  def test_check_lengths_strict_no_exception(self):
    x = [{"inputs": [9, 4, 3, 8, 1], "targets": [3, 9, 4, 5]}]
    ds = create_default_dataset(x)
    task_feature_lengths = {"inputs": 5, "targets": 4}
    ds = feature_converters._check_lengths(
        ds, task_feature_lengths, strict=True, error_label="initial")
    list(ds.as_numpy_iterator())

  def test_check_lengths_strict_exception(self):
    x = [{"inputs": [9, 4, 3, 8, 1], "targets": [3, 9, 4, 5]}]
    ds = create_default_dataset(x)
    task_feature_lengths = {"inputs": 7, "targets": 4}
    expected_msg = (
        r".*Feature \\'inputs\\' has length not equal to the expected length of"
        r" 7 during initial validation.*")
    with self.assertRaisesRegex(tf.errors.InvalidArgumentError, expected_msg):
      ds = feature_converters._check_lengths(
          ds, task_feature_lengths, strict=True, error_label="initial")
      list(ds.as_numpy_iterator())

  def test_check_lengths_not_strict_no_exception(self):
    x = [{"inputs": [9, 4, 3, 8, 1], "targets": [3, 9, 4, 5]}]
    ds = create_default_dataset(x)
    task_feature_lengths = {"inputs": 7, "targets": 4}
    ds = feature_converters._check_lengths(
        ds, task_feature_lengths, strict=False, error_label="initial")
    list(ds.as_numpy_iterator())

  def test_check_lengths_not_strict_exception(self):
    x = [{"inputs": [9, 4, 3, 8, 1], "targets": [3, 9, 4, 5]}]
    ds = create_default_dataset(x)
    task_feature_lengths = {"inputs": 4, "targets": 4}
    expected_msg = (
        r".*Feature \\'inputs\\' has length not less than or equal to the "
        r"expected length of 4 during initial validation.*")
    with self.assertRaisesRegex(tf.errors.InvalidArgumentError, expected_msg):
      ds = feature_converters._check_lengths(
          ds, task_feature_lengths, strict=False, error_label="initial")
      list(ds.as_numpy_iterator())

  def test_check_lengths_extra_features(self):
    x = [{"targets": [3, 9, 4, 5], "targets_plaintext": "some text"}]
    output_types = {"targets": tf.int64, "targets_plaintext": tf.string}
    output_shapes = {"targets": [4], "targets_plaintext": []}
    ds = tf.data.Dataset.from_generator(
        lambda: x, output_types=output_types, output_shapes=output_shapes)
    task_feature_lengths = {"targets": 4}
    ds = feature_converters._check_lengths(
        ds, task_feature_lengths, strict=True, error_label="initial")
    list(ds.as_numpy_iterator())

  def test_check_exact_match_redundant_features(self):
    extra = ["inputs", "random"]
    expected_msg = (
        "The input_dataset contains extra features not specified in the "
        f"task_feature_dtypes: {set(extra)}")
    with self.assertRaisesRegex(ValueError, expected_msg):
      feature_converters._check_exact_match(
          expected_features=["targets"],
          actual_features=["inputs", "targets", "random"],
          expected_feature_source="task_feature_dtypes",
          actual_feature_source="input_dataset")

  def test_check_exact_match_missing_features(self):
    missing = ["inputs", "random"]
    expected_msg = (
        "The input_dataset is missing features specified in the "
        f"task_feature_dtypes: {set(missing)}")
    with self.assertRaisesRegex(ValueError, expected_msg):
      feature_converters._check_exact_match(
          expected_features=["inputs", "targets", "random"],
          actual_features=["targets"],
          expected_feature_source="task_feature_dtypes",
          actual_feature_source="input_dataset")


def create_default_dataset(
    x: Sequence[Mapping[str, int]],
    feature_names: Sequence[str] = ("inputs", "targets"),
    output_types: Mapping[str, tf.dtypes.DType] = None) -> tf.data.Dataset:
  if output_types is None:
    output_types = {feature_name: tf.int32 for feature_name in feature_names}

  output_shapes = {feature_name: [None] for feature_name in feature_names}
  ds = tf.data.Dataset.from_generator(
      lambda: x, output_types=output_types, output_shapes=output_shapes)
  return ds


class FeatureConvertersTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    feature_converters.FeatureConverter.TASK_FEATURE_DTYPES = {}
    feature_converters.FeatureConverter.MODEL_FEATURE_DTYPES = {}
    feature_converters.FeatureConverter.PACKING_FEATURE_DTYPES = {}

  def tearDown(self):
    del feature_converters.FeatureConverter.TASK_FEATURE_DTYPES
    del feature_converters.FeatureConverter.MODEL_FEATURE_DTYPES
    del feature_converters.FeatureConverter.PACKING_FEATURE_DTYPES
    super().tearDown()

  def test_validate_dataset_missing_feature(self):
    x = [{"targets": [3, 9, 4, 5]}]
    ds = create_default_dataset(x, feature_names=["targets"])
    task_feature_lengths = {"inputs": 4, "targets": 4}

    with mock.patch.object(feature_converters.FeatureConverter,
                           "__abstractmethods__", set()):
      converter = feature_converters.FeatureConverter()
      expected_msg = ("Dataset is missing an expected feature during "
                      "initial validation: 'inputs'")
      with self.assertRaisesRegex(ValueError, expected_msg):
        converter._validate_dataset(
            ds,
            expected_features=task_feature_lengths.keys(),
            expected_dtypes={"inputs": tf.int32, "targets": tf.int32},
            expected_lengths=task_feature_lengths,
            strict=False,
            error_label="initial")

  def test_validate_dataset_incorrect_dtype(self):
    x = [{"inputs": [9, 4, 3, 8, 6], "targets": [3, 9, 4, 5]}]
    task_feature_dtypes = {"inputs": tf.int32, "targets": tf.int64}
    ds = tf.data.Dataset.from_generator(
        lambda: x, output_types=task_feature_dtypes,
        output_shapes={"inputs": [None], "targets": [None]})
    task_feature_lengths = {"inputs": 5, "targets": 4}

    with mock.patch.object(feature_converters.FeatureConverter,
                           "__abstractmethods__", set()):
      feature_converters.FeatureConverter.TASK_FEATURE_DTYPES = (
          task_feature_dtypes)
      converter = feature_converters.FeatureConverter()
      expected_msg = ("Dataset has incorrect type for feature 'inputs' during "
                      "initial validation: Got int32, expected int64")
      with self.assertRaisesRegex(ValueError, expected_msg):
        converter._validate_dataset(
            ds,
            expected_features=task_feature_lengths.keys(),
            expected_dtypes={"inputs": tf.int64, "targets": tf.int64},
            expected_lengths=task_feature_lengths,
            strict=False,
            error_label="initial")

  def test_validate_dataset_incorrect_rank(self):
    x = [{"inputs": [[9, 4, 3, 8, 6]], "targets": [3, 9, 4, 5]}]
    ds = tf.data.Dataset.from_generator(
        lambda: x, output_types={"inputs": tf.int64, "targets": tf.int64},
        output_shapes={"inputs": [None, 1], "targets": [None]})
    task_feature_lengths = {"inputs": 5, "targets": 4}

    with mock.patch.object(feature_converters.FeatureConverter,
                           "__abstractmethods__", set()):
      converter = feature_converters.FeatureConverter()
      expected_msg = ("Dataset has incorrect rank for feature 'inputs' during "
                      "initial validation: Got 2, expected 1")
      with self.assertRaisesRegex(ValueError, expected_msg):
        converter._validate_dataset(
            ds,
            expected_features=task_feature_lengths.keys(),
            expected_dtypes={"inputs": tf.int64, "targets": tf.int64},
            expected_lengths=task_feature_lengths,
            strict=False,
            expected_rank=1,
            error_label="initial")

  def test_call_missing_input_lengths(self):
    x = [{"inputs": [9, 4, 3, 8, 6], "targets": [3, 9, 4, 5]}]
    ds = tf.data.Dataset.from_generator(
        lambda: x, output_types={"inputs": tf.int64, "targets": tf.int64},
        output_shapes={"inputs": [5], "targets": [5]})
    task_feature_lengths = {"inputs": 5}

    with mock.patch.object(feature_converters.FeatureConverter,
                           "__abstractmethods__", set()):
      converter = feature_converters.FeatureConverter()
      feature_converters.FeatureConverter.TASK_FEATURE_DTYPES = {
          "inputs": tf.int64,
          "targets": tf.int64
      }
      expected_msg = ("The task_feature_lengths is missing features specified "
                      "in the TASK_FEATURE_DTYPES: {'targets'}")
      with self.assertRaisesRegex(ValueError, expected_msg):
        converter(ds, task_feature_lengths)

  def test_validate_dataset_plaintext_field(self):
    x = [{"targets": [3, 9, 4, 5], "targets_plaintext": "some text"}]
    output_types = {"targets": tf.int64, "targets_plaintext": tf.string}
    output_shapes = {"targets": [4], "targets_plaintext": []}
    ds = tf.data.Dataset.from_generator(
        lambda: x, output_types=output_types, output_shapes=output_shapes)

    task_feature_lengths = {"targets": 4}
    with mock.patch.object(feature_converters.FeatureConverter,
                           "__abstractmethods__", set()):
      converter = feature_converters.FeatureConverter()
      # _validate_dataset works even if ds has targets and targets_plaintext
      ds = converter._validate_dataset(
          ds,
          expected_features=task_feature_lengths.keys(),
          expected_dtypes={"targets": tf.int64},
          expected_lengths=task_feature_lengths,
          strict=True,
          error_label="initial")


class EncDecFeatureConverterTest(tf.test.TestCase):

  # def setUp(self):
  #   super().setUp()
  #   feature_converters.EncDecFeatureConverter._task_feature_dtypes = {
  #       "inputs": tf.int32,
  #       "targets": tf.int32
  #   }
  #   feature_converters.EncDecFeatureConverter._model_feature_dtypes = {
  #       "encoder_input_token": tf.int32,
  #       "decoder_target_token": tf.int32,
  #       "decoder_input_token": tf.int32,
  #       "decoder_loss_weight": tf.int32,
  #   }

  def test_encoder_decoder_unpacked(self):
    x = [{"inputs": [9, 4, 3, 8, 1], "targets": [3, 9, 4, 1]}]
    ds = create_default_dataset(x)
    task_feature_lengths = {"inputs": 7, "targets": 5}

    converter = feature_converters.EncDecFeatureConverter(pack=False)
    converted_ds = converter(ds, task_feature_lengths)

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
    x = [{"inputs": [9, 4, 3, 8, 1], "targets": [3, 9, 4, 5, 1]}]
    ds = create_default_dataset(x)
    task_feature_lengths = {"inputs": 5, "targets": 5}

    converter = feature_converters.EncDecFeatureConverter(pack=False)
    converted_ds = converter(ds, task_feature_lengths)

    expected = {
        "encoder_input_token": [9, 4, 3, 8, 1],
        "decoder_target_token": [3, 9, 4, 5, 1],
        "decoder_input_token": [0, 3, 9, 4, 5],
        "decoder_loss_weight": [1, 1, 1, 1, 1],
    }
    assert_dataset(converted_ds, expected)

  def test_encoder_decoder_extra_long_inputs(self):
    x = [{"inputs": [9, 4, 3, 8, 4, 5, 1], "targets": [3, 9, 4, 7, 8, 1]}]
    ds = create_default_dataset(x)
    task_feature_lengths = {"inputs": 5, "targets": 8}
    expected_msg = (
        r".*Feature \\'inputs\\' has length not less than or equal to the "
        r"expected length of 5 during input_validation.*")
    with self.assertRaisesRegex(tf.errors.InvalidArgumentError, expected_msg):
      converter = feature_converters.EncDecFeatureConverter(pack=False)
      converted_ds = converter(ds, task_feature_lengths)
      list(converted_ds.as_numpy_iterator())

  def test_encoder_decoder_packed(self):
    x = [{"inputs": [7, 8, 5, 1], "targets": [3, 9, 1]},
         {"inputs": [8, 4, 9, 3, 1], "targets": [4, 1]}]
    ds = create_default_dataset(x)
    task_feature_lengths = {"inputs": 10, "targets": 7}

    converter = feature_converters.EncDecFeatureConverter(pack=True)
    converted_ds = converter(ds, task_feature_lengths)
    expected = {
        "encoder_input_token": [7, 8, 5, 1, 8, 4, 9, 3, 1, 0],
        "encoder_segment_id": [1, 1, 1, 1, 2, 2, 2, 2, 2, 0],
        "decoder_target_token": [3, 9, 1, 4, 1, 0, 0],
        "decoder_input_token": [0, 3, 9, 0, 4, 0, 0],
        "decoder_loss_weight": [1, 1, 1, 1, 1, 0, 0],
        "decoder_segment_id": [1, 1, 1, 2, 2, 0, 0],
    }
    assert_dataset(converted_ds, expected)

  def test_encoder_decoder_packed_long_sequences(self):
    x = [{"inputs": [7, 8, 5, 6, 9, 4, 1], "targets": [3, 9, 1]},
         {"inputs": [8, 4, 9, 3, 5, 1], "targets": [4, 1]}]
    ds = create_default_dataset(x)
    task_feature_lengths = {"inputs": 7, "targets": 3}

    converter = feature_converters.EncDecFeatureConverter(pack=True)
    converted_ds = converter(ds, task_feature_lengths)

    # Corner case: packing is true but task_feature_lengths are too long for
    # packing to happen. We should still get the *_segment_id, *_position
    # fields.
    expected = [{
        "encoder_input_token": [7, 8, 5, 6, 9, 4, 1],
        "encoder_segment_id": [1, 1, 1, 1, 1, 1, 1],
        "decoder_target_token": [3, 9, 1],
        "decoder_input_token": [0, 3, 9],
        "decoder_loss_weight": [1, 1, 1],
        "decoder_segment_id": [1, 1, 1],
    }, {
        "encoder_input_token": [8, 4, 9, 3, 5, 1, 0],
        "encoder_segment_id": [1, 1, 1, 1, 1, 1, 0],
        "decoder_target_token": [4, 1, 0],
        "decoder_input_token": [0, 4, 0],
        "decoder_loss_weight": [1, 1, 0],
        "decoder_segment_id": [1, 1, 0],
    }]
    assert_dataset(converted_ds, expected)

  def test_encoder_decoder_plaintext_field(self):
    x = [{
        "inputs": [7, 8, 5, 1],
        "targets": [3, 9, 1],
        "targets_plaintext": "abc"
    }, {
        "inputs": [8, 4, 9, 3, 1],
        "targets": [4, 1],
        "targets_plaintext": "def"
    }]
    types = {
        "inputs": tf.int32,
        "targets": tf.int32,
        "targets_plaintext": tf.string
    }
    shapes = {"inputs": [None], "targets": [None], "targets_plaintext": []}
    ds = tf.data.Dataset.from_generator(
        lambda: x, output_types=types, output_shapes=shapes)

    task_feature_lengths = {"inputs": 10, "targets": 7}
    converter = feature_converters.EncDecFeatureConverter(pack=True)
    # Check whether convert_features raise error because targets_plaintext is
    # present in the ds but not in the task_feature_lengths
    converter(ds, task_feature_lengths)


if __name__ == "__main__":
  tf.test.main()
