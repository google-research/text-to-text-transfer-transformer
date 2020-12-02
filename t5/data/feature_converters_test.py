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
from t5.data import dataset_providers
from t5.data import feature_converters
from t5.data import test_utils
import tensorflow.compat.v2 as tf

tf.compat.v1.enable_eager_execution()

assert_dataset = test_utils.assert_dataset


def create_default_dataset(
    x: Sequence[Mapping[str, int]],
    feature_names: Sequence[str] = ("inputs", "targets"),
    output_types: Mapping[str, tf.dtypes.DType] = None) -> tf.data.Dataset:
  if output_types is None:
    output_types = {feature_name: tf.int64 for feature_name in feature_names}

  output_shapes = {feature_name: [None] for feature_name in feature_names}
  ds = tf.data.Dataset.from_generator(
      lambda: x, output_types=output_types, output_shapes=output_shapes)
  return ds


class FeatureConvertersTest(tf.test.TestCase):

  def test_validate_dataset_missing_feature(self):
    x = [{"targets": [3, 9, 4, 5]}]
    ds = create_default_dataset(x, feature_names=["targets"])
    input_lengths = {"inputs": 4, "targets": 4}

    with mock.patch.object(feature_converters.FeatureConverter,
                           "__abstractmethods__", set()):
      converter = feature_converters.FeatureConverter()
      expected_msg = ("Dataset is missing an expected feature during "
                      "initial validation: 'inputs'")
      with self.assertRaisesRegex(ValueError, expected_msg):
        converter._validate_dataset(
            ds,
            expected_features=input_lengths.keys(),
            expected_dtypes={"inputs": tf.int64, "targets": tf.int64},
            expected_lengths=input_lengths,
            strict=False,
            error_label="initial")

  def test_validate_dataset_incorrect_dtype(self):
    x = [{"inputs": [9, 4, 3, 8, 6], "targets": [3, 9, 4, 5]}]
    input_dtypes = {"inputs": tf.int32, "targets": tf.int64}
    ds = tf.data.Dataset.from_generator(
        lambda: x, output_types=input_dtypes,
        output_shapes={"inputs": [None], "targets": [None]})
    input_lengths = {"inputs": 5, "targets": 4}

    with mock.patch.object(feature_converters.FeatureConverter,
                           "__abstractmethods__", set()):
      converter = feature_converters.FeatureConverter(input_dtypes=input_dtypes)
      expected_msg = ("Dataset has incorrect type for feature 'inputs' during "
                      "initial validation: Got int32, expected int64")
      with self.assertRaisesRegex(ValueError, expected_msg):
        converter._validate_dataset(
            ds,
            expected_features=input_lengths.keys(),
            expected_dtypes={"inputs": tf.int64, "targets": tf.int64},
            expected_lengths=input_lengths,
            strict=False,
            error_label="initial")

  def test_validate_dataset_incorrect_rank(self):
    x = [{"inputs": [[9, 4, 3, 8, 6]], "targets": [3, 9, 4, 5]}]
    ds = tf.data.Dataset.from_generator(
        lambda: x, output_types={"inputs": tf.int64, "targets": tf.int64},
        output_shapes={"inputs": [None, 1], "targets": [None]})
    input_lengths = {"inputs": 5, "targets": 4}

    with mock.patch.object(feature_converters.FeatureConverter,
                           "__abstractmethods__", set()):
      converter = feature_converters.FeatureConverter()
      expected_msg = ("Dataset has incorrect rank for feature 'inputs' during "
                      "initial validation: Got 2, expected 1")
      with self.assertRaisesRegex(ValueError, expected_msg):
        converter._validate_dataset(
            ds,
            expected_features=input_lengths.keys(),
            expected_dtypes={"inputs": tf.int64, "targets": tf.int64},
            expected_lengths=input_lengths,
            strict=False,
            expected_rank=1,
            error_label="initial")

  def test_validate_dataset_missing_length(self):
    x = [{"inputs": [9, 4, 3, 8, 6], "targets": [3, 9, 4, 5]}]
    ds = tf.data.Dataset.from_generator(
        lambda: x, output_types={"inputs": tf.int64, "targets": tf.int64},
        output_shapes={"inputs": [5], "targets": [5]})
    input_lengths = {"inputs": 5}

    with mock.patch.object(feature_converters.FeatureConverter,
                           "__abstractmethods__", set()):
      converter = feature_converters.FeatureConverter()
      expected_msg = ("Sequence length for feature 'targets' is missing "
                      "during final validation")
      with self.assertRaisesRegex(ValueError, expected_msg):
        converter._validate_dataset(
            ds,
            expected_features=["inputs", "targets"],
            expected_dtypes={"inputs": tf.int64, "targets": tf.int64},
            expected_lengths=input_lengths,
            strict=True,
            error_label="final")

  def test_validate_dataset_plaintext_field(self):
    x = [{"targets": [3, 9, 4, 5], "targets_plaintext": "some text"}]
    output_types = {"targets": tf.int64, "targets_plaintext": tf.string}
    output_shapes = {"targets": [4], "targets_plaintext": []}
    ds = tf.data.Dataset.from_generator(
        lambda: x, output_types=output_types, output_shapes=output_shapes)

    input_lengths = {"targets": 4}
    with mock.patch.object(feature_converters.FeatureConverter,
                           "__abstractmethods__", set()):
      converter = feature_converters.FeatureConverter()
      # _validate_dataset works even if ds has targets and targets_plaintext
      ds = converter._validate_dataset(
          ds,
          expected_features=input_lengths.keys(),
          expected_dtypes={"targets": tf.int64},
          expected_lengths=input_lengths,
          strict=True,
          error_label="initial")

  def test_check_lengths_strict_no_exception(self):
    x = [{"inputs": [9, 4, 3, 8, 1], "targets": [3, 9, 4, 5]}]
    ds = create_default_dataset(x)
    input_lengths = {"inputs": 5, "targets": 4}
    ds = feature_converters._check_lengths(
        ds, input_lengths, strict=True, error_label="initial")
    list(ds.as_numpy_iterator())

  def test_check_lengths_strict_exception(self):
    x = [{"inputs": [9, 4, 3, 8, 1], "targets": [3, 9, 4, 5]}]
    ds = create_default_dataset(x)
    input_lengths = {"inputs": 7, "targets": 4}
    expected_msg = (
        r".*Feature \\'inputs\\' has length not equal to the expected length of"
        r" 7 during initial validation.*")
    with self.assertRaisesRegex(tf.errors.InvalidArgumentError, expected_msg):
      ds = feature_converters._check_lengths(
          ds, input_lengths, strict=True, error_label="initial")
      list(ds.as_numpy_iterator())

  def test_check_lengths_not_strict_no_exception(self):
    x = [{"inputs": [9, 4, 3, 8, 1], "targets": [3, 9, 4, 5]}]
    ds = create_default_dataset(x)
    input_lengths = {"inputs": 7, "targets": 4}
    ds = feature_converters._check_lengths(
        ds, input_lengths, strict=False, error_label="initial")
    list(ds.as_numpy_iterator())

  def test_check_lengths_not_strict_exception(self):
    x = [{"inputs": [9, 4, 3, 8, 1], "targets": [3, 9, 4, 5]}]
    ds = create_default_dataset(x)
    input_lengths = {"inputs": 4, "targets": 4}
    expected_msg = (
        r".*Feature \\'inputs\\' has length not less than or equal to the "
        r"expected length of 4 during initial validation.*")
    with self.assertRaisesRegex(tf.errors.InvalidArgumentError, expected_msg):
      ds = feature_converters._check_lengths(
          ds, input_lengths, strict=False, error_label="initial")
      list(ds.as_numpy_iterator())

  def test_check_lengths_extra_features(self):
    x = [{"targets": [3, 9, 4, 5], "targets_plaintext": "some text"}]
    output_types = {"targets": tf.int64, "targets_plaintext": tf.string}
    output_shapes = {"targets": [4], "targets_plaintext": []}
    ds = tf.data.Dataset.from_generator(
        lambda: x, output_types=output_types, output_shapes=output_shapes)
    input_lengths = {"targets": 4}
    ds = feature_converters._check_lengths(
        ds, input_lengths, strict=True, error_label="initial")
    list(ds.as_numpy_iterator())


if __name__ == "__main__":
  tf.test.main()
