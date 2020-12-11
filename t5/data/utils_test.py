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
"""Tests for t5.data.utils."""
from typing import Mapping, Sequence

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from t5.data import test_utils
from t5.data import utils
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

tf.compat.v1.enable_eager_execution()

mock = absltest.mock
assert_dataset = test_utils.assert_dataset


class AnyArg(object):

  def __eq__(self, var):
    return True


class LazyTfdsLoaderTest(absltest.TestCase):

  def setUp(self):
    utils.LazyTfdsLoader._MEMOIZED_BUILDERS = {}
    super().setUp()

  @mock.patch("tensorflow_datasets.builder")
  def test_builder_memoization(self, mock_tfds_builder):
    mock_tfds_builder.side_effect = (
        lambda name, data_dir: ",".join([name, data_dir or ""])
    )

    ds1 = utils.LazyTfdsLoader("ds1")
    self.assertEqual("ds1,", ds1.builder)
    self.assertEqual(1, tfds.builder.call_count)

    # Builder should be cached with same name.
    self.assertEqual("ds1,", ds1.builder)
    self.assertEqual(1, tfds.builder.call_count)

    # Same name but different data dir is a cache miss.
    ds1_dir1 = utils.LazyTfdsLoader("ds1", "dir1")
    self.assertEqual("ds1,dir1", ds1_dir1.builder)
    self.assertEqual(2, tfds.builder.call_count)
    # Same name and data dir is a cache hit.
    self.assertEqual("ds1,dir1", ds1_dir1.builder)
    self.assertEqual(2, tfds.builder.call_count)

    # Different name is a cache miss.
    ds2 = utils.LazyTfdsLoader("ds2")
    self.assertEqual("ds2,", ds2.builder)
    self.assertEqual(3, tfds.builder.call_count)

    # Different split map name is a cache hit.
    ds2 = utils.LazyTfdsLoader("ds2", split_map={"train": "validation"})
    self.assertEqual("ds2,", ds2.builder)
    self.assertEqual(3, tfds.builder.call_count)

    # Try calling everything again, order shouldn't matter.
    self.assertEqual("ds1,", ds1.builder)
    self.assertEqual("ds1,dir1", ds1_dir1.builder)
    self.assertEqual("ds2,", ds2.builder)
    self.assertEqual(3, tfds.builder.call_count)

  @mock.patch("tensorflow_datasets.load")
  def test_split_map(self, mock_tfds_load):
    seed = 0
    utils.LazyTfdsLoader._MEMOIZED_BUILDERS[("ds/c1", None)] = mock.Mock(
        info=mock.Mock(splits={
            "validation": mock.Mock(
                num_examples=420,
                file_instructions=["f1", "f2"]),
            "test": mock.Mock(
                num_examples=42,
                file_instructions=["f3"]),
        }))

    ds = utils.LazyTfdsLoader(
        "ds/c1", split_map={"train": "validation", "validation": "test"})

    # test .load()
    ds.load("train", shuffle_files=False, seed=seed)
    mock_tfds_load.assert_called_once_with(
        "ds/c1",
        split="validation",
        data_dir=None,
        shuffle_files=False,
        download=True,
        try_gcs=True,
        read_config=AnyArg())

    # test .size()
    self.assertEqual(420, ds.size(split="train"))
    self.assertEqual(42, ds.size(split="validation"))
    with self.assertRaises(KeyError):
      ds.size(split="test")

    # test .files()
    self.assertListEqual(["f1", "f2"], ds.files(split="train"))
    self.assertListEqual(["f3"], ds.files(split="validation"))
    with self.assertRaises(KeyError):
      ds.files(split="test")


class UtilsTest(parameterized.TestCase):

  def test_dict_to_tfexample(self):
    tfe = utils.dict_to_tfexample({
        "inputs": "this is an input",
        "targets": "this is a target",
        "weight": 5.0,
    })

    self.assertLen(tfe.features.feature, 3)
    self.assertEqual(tfe.features.feature["inputs"].bytes_list.value,
                     [b"this is an input"])
    self.assertEqual(tfe.features.feature["targets"].bytes_list.value,
                     [b"this is a target"])
    self.assertEqual(tfe.features.feature["weight"].float_list.value, [5.0])

  def test_stateless_shuffle(self):
    value = np.arange(6)
    expected_output_1 = np.array([0, 3, 4, 2, 1, 5])
    expected_output_2 = np.array([3, 4, 0, 2, 5, 1])
    np.testing.assert_array_equal(
        utils.stateless_shuffle(value, (0, 1)),
        expected_output_1)
    np.testing.assert_array_equal(
        utils.stateless_shuffle(value.reshape((2, 3)), (0, 1)),
        expected_output_1.reshape((2, 3)))
    np.testing.assert_array_equal(
        utils.stateless_shuffle(value, (2, 3)),
        expected_output_2)

  def test_map_over_dataset(self):
    inputs = tf.data.Dataset.range(5)

    @utils.map_over_dataset
    def test_fn(x):
      return x + 1

    self.assertEqual(list(test_fn(inputs).as_numpy_iterator()), [1, 2, 3, 4, 5])

  # We disable no-value-for-parameter since the utils.map_over_dataset leads to
  # a false positive when seeds are provided.
  # pylint:disable=no-value-for-parameter

  def test_map_over_dataset_with_one_seed(self):
    inputs = tf.data.Dataset.range(2)

    utils._NEXT_MAP_SEED = 42
    @utils.map_over_dataset(num_seeds=1)
    def test_fn(x, seed):
      return x + seed

    expected = [
        np.array([2985944072, 3810604164]),
        np.array([4132877645, 4228622226])
    ]
    for exp, act in zip(expected, test_fn(inputs).as_numpy_iterator()):
      np.testing.assert_array_equal(exp, act)

  def test_map_over_dataset_with_seeds(self):
    inputs = tf.data.Dataset.range(2)

    utils._NEXT_MAP_SEED = 42
    @utils.map_over_dataset(num_seeds=2)
    def test_fn(x, seeds):
      return x + seeds

    expected = [
        np.array([[2985944072, 3810604164], [64669036, 3548694723]]),
        np.array([[4132877645, 4228622226], [2495033825, 798765318]])
    ]
    for exp, act in zip(expected, test_fn(inputs).as_numpy_iterator()):
      np.testing.assert_array_equal(exp, act)

  # pylint:enable=no-value-for-parameter

  def test_map_seed_manager(self):
    utils._NEXT_MAP_SEED = None
    self.assertIsNone(utils._NEXT_MAP_SEED)
    with utils.map_seed_manager(42):
      self.assertEqual(utils._NEXT_MAP_SEED, 42)
      with utils.map_seed_manager(410):
        self.assertEqual(utils._NEXT_MAP_SEED, 410)
        utils._NEXT_MAP_SEED += 10
        self.assertEqual(utils._NEXT_MAP_SEED, 420)
      utils._NEXT_MAP_SEED += 10
      self.assertEqual(utils._NEXT_MAP_SEED, 52)
    self.assertIsNone(utils._NEXT_MAP_SEED)

  def test_trim_and_pad_dataset(self):
    x = [{"inputs": [7, 8, 5, 6, 1], "targets": [3, 9, 1], "idx": [0]},
         {"inputs": [8, 4, 9, 3, 5, 7, 9, 1], "targets": [4, 1], "idx": [1, 2]}]
    ds = create_default_dataset(x, feature_names=("inputs", "targets", "idx"))
    padded_ds = utils.trim_and_pad_dataset(
        ds,
        feature_lengths={"inputs": 7, "targets": 3})
    expected = [
        {
            "inputs": [7, 8, 5, 6, 1, 0, 0],
            "targets": [3, 9, 1],
            "idx": [0],
        },
        {
            # EOS is trimmed
            "inputs": [8, 4, 9, 3, 5, 7, 9],
            "targets": [4, 1, 0],
            "idx": [1, 2],
        }
    ]
    assert_dataset(
        padded_ds, expected, {"inputs": tf.int32, "targets": tf.int32})

  _PACK_PARAMETERS = ({"use_custom_ops": False},)

  @parameterized.parameters(*_PACK_PARAMETERS)
  def test_trim_and_pack_dataset(self, use_custom_ops):
    x = [{"inputs": [7, 8, 5, 1], "targets": [3, 9, 1], "idx": [0]},
         {"inputs": [8, 4, 9, 3, 1], "targets": [4, 1], "idx": [1]}]
    ds = create_default_dataset(x, feature_names=("inputs", "targets", "idx"))
    packed_ds = utils.trim_and_pack_dataset(
        ds,
        feature_lengths={"inputs": 10, "targets": 7},
        use_custom_ops=use_custom_ops)

    expected = {
        "inputs": [7, 8, 5, 1, 8, 4, 9, 3, 1, 0],
        "inputs_segment_ids": [1, 1, 1, 1, 2, 2, 2, 2, 2, 0],
        "inputs_positions": [0, 1, 2, 3, 0, 1, 2, 3, 4, 0],
        "targets": [3, 9, 1, 4, 1, 0, 0],
        "targets_positions": [0, 1, 2, 0, 1, 0, 0],
        "targets_segment_ids": [1, 1, 1, 2, 2, 0, 0],
    }
    assert_dataset(
        packed_ds, expected, {"inputs": tf.int32, "targets": tf.int32})

  @parameterized.parameters(*_PACK_PARAMETERS)
  def test_trim_and_pack_dataset_no_eos(self, use_custom_ops):
    x = [{"inputs": [7, 8, 5], "targets": [3, 9]},
         {"inputs": [8, 4, 9, 3], "targets": [4]}]
    ds = create_default_dataset(x)
    packed_ds = utils.trim_and_pack_dataset(
        ds,
        feature_lengths={"inputs": 8, "targets": 5},
        use_custom_ops=use_custom_ops)

    # Packing still works without the eos.
    expected = {
        "inputs": [7, 8, 5, 8, 4, 9, 3, 0],
        "inputs_segment_ids": [1, 1, 1, 2, 2, 2, 2, 0],
        "inputs_positions": [0, 1, 2, 0, 1, 2, 3, 0],
        "targets": [3, 9, 4, 0, 0],
        "targets_positions": [0, 1, 0, 0, 0],
        "targets_segment_ids": [1, 1, 2, 0, 0],
    }
    assert_dataset(
        packed_ds, expected, {"inputs": tf.int32, "targets": tf.int32})

  @parameterized.parameters(*_PACK_PARAMETERS)
  def test_trim_and_pack_dataset_long_seq(self, use_custom_ops):
    x = [{"inputs": [7, 8, 5, 6, 9, 4, 1], "targets": [3, 9, 1]},
         {"inputs": [8, 4, 9, 3, 5, 7, 9, 1], "targets": [4, 1]}]
    ds = create_default_dataset(x)
    packed_ds = utils.trim_and_pack_dataset(
        ds,
        feature_lengths={"inputs": 7, "targets": 3},
        use_custom_ops=use_custom_ops)
    expected = [{
        "inputs": [7, 8, 5, 6, 9, 4, 1],
        "inputs_segment_ids": [1, 1, 1, 1, 1, 1, 1],
        "inputs_positions": [0, 1, 2, 3, 4, 5, 6],
        "targets": [3, 9, 1],
        "targets_positions": [0, 1, 2],
        "targets_segment_ids": [1, 1, 1],
    }, {
        # EOS is trimmed
        "inputs": [8, 4, 9, 3, 5, 7, 9],
        "inputs_segment_ids": [1, 1, 1, 1, 1, 1, 1],
        "inputs_positions": [0, 1, 2, 3, 4, 5, 6],
        "targets": [4, 1, 0],
        "targets_positions": [0, 1, 0],
        "targets_segment_ids": [1, 1, 0],
    }]
    assert_dataset(
        packed_ds, expected, {"inputs": tf.int32, "targets": tf.int32})


def create_default_dataset(
    x: Sequence[Mapping[str, int]],
    feature_names: Sequence[str] = ("inputs", "targets")) -> tf.data.Dataset:
  output_types = {feature_name: tf.int32 for feature_name in feature_names}
  output_shapes = {feature_name: [None] for feature_name in feature_names}

  ds = tf.data.Dataset.from_generator(
      lambda: x, output_types=output_types, output_shapes=output_shapes)
  return ds

if __name__ == "__main__":
  absltest.main()
