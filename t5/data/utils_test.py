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
from absl.testing import absltest
import numpy as np
from t5.data import utils
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

tf.compat.v1.enable_eager_execution()

mock = absltest.mock


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


class UtilsTest(absltest.TestCase):

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

  def test_map_over_dataset_with_seeds(self):
    inputs = tf.data.Dataset.range(2)

    utils._NEXT_MAP_SEED = 42
    @utils.map_over_dataset(num_seeds=2)
    def test_fn(x, seeds=None):
      return x + seeds

    expected = [
        np.array([[2985944072, 3810604164], [64669036, 3548694723]]),
        np.array([[4132877645, 4228622226], [2495033825, 798765318]])
    ]
    for exp, act in zip(expected, test_fn(inputs).as_numpy_iterator()):
      np.testing.assert_array_equal(exp, act)

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


if __name__ == "__main__":
  absltest.main()
