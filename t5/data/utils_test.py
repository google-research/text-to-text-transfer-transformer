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
from t5.data import utils
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds

tf.disable_v2_behavior()
tf.enable_eager_execution()

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


if __name__ == "__main__":
  absltest.main()
