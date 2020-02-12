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

import os

from absl.testing import absltest
import numpy as np
from t5.data import test_utils
from t5.data import utils
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds

tf.disable_v2_behavior()
tf.enable_eager_execution()

TaskRegistry = utils.TaskRegistry
mock = absltest.mock


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
    ds.load("train", shuffle_files=False)
    mock_tfds_load.assert_called_once_with(
        "ds/c1",
        split="validation",
        data_dir=None,
        shuffle_files=False,
        download=True,
        try_gcs=True)

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


class TasksTest(test_utils.FakeTaskTest):

  def test_invalid_name(self):
    with self.assertRaisesRegex(
        ValueError,
        "Task name 'invalid/name' contains invalid characters. "
        "Must match regex: .*"):
      test_utils.add_tfds_task("invalid/name")

  def test_repeat_name(self):
    with self.assertRaisesRegex(
        ValueError, "Attempting to register duplicate provider: cached_task"):
      test_utils.add_tfds_task("cached_task")

  def test_dataset_fn_signature(self):
    # Good signatures.
    def good_fn(split, shuffle_files):
      del split
      del shuffle_files
    test_utils.add_task("good_fn", good_fn)

    def default_good_fn(split, shuffle_files=False):
      del split
      del shuffle_files
    test_utils.add_task("default_good_fn", default_good_fn)

    def extra_kwarg_good_fn(split, shuffle_files, unused_kwarg=True):
      del split
      del shuffle_files
    test_utils.add_task("extra_kwarg_good_fn", extra_kwarg_good_fn)

    # Bad signatures.
    with self.assertRaisesRegex(
        ValueError,
        r"'missing_shuff' must have positional args \('split', "
        r"'shuffle_files'\), got: \('split',\)"):
      def missing_shuff(split):
        del split
      test_utils.add_task("fake_task", missing_shuff)

    with self.assertRaisesRegex(
        ValueError,
        r"'missing_split' must have positional args \('split', "
        r"'shuffle_files'\), got: \('shuffle_files',\)"):
      def missing_split(shuffle_files):
        del shuffle_files
      test_utils.add_task("fake_task", missing_split)

    with self.assertRaisesRegex(
        ValueError,
        r"'extra_pos_arg' may only have positional args \('split', "
        r"'shuffle_files'\), got: \('split', 'shuffle_files', 'unused_arg'\)"):
      def extra_pos_arg(split, shuffle_files, unused_arg):
        del split
        del shuffle_files
      test_utils.add_task("fake_task", extra_pos_arg)

  def test_dataset_fn(self):
    test_utils.add_task("fn_task", test_utils.get_fake_dataset)
    fn_task = TaskRegistry.get("fn_task")
    test_utils.verify_task_matches_fake_datasets(fn_task, use_cached=False)

  def test_text_line_task(self):
    test_utils.verify_task_matches_fake_datasets(
        self.text_line_task, use_cached=False, splits=["train"])

  def test_no_tfds_version(self):
    with self.assertRaisesRegex(
        ValueError, "TFDS name must contain a version number, got: fake"):
      test_utils.add_tfds_task("fake_task", tfds_name="fake")

  def test_num_input_examples(self):
    self.assertEqual(30, self.cached_task.num_input_examples("train"))
    self.assertEqual(10, self.cached_task.num_input_examples("validation"))

  def test_cache_exists(self):
    self.assertTrue(self.cached_task.cache_dir)
    self.cached_task.assert_cached()
    self.assertEqual(
        os.path.join(self.test_data_dir, "cached_task"),
        self.cached_task.cache_dir)

    self.assertFalse(self.uncached_task.cache_dir)
    with self.assertRaisesRegex(  # pylint:disable=g-error-prone-assert-raises
        AssertionError,
        "'uncached_task' does not exist in any of the task cache directories"):
      self.uncached_task.assert_cached()

  def test_get_cached_stats(self):
    expected_train_stats = {
        "examples": 3, "inputs_tokens": 36, "targets_tokens": 18}
    self.assertEqual(
        expected_train_stats,
        self.cached_task.get_cached_stats("train"))
    # Check repeated call.
    self.assertEqual(
        expected_train_stats,
        self.cached_task.get_cached_stats("train"))
    expected_validation_stats = {
        "examples": 2, "inputs_tokens": 23, "targets_tokens": 36}
    self.assertEqual(
        expected_validation_stats,
        self.cached_task.get_cached_stats("validation"))
    with self.assertRaisesRegex(
        ValueError, "Stats do not exist for 'cached_task' split: fake"):
      self.cached_task.get_cached_stats("fake")
    with self.assertRaisesRegex(
        AssertionError,
        "'uncached_task' does not exist in any of the task cache directories"):
      self.uncached_task.get_cached_stats("train")

  def test_set_global_cache_dirs(self):
    utils.set_global_cache_dirs([])
    self.assertFalse(self.cached_task.cache_dir)

    utils.set_global_cache_dirs([self.test_data_dir])
    self.assertTrue(self.cached_task.cache_dir)

  def test_get_dataset_cached(self):
    test_utils.verify_task_matches_fake_datasets(
        self.cached_task, use_cached=True)

    # Test with token preprocessor.
    self.cached_task._token_preprocessor = test_utils.test_token_preprocessor
    test_utils.verify_task_matches_fake_datasets(
        self.cached_task, use_cached=False, token_preprocessed=True)

  def test_get_dataset_onthefly(self):
    test_utils.verify_task_matches_fake_datasets(
        self.uncached_task, use_cached=False)

    # Test with token preprocessor.
    self.uncached_task._token_preprocessor = test_utils.test_token_preprocessor
    test_utils.verify_task_matches_fake_datasets(
        self.uncached_task, use_cached=False, token_preprocessed=True)

    # Override mock to get more examples.
    def fake_load(s, shuffle_files=False):
      del shuffle_files  # Unused, to mimic TFDS API
      return test_utils.get_fake_dataset(s).repeat().take(20)
    self._tfds_patcher.new.return_value = (
        self._tfds_patcher.new.return_value._replace(load=fake_load))

  def test_invalid_text_preprocessors(self):
    def _dummy_preprocessor(output):
      return lambda _: tf.data.Dataset.from_tensors(output)

    test_utils.add_tfds_task(
        "text_prep_ok",
        text_preprocessor=_dummy_preprocessor(
            {"inputs": "a", "targets": "b", "other": [0]}))
    TaskRegistry.get_dataset(
        "text_prep_ok", {"inputs": 13, "targets": 13},
        "train", use_cached=False)

    test_utils.add_tfds_task(
        "text_prep_missing_feature",
        text_preprocessor=_dummy_preprocessor({"inputs": "a"}))
    with self.assertRaisesRegex(
        ValueError,
        "Task dataset is missing expected output feature after text "
        "preprocessing: targets"):
      TaskRegistry.get_dataset(
          "text_prep_missing_feature", {"inputs": 13, "targets": 13},
          "train", use_cached=False)

    test_utils.add_tfds_task(
        "text_prep_wrong_type",
        text_preprocessor=_dummy_preprocessor({"inputs": 0, "targets": 1}))
    with self.assertRaisesRegex(
        ValueError,
        "Task dataset has incorrect type for feature 'inputs' after text "
        "preprocessing: Got int32, expected string"):
      TaskRegistry.get_dataset(
          "text_prep_wrong_type", {"inputs": 13, "targets": 13},
          "train", use_cached=False)

    test_utils.add_tfds_task(
        "text_prep_wrong_shape",
        text_preprocessor=_dummy_preprocessor(
            {"inputs": "a", "targets": ["a", "b"]}))
    with self.assertRaisesRegex(
        ValueError,
        "Task dataset has incorrect rank for feature 'targets' after text "
        "preprocessing: Got 1, expected 0"):
      TaskRegistry.get_dataset(
          "text_prep_wrong_shape", {"inputs": 13, "targets": 13},
          "train", use_cached=False)

  def test_invalid_token_preprocessors(self):
    def _dummy_preprocessor(output):
      return lambda _, **unused: tf.data.Dataset.from_tensors(output)
    i64_arr = lambda x: np.array(x, dtype=np.int64)
    def _materialize(task):
      list(tfds.as_numpy(TaskRegistry.get_dataset(
          task, {"inputs": 13, "targets": 13},
          "train", use_cached=False)))

    test_utils.add_tfds_task(
        "token_prep_ok",
        token_preprocessor=_dummy_preprocessor(
            {"inputs": i64_arr([2, 3]), "targets": i64_arr([3]),
             "other": "test"}))
    _materialize("token_prep_ok")

    test_utils.add_tfds_task(
        "token_prep_missing_feature",
        token_preprocessor=_dummy_preprocessor({"inputs": i64_arr([2, 3])}))
    with self.assertRaisesRegex(
        ValueError,
        "Task dataset is missing expected output feature after token "
        "preprocessing: targets"):
      _materialize("token_prep_missing_feature")

    test_utils.add_tfds_task(
        "token_prep_wrong_type",
        token_preprocessor=_dummy_preprocessor(
            {"inputs": "a", "targets": i64_arr([3])}))
    with self.assertRaisesRegex(
        ValueError,
        "Task dataset has incorrect type for feature 'inputs' after token "
        "preprocessing: Got string, expected int64"):
      _materialize("token_prep_wrong_type")

    test_utils.add_tfds_task(
        "token_prep_wrong_shape",
        token_preprocessor=_dummy_preprocessor(
            {"inputs": i64_arr([2, 3]), "targets": i64_arr(1)}))
    with self.assertRaisesRegex(
        ValueError,
        "Task dataset has incorrect rank for feature 'targets' after token "
        "preprocessing: Got 0, expected 1"):
      _materialize("token_prep_wrong_shape")

    test_utils.add_tfds_task(
        "token_prep_has_eos",
        token_preprocessor=_dummy_preprocessor(
            {"inputs": i64_arr([1, 3]), "targets": i64_arr([4])}))
    with self.assertRaisesRegex(
        tf.errors.InvalidArgumentError,
        r".*Feature \\'inputs\\' unexpectedly contains EOS=1 token after token "
        r"preprocessing\..*"):
      _materialize("token_prep_has_eos")

  def test_splits(self):
    test_utils.add_tfds_task("task_with_splits", splits=["validation"])
    task = TaskRegistry.get("task_with_splits")
    self.assertListEqual(["validation"], task.splits)

    test_utils.add_tfds_task("task_with_sliced_splits",
                             splits={"validation": "train[0:1%]"})
    task = TaskRegistry.get("task_with_splits")
    self.assertListEqual(["validation"], task.splits)

  def test_no_eos(self):
    features = {
        "inputs": utils.Feature(add_eos=True),
        "targets": utils.Feature(add_eos=False),
    }
    test_utils.add_task(
        "task_no_eos", test_utils.get_fake_dataset, output_features=features
    )
    fn_task = TaskRegistry.get("task_no_eos")
    test_utils.verify_task_matches_fake_datasets(fn_task, use_cached=False)


if __name__ == "__main__":
  absltest.main()
