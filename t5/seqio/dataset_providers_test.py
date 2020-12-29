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
"""Tests for seqio.dataset_providers."""

import functools
import os
from typing import Callable, Sequence

from absl.testing import absltest
from t5.seqio import dataset_providers
from t5.seqio import feature_converters
from t5.seqio import preprocessors
from t5.seqio import test_utils
from t5.seqio import utils
import tensorflow.compat.v2 as tf

tf.compat.v1.enable_eager_execution()

TaskRegistry = dataset_providers.TaskRegistry
MixtureRegistry = dataset_providers.MixtureRegistry
mock = absltest.mock
assert_dataset = test_utils.assert_dataset
create_default_dataset = test_utils.create_default_dataset


class TasksTest(test_utils.FakeTaskTest):

  def test_invalid_name(self):
    with self.assertRaisesRegex(
        ValueError,
        "Task name 'invalid/name' contains invalid characters. "
        "Must match regex: .*"):
      self.add_task("invalid/name", self.function_source)

  def test_repeat_name(self):
    with self.assertRaisesRegex(
        ValueError,
        "Attempting to register duplicate provider: text_line_task"):
      self.add_task("text_line_task", self.text_line_source)

  def test_function_source_signature(self):
    # Good signatures.
    def good_fn(split, shuffle_files):
      del split
      del shuffle_files
    dataset_providers.FunctionDataSource(good_fn, splits=("train",))

    def default_good_fn(split, shuffle_files=False):
      del split
      del shuffle_files
    dataset_providers.FunctionDataSource(default_good_fn, splits=("train",))

    def seed_fn(split, shuffle_files=True, seed=0):
      del split
      del shuffle_files
      del seed
    dataset_providers.FunctionDataSource(seed_fn, splits=("train",))

    def extra_kwarg_good_fn(split, shuffle_files, unused_kwarg=True):
      del split
      del shuffle_files
    dataset_providers.FunctionDataSource(extra_kwarg_good_fn, splits=("train",))

    # Bad signatures.
    with self.assertRaisesRegex(
        ValueError,
        r"'missing_shuff' must have positional args \('split', "
        r"'shuffle_files'\), got: \('split',\)"):
      def missing_shuff(split):
        del split
      dataset_providers.FunctionDataSource(missing_shuff, splits=("train",))

    with self.assertRaisesRegex(
        ValueError,
        r"'missing_split' must have positional args \('split', "
        r"'shuffle_files'\), got: \('shuffle_files',\)"):
      def missing_split(shuffle_files):
        del shuffle_files
      dataset_providers.FunctionDataSource(missing_split, splits=("train",))

    with self.assertRaisesRegex(
        ValueError,
        r"'extra_pos_arg' may only have positional args \('split', "
        r"'shuffle_files'\), got: \('split', 'shuffle_files', 'unused_arg'\)"):
      def extra_pos_arg(split, shuffle_files, unused_arg):
        del split
        del shuffle_files
      dataset_providers.FunctionDataSource(extra_pos_arg, splits=("train",))

  def test_no_tfds_version(self):
    with self.assertRaisesRegex(
        ValueError, "TFDS name must contain a version number, got: fake"):
      dataset_providers.TfdsDataSource(tfds_name="fake")

  def test_tfds_task(self):
    self.verify_task_matches_fake_datasets(
        "tfds_task", use_cached=False)

  def test_function_task(self):
    self.verify_task_matches_fake_datasets(
        "function_task", use_cached=False)

  def test_text_line_task(self):
    self.verify_task_matches_fake_datasets(
        "text_line_task", use_cached=False, splits=["train"])

  def test_tf_example_task(self):
    self.verify_task_matches_fake_datasets(
        "tf_example_task", use_cached=False, splits=["train"])

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
        "'tfds_task' does not exist in any of the task cache directories"):
      TaskRegistry.get("tfds_task").assert_cached()

  def test_get_cached_stats(self):
    expected_train_stats = {
        "examples": 3,
        "inputs_tokens": 36, "inputs_max_tokens": 13,
        "targets_tokens": 18, "targets_max_tokens": 6}
    self.assertEqual(
        expected_train_stats,
        self.cached_task.get_cached_stats("train"))
    # Check repeated call.
    self.assertEqual(
        expected_train_stats,
        self.cached_task.get_cached_stats("train"))
    expected_validation_stats = {
        "examples": 2,
        "inputs_tokens": 23, "inputs_max_tokens": 12,
        "targets_tokens": 36, "targets_max_tokens": 21}
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
    self.verify_task_matches_fake_datasets(
        "cached_task", use_cached=True, token_preprocessed=False)

    # Test with token preprocessor.
    self.cached_task._preprocessors = self.DEFAULT_PREPROCESSORS + (
        test_utils.test_token_preprocessor,)
    self.verify_task_matches_fake_datasets(
        "cached_task", use_cached=True, token_preprocessed=True)

  def test_get_dataset_onthefly(self):
    self.verify_task_matches_fake_datasets(
        "uncached_task", use_cached=False)

    # Test with token preprocessor.
    self.cached_task._preprocessors = self.DEFAULT_PREPROCESSORS + (
        test_utils.test_token_preprocessor,)
    self.verify_task_matches_fake_datasets(
        "cached_task", use_cached=False, token_preprocessed=True)

  def test_get_dataset_no_truncation(self):
    self.verify_task_matches_fake_datasets(
        "uncached_task", use_cached=False, sequence_length=None)

  def test_sharding(self):
    for i in range(3):
      self.verify_task_matches_fake_datasets(
          "cached_task", use_cached=False, num_shards=i,
          token_preprocessed=False)
      self.verify_task_matches_fake_datasets(
          "cached_task", use_cached=True, num_shards=i,
          token_preprocessed=False)

  def test_feature_validation(self):
    default_vocab = test_utils.sentencepiece_vocab()
    features = {
        "inputs":
            dataset_providers.Feature(vocabulary=default_vocab, required=False),
        "targets":
            dataset_providers.Feature(vocabulary=default_vocab, required=True),
    }

    def _materialize(output):
      task = dataset_providers.Task(
          "feature_validation_task",
          self.function_source,
          output_features=features,
          preprocessors=(lambda _: tf.data.Dataset.from_tensors(output),),
          metric_fns=[],
      )
      list(
          task.get_dataset(
              {"inputs": 13, "targets": 13}, "train", use_cached=False
          ).as_numpy_iterator()
      )

    # Missing optional feature: OK
    _materialize({"targets": [0]})

    # Missing required feature.
    with self.assertRaisesRegex(
        ValueError,
        "Task dataset is missing expected output feature after preprocessing: "
        "targets"):
      _materialize({"inputs": [0]})

    # Wrong type.
    with self.assertRaisesRegex(
        ValueError,
        "Task dataset has incorrect type for feature 'targets' after "
        "preprocessing: Got string, expected int32"):
      _materialize({"targets": ["wrong type"]})

    # Wrong rank.
    with self.assertRaisesRegex(
        ValueError,
        "Task dataset has incorrect rank for feature 'targets' after "
        "preprocessing: Got 0, expected 1"):
      _materialize({"targets": 0})

  def test_value_errors(self):
    dataset_fn = (
        lambda split, shuffle_files: tf.data.Dataset.from_tensors(["test"]))
    output_features = {
        "inputs": dataset_providers.Feature(test_utils.sentencepiece_vocab())
    }

    with self.assertRaisesRegex(
        ValueError, "`CacheDatasetPlaceholder` can appear at most once in the "
        "preprocessing pipeline. Found 2 in 'multiple_cache_placeholders'."):
      dataset_providers.Task(
          "multiple_cache_placeholders",
          source=dataset_providers.FunctionDataSource(
              dataset_fn=dataset_fn,
              splits=["train", "validation"]
          ),
          preprocessors=[
              test_utils.test_text_preprocessor,
              preprocessors.tokenize,
              dataset_providers.CacheDatasetPlaceholder(),
              test_utils.test_token_preprocessor,
              dataset_providers.CacheDatasetPlaceholder()
          ],
          output_features=output_features,
          metric_fns=[])

    with self.assertRaisesRegex(
        ValueError,
        "'test_token_preprocessor' has a `sequence_length` argument but occurs "
        "before `CacheDatasetPlaceholder` in 'sequence_length_pre_cache'. This "
        "is not allowed since the sequence length is specified at run time."):
      dataset_providers.Task(
          "sequence_length_pre_cache",
          dataset_providers.FunctionDataSource(
              dataset_fn=dataset_fn,
              splits=["train"],
          ),
          preprocessors=[
              test_utils.test_text_preprocessor,
              preprocessors.tokenize,
              test_utils.test_token_preprocessor,
              dataset_providers.CacheDatasetPlaceholder()
          ],
          output_features=output_features,
          metric_fns=[])

  def test_tfds_source_splits(self):
    default_splits_src = dataset_providers.TfdsDataSource("fake:0.0.0")
    self.assertSameElements(["train", "validation"], default_splits_src.splits)

    validation_split_src = dataset_providers.TfdsDataSource(
        "fake:0.0.0", splits=["validation"])
    self.assertSameElements(["validation"], validation_split_src.splits)

    sliced_split_src = dataset_providers.TfdsDataSource(
        "fake:0.0.0", splits={"validation": "train[0:1%]"})
    self.assertSameElements(["validation"], sliced_split_src.splits)

  def test_no_eos(self):
    default_vocab = test_utils.sentencepiece_vocab()
    features = {
        "inputs":
            dataset_providers.Feature(add_eos=True, vocabulary=default_vocab),
        "targets":
            dataset_providers.Feature(add_eos=False, vocabulary=default_vocab),
    }
    self.add_task("task_no_eos", self.function_source, output_features=features)
    self.verify_task_matches_fake_datasets("task_no_eos", use_cached=False)

  def test_dtype(self):
    default_vocab = test_utils.sentencepiece_vocab()
    features = {
        "inputs":
            # defaults to int32
            dataset_providers.Feature(vocabulary=default_vocab),
        "targets":
            dataset_providers.Feature(dtype=tf.int64, vocabulary=default_vocab),
    }

    self.add_task(
        "task_dtypes",
        self.function_source,
        preprocessors=self.DEFAULT_PREPROCESSORS + (
            utils.map_over_dataset(
                lambda x: {k: tf.cast(v, tf.int64) if k == "targets" else v  # pylint:disable=g-long-lambda
                           for k, v in x.items()}
            ),
        ),
        output_features=features
    )
    self.verify_task_matches_fake_datasets("task_dtypes", use_cached=False)

  def test_same_seeds_cached_match(self):
    dataset1 = self.cached_task.get_dataset(
        {"inputs": 13, "targets": 13},
        split="train", use_cached=True, shuffle=True, seed=0)
    dataset2 = self.cached_task.get_dataset(
        {"inputs": 13, "targets": 13},
        split="train", use_cached=True, shuffle=True, seed=0)
    test_utils.assert_datasets_eq(dataset1, dataset2)

  def test_different_seeds_cached_mismatch(self):
    dataset1 = self.cached_task.get_dataset(
        {"inputs": 13, "targets": 13},
        split="train", use_cached=True, shuffle=True, seed=0)
    dataset2 = self.cached_task.get_dataset(
        {"inputs": 13, "targets": 13},
        split="train", use_cached=True, shuffle=True, seed=42)
    test_utils.assert_datasets_neq(dataset1, dataset2)

  def test_same_seeds_uncached_match(self):
    dataset1 = self.uncached_task.get_dataset(
        {"inputs": 13, "targets": 13},
        split="train", use_cached=False, shuffle=True, seed=0)
    dataset2 = self.uncached_task.get_dataset(
        {"inputs": 13, "targets": 13},
        split="train", use_cached=False, shuffle=True, seed=0)
    test_utils.assert_datasets_eq(dataset1, dataset2)

  def test_different_seeds_uncached_mismatch(self):
    dataset1 = self.uncached_task.get_dataset(
        {"inputs": 13, "targets": 13},
        split="train", use_cached=False, shuffle=True, seed=0)
    dataset2 = self.uncached_task.get_dataset(
        {"inputs": 13, "targets": 13},
        split="train", use_cached=False, shuffle=True, seed=42)
    test_utils.assert_datasets_neq(dataset1, dataset2)

  def test_same_seeds_random_tp_uncached_match(self):
    dataset1 = self.random_task.get_dataset(
        {"inputs": 13, "targets": 13},
        split="train", use_cached=False, shuffle=True, seed=0).repeat(4)
    dataset2 = self.random_task.get_dataset(
        {"inputs": 13, "targets": 13},
        split="train", use_cached=False, shuffle=True, seed=0).repeat(4)
    test_utils.assert_datasets_eq(dataset1, dataset2)

  def test_different_seeds_random_tp_uncached_mismatch(self):
    dataset1 = self.random_task.get_dataset(
        {"inputs": 13, "targets": 13},
        split="train", use_cached=False, shuffle=True, seed=0)
    dataset2 = self.random_task.get_dataset(
        {"inputs": 13, "targets": 13},
        split="train", use_cached=False, shuffle=True, seed=42)
    test_utils.assert_datasets_neq(dataset1, dataset2)

  def test_no_shuffle_with_seed_cached_match(self):
    dataset1 = self.cached_task.get_dataset(
        {"inputs": 13, "targets": 13},
        split="train", use_cached=True, shuffle=False, seed=0)
    dataset2 = self.cached_task.get_dataset(
        {"inputs": 13, "targets": 13},
        split="train", use_cached=True, shuffle=False, seed=42)
    test_utils.assert_datasets_eq(dataset1, dataset2)

  def test_no_shuffle_with_seed_uncached_match(self):
    dataset1 = self.uncached_task.get_dataset(
        {"inputs": 13, "targets": 13},
        split="train", use_cached=False, shuffle=False, seed=0)
    dataset2 = self.uncached_task.get_dataset(
        {"inputs": 13, "targets": 13},
        split="train", use_cached=False, shuffle=False, seed=42)
    test_utils.assert_datasets_eq(dataset1, dataset2)

  def test_no_shuffle_different_seeds_random_tp_uncached_mismatch(self):
    dataset1 = self.random_task.get_dataset(
        {"inputs": 13, "targets": 13},
        split="train", use_cached=False, shuffle=False, seed=0)
    dataset2 = self.random_task.get_dataset(
        {"inputs": 13, "targets": 13},
        split="train", use_cached=False, shuffle=False, seed=42)
    test_utils.assert_datasets_neq(dataset1, dataset2)

  def test_plaintext_to_pretokenized_rename(self):
    ds = self.cached_plaintext_task.get_dataset(
        {"inputs": 13, "targets": 13},
        split="train", use_cached=True, shuffle=False)
    keys = next(ds.as_numpy_iterator()).keys()
    self.assertSetEqual(
        set(keys),
        set(["inputs", "inputs_pretokenized",
             "targets", "targets_pretokenized"]))


class MixturesTest(test_utils.FakeTaskTest):

  def test_tasks(self):
    self.add_task("task1", self.function_source)
    self.add_task("task2", self.function_source)
    MixtureRegistry.add("test_mix1", [("task1", 1), ("task2", 1)])
    mix = MixtureRegistry.get("test_mix1")
    self.assertEqual(len(mix.tasks), 2)

    for task in mix.tasks:
      self.verify_task_matches_fake_datasets(task.name, use_cached=False)
      self.assertEqual(mix.get_rate(task), 1)

  def test_num_examples(self):
    MixtureRegistry.add("test_mix2", [(self.cached_task.name, 1)])
    mix = MixtureRegistry.get("test_mix2")
    self.assertEqual(mix.num_input_examples(split="train"), 30)

  def test_splits(self):
    MixtureRegistry.add(
        "test_mix",
        [(self.cached_task.name, 1), (self.uncached_task.name, 1)]
    )
    mix = MixtureRegistry.get("test_mix")
    self.assertSameElements(["train", "validation"], mix.splits, 30)

  def test_get_dataset(self):
    MixtureRegistry.add("test_mix3", [(self.cached_task.name, 1)])

    task_ds = TaskRegistry.get_dataset(
        self.cached_task.name, {
            "inputs": 13,
            "targets": 13
        },
        "validation",
        use_cached=False,
        shuffle=False)

    mix_ds = MixtureRegistry.get("test_mix3").get_dataset(
        {
            "inputs": 13,
            "targets": 13
        }, "validation", use_cached=False, shuffle=False)

    # mix.get_dataset strips non-output features
    task_ds = task_ds.map(lambda x: {k: x[k] for k in ["inputs", "targets"]})

    # limit size since get_dataset repeats the dataset
    test_utils.assert_datasets_eq(task_ds.repeat(2), mix_ds.take(4))

  def test_get_dataset_mix(self):
    def _constant_preprocessor(_, val):
      return tf.data.Dataset.from_tensors({
          "targets": tf.constant([val], tf.int32),
          "inputs": tf.constant([val], tf.int32),
      })

    self.add_task(
        "two_task",
        self.function_source,
        preprocessors=(functools.partial(_constant_preprocessor, val=2),)
    )

    self.add_task(
        "three_task",
        self.function_source,
        preprocessors=(functools.partial(_constant_preprocessor, val=3),)
    )

    MixtureRegistry.add("test_mix", [("two_task", 1), ("three_task", 1)])

    sequence_length = {"inputs": 2, "targets": 2}
    mix_ds = MixtureRegistry.get("test_mix").get_dataset(
        sequence_length, "train", seed=13).take(1000)

    res = sum(int(item["inputs"][0]) for item in mix_ds.as_numpy_iterator())
    self.assertEqual(res, 2500)

  def test_copy_pretokenized(self):
    def _constant_preprocessor(_, val):
      return tf.data.Dataset.from_tensors({
          "targets": tf.constant([val], tf.int32),
          "targets_pretokenized": tf.constant(f"targets_{val}"),
          "inputs": tf.constant([val], tf.int32),
          "inputs_pretokenized": tf.constant(f"inputs_{val}")
      })

    self.add_task(
        "two_task",
        self.function_source,
        preprocessors=(functools.partial(_constant_preprocessor, val=2),)
    )

    self.add_task(
        "three_task",
        self.function_source,
        preprocessors=(functools.partial(_constant_preprocessor, val=3),)
    )

    MixtureRegistry.add("test_mix", [("two_task", 1), ("three_task", 1)])

    sequence_length = {"inputs": 2, "targets": 2}

    mix_ds = MixtureRegistry.get("test_mix").get_dataset(
        sequence_length, "train", seed=13, copy_pretokenized=True).take(1000)
    inputs_pretokenized = set(
        ex["inputs_pretokenized"] for ex in mix_ds.as_numpy_iterator())
    targets_pretokenized = set(
        ex["targets_pretokenized"] for ex in mix_ds.as_numpy_iterator())
    self.assertCountEqual([b"inputs_2", b"inputs_3"], inputs_pretokenized)
    self.assertCountEqual([b"targets_2", b"targets_3"], targets_pretokenized)

    mix_ds = MixtureRegistry.get("test_mix").get_dataset(
        sequence_length, "train", seed=13, copy_pretokenized=False).take(1000)
    for ex in mix_ds.as_numpy_iterator():
      self.assertNoCommonElements(
          ["inputs_pretokenized", "targets_pretokenized"], ex.keys())

  def test_get_rate_with_callable(self):
    def fn(t):
      self.assertEqual(t.name, "task4")
      return 42
    self.add_task("task4", self.function_source)
    task = TaskRegistry.get("task4")
    MixtureRegistry.add("test_mix5", [("task4", fn)])
    mix = MixtureRegistry.get("test_mix5")
    self.assertEqual(mix.get_rate(task), 42)

  def test_mixture_of_mixtures(self):
    self.add_task("task_a", self.function_source)
    self.add_task("task_b", self.function_source)
    self.add_task("task_c", self.function_source)
    MixtureRegistry.add("another_mix", [("task_a", 1), ("task_b", 1)])
    MixtureRegistry.add("supermix", [("another_mix", 1), ("task_c", 1)])
    supermix = MixtureRegistry.get("supermix")
    names = [task.name for task in supermix.tasks]
    self.assertEqual(names, ["task_a", "task_b", "task_c"])
    self.assertEqual([supermix.get_rate(t) for t in supermix.tasks],
                     [0.5, 0.5, 1])

  def test_mixture_of_mixtures_dupe(self):
    self.add_task("task2_a", self.function_source)
    self.add_task("task2_b", self.function_source)
    self.add_task("task2_c", self.function_source)
    MixtureRegistry.add("yet_another_mix", [("task2_a", 1), ("task2_b", 1)])
    MixtureRegistry.add("supermix_with_dupe", [("yet_another_mix", 1),
                                               ("task2_a", 1), ("task2_c", 1)])
    supermix = MixtureRegistry.get("supermix_with_dupe")
    names = [task.name for task in supermix.tasks]
    self.assertEqual(names, ["task2_a", "task2_b", "task2_c"])
    self.assertEqual([supermix.get_rate(t) for t in supermix.tasks],
                     [1.5, 0.5, 1])


class GetDatasetTest(tf.test.TestCase):

  def test_get_dataset_enc_dec_unpacked(self):
    mixture_or_task_name = "enc_dec_unpacked"
    x = [{"inputs": [7, 8, 5, 6, 9, 4, 3], "targets": [3, 9]},
         {"inputs": [8, 4], "targets": [4]},
         {"inputs": [5, 6, 7], "targets": [6, 5]}]
    ds = create_default_dataset(x)
    dataset_fn = lambda split, shuffle_files: ds
    register_dummy_task(mixture_or_task_name, dataset_fn=dataset_fn)

    task_feature_lengths = {"inputs": 7, "targets": 5}
    converter = feature_converters.EncDecFeatureConverter(pack=False)
    output_ds = dataset_providers.get_dataset(
        mixture_or_task_name=mixture_or_task_name,
        task_feature_lengths=task_feature_lengths,
        dataset_split="train",
        shuffle=False,
        feature_converter=converter)

    expected = [{
        "encoder_input_tokens": [7, 8, 5, 6, 9, 4, 1],
        "decoder_target_tokens": [3, 9, 1, 0, 0],
        "decoder_input_tokens": [0, 3, 9, 1, 0],
        "decoder_loss_weights": [1, 1, 1, 0, 0],
    }, {
        "encoder_input_tokens": [8, 4, 1, 0, 0, 0, 0],
        "decoder_target_tokens": [4, 1, 0, 0, 0],
        "decoder_input_tokens": [0, 4, 1, 0, 0],
        "decoder_loss_weights": [1, 1, 0, 0, 0],
    }, {
        "encoder_input_tokens": [5, 6, 7, 1, 0, 0, 0],
        "decoder_target_tokens": [6, 5, 1, 0, 0],
        "decoder_input_tokens": [0, 6, 5, 1, 0],
        "decoder_loss_weights": [1, 1, 1, 0, 0],
    }]
    expected_dtypes = {feat: tf.int32 for feat in expected[0].keys()}
    assert_dataset(output_ds, expected, expected_dtypes=expected_dtypes)

  def test_get_dataset_enc_dec_packed(self):
    mixture_or_task_name = "enc_dec_packed"
    x = [{"inputs": [7, 8, 5, 6, 9, 4, 3], "targets": [3, 9]},
         {"inputs": [8, 4], "targets": [4]},
         {"inputs": [5, 6, 7], "targets": [6, 5]}]
    ds = create_default_dataset(x)
    dataset_fn = lambda split, shuffle_files: ds
    register_dummy_task(mixture_or_task_name, dataset_fn=dataset_fn)

    task_feature_lengths = {"inputs": 7, "targets": 5}
    converter = feature_converters.EncDecFeatureConverter(pack=True)
    output_ds = dataset_providers.get_dataset(
        mixture_or_task_name=mixture_or_task_name,
        task_feature_lengths=task_feature_lengths,
        dataset_split="train",
        shuffle=False,
        feature_converter=converter)

    expected = [{
        # Example 1 is trimmed
        "encoder_input_tokens": [7, 8, 5, 6, 9, 4, 1],
        "encoder_segment_ids": [1, 1, 1, 1, 1, 1, 1],
        "encoder_positions": [0, 1, 2, 3, 4, 5, 6],
        "decoder_target_tokens": [3, 9, 1, 0, 0],
        "decoder_input_tokens": [0, 3, 9, 0, 0],
        "decoder_loss_weights": [1, 1, 1, 0, 0],
        "decoder_segment_ids": [1, 1, 1, 0, 0],
        "decoder_positions": [0, 1, 2, 0, 0],
    }, {
        # Example 2 and 3 are packed together
        "encoder_input_tokens": [8, 4, 1, 5, 6, 7, 1],
        "encoder_segment_ids": [1, 1, 1, 2, 2, 2, 2],
        "encoder_positions": [0, 1, 2, 0, 1, 2, 3],
        "decoder_target_tokens": [4, 1, 6, 5, 1],
        "decoder_input_tokens": [0, 4, 0, 6, 5],
        "decoder_loss_weights": [1, 1, 1, 1, 1],
        "decoder_segment_ids": [1, 1, 2, 2, 2],
        "decoder_positions": [0, 1, 0, 1, 2],
    }]
    expected_dtypes = {feat: tf.int32 for feat in expected[0].keys()}
    assert_dataset(output_ds, expected, expected_dtypes=expected_dtypes)

  def test_get_dataset_both_train_and_validation_splits(self):
    mixture_or_task_name = "both_train_and_validation_splits"
    x_train = [{"inputs": [7, 8, 5, 6, 9, 4, 3], "targets": [3, 9]}]
    x_val = [{"inputs": [8, 4], "targets": [4]}]
    datasets = {
        "train": create_default_dataset(x_train),
        "validation": create_default_dataset(x_val)
    }
    dataset_fn = lambda split, shuffle_files: datasets[split]
    register_dummy_task(mixture_or_task_name, dataset_fn=dataset_fn)

    task_feature_lengths = {"inputs": 7, "targets": 5}
    output_ds = {}
    for split in ["train", "validation"]:
      converter = feature_converters.EncDecFeatureConverter(pack=False)
      output_ds[split] = dataset_providers.get_dataset(
          mixture_or_task_name=mixture_or_task_name,
          task_feature_lengths=task_feature_lengths,
          dataset_split=split,
          shuffle=False,
          feature_converter=converter)

    expected_train = {
        "encoder_input_tokens": [7, 8, 5, 6, 9, 4, 1],
        "decoder_target_tokens": [3, 9, 1, 0, 0],
        "decoder_input_tokens": [0, 3, 9, 1, 0],
        "decoder_loss_weights": [1, 1, 1, 0, 0],
    }
    expected_val = {
        "encoder_input_tokens": [8, 4, 1, 0, 0, 0, 0],
        "decoder_target_tokens": [4, 1, 0, 0, 0],
        "decoder_input_tokens": [0, 4, 1, 0, 0],
        "decoder_loss_weights": [1, 1, 0, 0, 0],
    }
    expected_dtypes = {feat: tf.int32 for feat in expected_train.keys()}
    assert_dataset(
        output_ds["train"], expected_train, expected_dtypes=expected_dtypes)
    assert_dataset(
        output_ds["validation"], expected_val, expected_dtypes=expected_dtypes)

  def test_get_dataset_enc_dec_sharded(self):
    mixture_or_task_name = "enc_dec_sharded"
    x = [{"inputs": [7, 8, 5, 6, 9, 4, 3], "targets": [3, 9]},
         {"inputs": [8, 4], "targets": [4]},
         {"inputs": [5, 6, 7], "targets": [6, 5]}]
    ds = create_default_dataset(x)
    dataset_fn = lambda split, shuffle_files: ds
    register_dummy_task(mixture_or_task_name, dataset_fn=dataset_fn)

    task_feature_lengths = {"inputs": 7, "targets": 5}
    converter = feature_converters.EncDecFeatureConverter(pack=False)
    shard_info = dataset_providers.ShardInfo(index=0, num_shards=2)
    output_ds = dataset_providers.get_dataset(
        mixture_or_task_name=mixture_or_task_name,
        task_feature_lengths=task_feature_lengths,
        dataset_split="train",
        shuffle=False,
        feature_converter=converter,
        shard_info=shard_info)

    # Example index 1 should not be present in the sharded dataset.
    expected = [{
        "encoder_input_tokens": [7, 8, 5, 6, 9, 4, 1],
        "decoder_target_tokens": [3, 9, 1, 0, 0],
        "decoder_input_tokens": [0, 3, 9, 1, 0],
        "decoder_loss_weights": [1, 1, 1, 0, 0],
    }, {
        "encoder_input_tokens": [5, 6, 7, 1, 0, 0, 0],
        "decoder_target_tokens": [6, 5, 1, 0, 0],
        "decoder_input_tokens": [0, 6, 5, 1, 0],
        "decoder_loss_weights": [1, 1, 1, 0, 0],
    }]
    expected_dtypes = {feat: tf.int32 for feat in expected[0].keys()}
    assert_dataset(output_ds, expected, expected_dtypes=expected_dtypes)

  def test_get_dataset_enc_dec_sharded_and_packed(self):
    mixture_or_task_name = "enc_dec_sharded_and_packed"
    x = [{"inputs": [7, 8], "targets": [3, 9]},
         {"inputs": [8, 4], "targets": [4]},
         {"inputs": [5, 6, 7], "targets": [6]}]
    ds = create_default_dataset(x)
    dataset_fn = lambda split, shuffle_files: ds
    register_dummy_task(mixture_or_task_name, dataset_fn=dataset_fn)

    task_feature_lengths = {"inputs": 7, "targets": 5}
    converter = feature_converters.EncDecFeatureConverter(pack=True)
    shard_info = dataset_providers.ShardInfo(index=0, num_shards=2)
    output_ds = dataset_providers.get_dataset(
        mixture_or_task_name=mixture_or_task_name,
        task_feature_lengths=task_feature_lengths,
        dataset_split="train",
        shuffle=False,
        feature_converter=converter,
        shard_info=shard_info)

    # Packing should be done after the sharding.
    expected = {
        "encoder_input_tokens": [7, 8, 1, 5, 6, 7, 1],
        "encoder_segment_ids": [1, 1, 1, 2, 2, 2, 2],
        "encoder_positions": [0, 1, 2, 0, 1, 2, 3],
        "decoder_target_tokens": [3, 9, 1, 6, 1],
        "decoder_input_tokens": [0, 3, 9, 0, 6],
        "decoder_loss_weights": [1, 1, 1, 1, 1],
        "decoder_segment_ids": [1, 1, 1, 2, 2],
        "decoder_positions": [0, 1, 2, 0, 1],
    }
    expected_dtypes = {feat: tf.int32 for feat in expected.keys()}
    assert_dataset(output_ds, expected, expected_dtypes=expected_dtypes)


def register_dummy_task(
    task_name: str,
    dataset_fn: Callable[[str, str], tf.data.Dataset],
    output_feature_names: Sequence[str] = ("inputs", "targets")) -> None:
  """Register a dummy task for GetDatasetTest."""
  dataset_providers.TaskRegistry.add(
      task_name,
      source=dataset_providers.FunctionDataSource(
          dataset_fn=dataset_fn, splits=["train", "validation"]),
      preprocessors=[
          dataset_providers.CacheDatasetPlaceholder(),
          preprocessors.append_eos_after_trim,
      ],
      output_features={
          feat: dataset_providers.Feature(test_utils.sentencepiece_vocab())
          for feat in output_feature_names
      },
      metric_fns=[])


if __name__ == "__main__":
  absltest.main()
