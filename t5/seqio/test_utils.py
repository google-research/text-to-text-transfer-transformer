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
"""SeqIO test utilities."""

import collections
import copy
import functools
import os
import shutil
import sys
from typing import Any, Mapping, Optional, Sequence, Union

from absl import flags
from absl import logging
from absl.testing import absltest
import numpy as np
from t5.seqio import dataset_providers
from t5.seqio import preprocessors
from t5.seqio import utils as dataset_utils
from t5.seqio import vocabularies
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

TaskRegistry = dataset_providers.TaskRegistry
MixtureRegistry = dataset_providers.MixtureRegistry

mock = absltest.mock

TEST_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "test_data")


# _ProxyTest is required because py2 does not allow instantiating
# absltest.TestCase directly.
class _ProxyTest(absltest.TestCase):
  """Instance of TestCase to reuse methods for testing."""
  maxDiff = None

  def runTest(self):
    pass


_pyunit_proxy = _ProxyTest()

_FAKE_DATASET = {
    "train": [
        {"prefix": "this", "suffix": "is a test"},
        {"prefix": "that", "suffix": "was a test"},
        {"prefix": "those", "suffix": "were tests"}
    ],
    "validation": [
        {
            "idx": 0, "idxs": (100,), "id": "a", "ids": ("a1", "a2"),
            "prefix": "this", "suffix": "is a validation"
        }, {
            "idx": 1, "idxs": (200, 201), "id": "b", "ids": ("b1",),
            "prefix": "that", "suffix": "was another validation"
        },
    ]
}

# Text preprocessed and tokenized.
_FAKE_TOKENIZED_DATASET = {
    "train": [
        {
            "inputs": (3, 13, 7, 14, 15, 9, 4, 16, 12, 11, 8, 6),
            "inputs_pretokenized": "complete: this",
            "targets": (3, 8, 6, 3, 5, 10),
            "targets_pretokenized": "is a test"
        }, {
            "inputs": (3, 13, 7, 14, 15, 9, 4, 16, 12, 11, 18),
            "inputs_pretokenized": "complete: that",
            "targets": (17, 5, 6, 3, 5, 10),
            "targets_pretokenized": "was a test"
        }, {
            "inputs": (3, 13, 7, 14, 15, 9, 4, 16, 12, 11, 7, 6, 4),
            "inputs_pretokenized": "complete: those",
            "targets": (17, 4, 23, 4, 10, 6),
            "targets_pretokenized": "were tests"
        },
    ],
    "validation": [
        {
            "idx": 0, "idxs": (100,), "id": "a", "ids": ("a1", "a2"),
            "inputs": (3, 13, 7, 14, 15, 9, 4, 16, 12, 11, 8, 6),
            "inputs_pretokenized": "complete: this",
            "targets": (3, 8, 6, 3, 5, 3, 25, 5, 9, 8, 21, 18, 8, 7, 22),
            "targets_pretokenized": "is a validation",
        }, {
            "idx": 1, "idxs": (200, 201), "id": "b", "ids": ("b1",),
            "inputs": (3, 13, 7, 14, 15, 9, 4, 16, 12, 11, 18),
            "inputs_pretokenized": "complete: that",
            "targets": (17, 5, 6, 3, 5, 22, 7, 24, 20, 4, 23, 3, 25, 5, 9, 8,
                        21, 18, 8, 7, 22),
            "targets_pretokenized": "was another validation",
        }
    ]
}

# Text preprocessed and tokenized.
# Simulates legacy cached dataset that used '_plaintext' suffix instead of
# '_pretokenized'.
_FAKE_PLAINTEXT_TOKENIZED_DATASET = {
    "train": [
        {
            "inputs": (3, 13, 7, 14, 15, 9, 4, 16, 12, 11, 8, 6),
            "inputs_plaintext": "complete: this",
            "targets": (3, 8, 6, 3, 5, 10),
            "targets_plaintext": "is a test"
        }, {
            "inputs": (3, 13, 7, 14, 15, 9, 4, 16, 12, 11, 18),
            "inputs_plaintext": "complete: that",
            "targets": (17, 5, 6, 3, 5, 10),
            "targets_plaintext": "was a test"
        }, {
            "inputs": (3, 13, 7, 14, 15, 9, 4, 16, 12, 11, 7, 6, 4),
            "inputs_plaintext": "complete: those",
            "targets": (17, 4, 23, 4, 10, 6),
            "targets_plaintext": "were tests"
        },
    ],
}

# Text preprocessed and tokenized.
_FAKE_TOKEN_PREPROCESSED_DATASET = {
    "train": [
        {
            "inputs": (3, 13, 7, 14, 15, 9, 4, 50, 12, 11, 8, 6),
            "inputs_pretokenized": "complete: this",
            "targets": (3, 8, 6, 3, 5, 10),
            "targets_pretokenized": "is a test"
        }, {
            "inputs": (3, 13, 7, 14, 15, 9, 4, 50, 12, 11, 50),
            "inputs_pretokenized": "complete: that",
            "targets": (17, 5, 6, 3, 5, 10),
            "targets_pretokenized": "was a test"
        }, {
            "inputs": (3, 13, 7, 14, 15, 9, 4, 50, 12, 11, 7, 6, 4),
            "inputs_pretokenized": "complete: those",
            "targets": (17, 4, 23, 4, 10, 6),
            "targets_pretokenized": "were tests"
        },
    ],
    "validation": [
        {
            "idx": 0, "idxs": (100,), "id": "a", "ids": ("a1", "a2"),
            "inputs": (3, 13, 7, 14, 15, 9, 4, 50, 12, 11, 8, 6),
            "inputs_pretokenized": "complete: this",
            "targets": (3, 8, 6, 3, 5, 3, 25, 5, 9, 8, 21, 18, 8, 7, 22),
            "targets_pretokenized": "is a validation",
        }, {
            "idx": 1, "idxs": (200, 201), "id": "b", "ids": ("b1",),
            "inputs": (3, 13, 7, 14, 15, 9, 4, 50, 12, 11, 50),
            "inputs_pretokenized": "complete: that",
            "targets": (17, 5, 6, 3, 5, 22, 7, 24, 20, 4, 23, 3, 25, 5, 9, 8,
                        21, 18, 8, 7, 22),
            "targets_pretokenized": "was another validation",
        }
    ]
}

_FAKE_DATASETS = {
    "input": _FAKE_DATASET,
    "tokenized": _FAKE_TOKENIZED_DATASET,
    "token_preprocessed": _FAKE_TOKEN_PREPROCESSED_DATASET
}

_DEFAULT_SEQUENCE_LENGTH = {"inputs": 13, "targets": 13}


def get_fake_dataset(split, shuffle_files=False, seed=None, shard_info=None):
  """Returns a tf.data.Dataset with fake data."""
  del shuffle_files  # Unused, to be compatible with TFDS API.
  del seed

  output_types = {"prefix": tf.string, "suffix": tf.string}
  if split == "validation":
    output_types.update(
        {"idx": tf.int32, "idxs": tf.int32,
         "id": tf.string, "ids": tf.string})
  output_shapes = {k: [] for k in output_types}
  if split == "validation":
    output_shapes.update({"idxs": [None], "ids": [None]})

  ds = tf.data.Dataset.from_generator(
      lambda: _FAKE_DATASET[split], output_types, output_shapes)
  if shard_info:
    ds = ds.shard(num_shards=shard_info.num_shards, index=shard_info.index)
  return ds


def _get_comparable_examples_from_ds(ds):
  """Puts dataset into format that allows examples to be compared in Py2/3."""
  examples = []
  def _clean_value(v):
    if isinstance(v, bytes):
      return tf.compat.as_text(v)
    if isinstance(v, np.ndarray):
      if isinstance(v[0], bytes):
        return tuple(tf.compat.as_text(s) for s in v)
      return tuple(v)
    return v

  for ex in tfds.as_numpy(ds):
    examples.append(
        tuple((k, _clean_value(v)) for k, v in sorted(ex.items())))
  return examples


def _dump_examples_to_tfrecord(path, examples):
  """Writes list of example dicts to a TFRecord file of tf.Example protos."""
  logging.info("Writing examples to TFRecord: %s", path)
  with tf.io.TFRecordWriter(path) as writer:
    for ex in examples:
      writer.write(dataset_utils.dict_to_tfexample(ex).SerializeToString())


def _dump_examples_to_tsv(path, examples, field_names=("prefix", "suffix")):
  """Writes list of example dicts to a TSV."""
  logging.info("Writing examples to TSV: %s", path)
  with tf.io.gfile.GFile(path, "w") as writer:
    writer.write("\t".join(field_names) + "\n")
    for ex in examples:
      writer.write("\t".join([ex[field] for field in field_names]) + "\n")


def _dump_fake_dataset(path, fake_examples, shard_sizes, dump_fn):
  """Dumps the fake dataset split to sharded TFRecord file."""
  offsets = np.cumsum([0] + shard_sizes)
  for i in range(len(offsets) - 1):
    start, end = offsets[i:i+2]
    shard_path = "%s-%05d-of-%05d" % (path, i, len(shard_sizes))
    dump_fn(shard_path, fake_examples[start:end])


def _maybe_as_bytes(v):
  if isinstance(v, list):
    return [_maybe_as_bytes(x) for x in v]
  if isinstance(v, str):
    return tf.compat.as_bytes(v)
  return v


def _maybe_as_text(v):
  if isinstance(v, list):
    return [_maybe_as_text(x) for x in v]
  if isinstance(v, bytes):
    return tf.compat.as_text(v)
  return v


def dataset_as_text(ds):
  for ex in tfds.as_numpy(ds):
    yield {k: _maybe_as_text(v) for k, v in ex.items()}


def assert_dataset(
    dataset: tf.data.Dataset,
    expected: Union[Mapping[str, Any], Sequence[Mapping[str, Any]]],
    expected_dtypes: Optional[Mapping[str, tf.DType]] = None):
  """Tests whether the entire dataset == expected or [expected].

  Args:
    dataset: a tf.data dataset
    expected: either a single example, or a list of examples. Each example is a
      dictionary.
    expected_dtypes: an optional mapping from feature key to expected dtype.
  """

  if not isinstance(expected, list):
    expected = [expected]
  actual = list(tfds.as_numpy(dataset))
  _pyunit_proxy.assertEqual(len(actual), len(expected))

  def _compare_dict(actual_dict, expected_dict):
    _pyunit_proxy.assertEqual(
        set(actual_dict.keys()), set(expected_dict.keys()))
    for key, actual_value in actual_dict.items():
      if isinstance(actual_value, dict):
        _compare_dict(actual_value, expected_dict[key])
        continue
      if isinstance(actual_value, tf.RaggedTensor):
        actual_value = actual_value.to_list()
      np.testing.assert_array_equal(
          actual_value, _maybe_as_bytes(expected_dict[key]), key)

  for actual_ex, expected_ex in zip(actual, expected):
    _compare_dict(actual_ex, expected_ex)

  if expected_dtypes:
    actual_dtypes = {k: dataset.element_spec[k].dtype for k in expected_dtypes}
    _pyunit_proxy.assertDictEqual(expected_dtypes, actual_dtypes)


def assert_datasets_eq(dataset1: tf.data.Dataset, dataset2: tf.data.Dataset):
  """Assert that two tfds datasets are equal."""

  dataset1 = list(tfds.as_numpy(dataset1))
  dataset2 = list(tfds.as_numpy(dataset2))
  _pyunit_proxy.assertEqual(len(dataset1), len(dataset2))

  def _compare_dict(dataset1, dataset2):
    _pyunit_proxy.assertEqual(
        set(dataset1.keys()), set(dataset2.keys()))
    for key, value1 in dataset1.items():
      if isinstance(value1, dict):
        _compare_dict(value1, dataset2[key])
        continue
      if isinstance(value1, tf.RaggedTensor):
        value1 = value1.to_list()
      np.testing.assert_array_equal(
          value1, _maybe_as_bytes(dataset2[key]), key)

  for ex1, ex2 in zip(dataset1, dataset2):
    _compare_dict(ex1, ex2)


def assert_datasets_neq(dataset1, dataset2):
  """Assert that two tfds datasets are unequal."""

  _pyunit_proxy.assertRaises(AssertionError,
                             assert_datasets_eq, dataset1, dataset2)


def _assert_compare_to_fake_dataset(
    ds, split, features, sequence_length, token_preprocessed=False):
  """Calls assertion to compare fake examples to actual dataaset."""
  fake_examples = copy.deepcopy(_FAKE_DATASETS[
      "token_preprocessed" if token_preprocessed else "tokenized"][split])

  for key, feat in features.items():
    for n, ex in enumerate(fake_examples):
      if sequence_length:
        fake_examples[n][key] = ex[key][:sequence_length[key] - feat.add_eos]
      if feat.add_eos:
        fake_examples[n][key] = fake_examples[n][key] + (1,)

  expected_output_shapes = {
      "inputs": [None], "targets": [None],
      "inputs_pretokenized": [], "targets_pretokenized": []}
  if split == "validation":
    expected_output_shapes.update(
        {"id": [], "ids": [None], "idx": [], "idxs": [None]})
  _pyunit_proxy.assertDictEqual(
      expected_output_shapes,
      {k: v.shape.as_list() for k, v in ds.element_spec.items()})

  expected_output_dtypes = {k: f.dtype for k, f in features.items()}
  _pyunit_proxy.assertDictContainsSubset(
      expected_output_dtypes,
      {k: v.dtype for k, v in ds.element_spec.items()})

  actual_examples = _get_comparable_examples_from_ds(ds)
  expected_examples = [
      tuple(sorted(ex.items())) for ex in fake_examples]
  _pyunit_proxy.assertCountEqual(expected_examples, actual_examples)


def test_text_preprocessor(dataset):
  """Performs preprocessing on the text dataset."""

  def my_fn(ex):
    res = dict(ex)
    del res["prefix"]
    del res["suffix"]
    res.update({
        "inputs": tf.strings.join(["complete: ", ex["prefix"]]),
        "targets": ex["suffix"]
    })
    return res

  return dataset.map(my_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def split_tsv_preprocessor(dataset, field_names=("prefix", "suffix")):
  """Splits TSV into dictionary."""

  def parse_line(line):
    return dict(zip(
        field_names,
        tf.io.decode_csv(
            line, record_defaults=[""] * len(field_names),
            field_delim="\t", use_quote_delim=False)
    ))

  return dataset.map(
      parse_line, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def test_token_preprocessor(dataset, output_features, sequence_length):
  """Change all occurrences of non-zero even numbered tokens in inputs to 50."""
  del output_features
  del sequence_length

  def my_fn(ex):
    inputs = ex["inputs"]
    res = ex.copy()
    res["inputs"] = tf.where(
        tf.greater(inputs, 15),
        tf.constant(50, inputs.dtype),
        inputs)
    return res

  return dataset.map(my_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)


@dataset_utils.map_over_dataset(num_seeds=1)
def random_token_preprocessor(ex, seed):
  """Selects a random shift to roll the tokens by for each feature."""
  for feat in ["inputs", "targets"]:
    tokens = ex[feat]
    res = ex.copy()
    n_tokens = tf.size(tokens)
    random_shift = tf.random.stateless_uniform(
        [], maxval=n_tokens, dtype=tf.int32, seed=seed)
    res[feat] = tf.roll(tokens, shift=random_shift, axis=0)
  return res


def token_preprocessor_no_sequence_length(dataset, output_features):
  return test_token_preprocessor(dataset, output_features, sequence_length=None)


class MockVocabulary(object):
  """Mocks a vocabulary object for testing."""

  def __init__(self, encode_dict, vocab_size=None):
    self._encode_dict = encode_dict
    self._vocab_size = vocab_size

  def encode(self, s):
    return self._encode_dict[s]

  def encode_tf(self, s):
    res = tf.constant([-1], tf.int32)
    for k, v in self._encode_dict.items():
      if tf.equal(s, k):
        res = tf.constant(v, tf.int32)
      else:
        pass
    return res

  @property
  def vocab_size(self):
    return self._vocab_size


def sentencepiece_vocab(extra_ids=0):
  return vocabularies.SentencePieceVocabulary(
      os.path.join(TEST_DATA_DIR, "sentencepiece", "sentencepiece.model"),
      extra_ids=extra_ids)


def clear_tasks():
  TaskRegistry._REGISTRY = {}  # pylint:disable=protected-access


def clear_mixtures():
  MixtureRegistry._REGISTRY = {}  # pylint:disable=protected-access


# pylint:disable=invalid-name
FakeLazyTfds = collections.namedtuple(
    "FakeLazyTfds",
    ["name", "load", "load_shard", "info", "files", "size"])
FakeTfdsInfo = collections.namedtuple("FakeTfdsInfo", ["splits"])
# pylint:enable=invalid-name


class FakeTaskTest(absltest.TestCase):
  """TestCase that sets up fake cached and uncached tasks."""

  DEFAULT_PREPROCESSORS = (
      test_text_preprocessor,
      preprocessors.tokenize,
      dataset_providers.CacheDatasetPlaceholder()
  )

  def add_task(
      self,
      name,
      source,
      preprocessors=DEFAULT_PREPROCESSORS,  # pylint:disable=redefined-outer-name
      output_features=None,
      **kwargs):

    if not output_features:
      output_features = {
          "inputs": dataset_providers.Feature(sentencepiece_vocab()),
          "targets": dataset_providers.Feature(sentencepiece_vocab())
      }

    return TaskRegistry.add(
        name,
        source=source,
        preprocessors=preprocessors,
        output_features=output_features,
        **kwargs)

  def get_tempdir(self):
    try:
      flags.FLAGS.test_tmpdir
    except flags.UnparsedFlagAccessError:
      # Need to initialize flags when running `pytest`.
      flags.FLAGS(sys.argv)
    return self.create_tempdir().full_path

  def setUp(self):
    super().setUp()
    self.maxDiff = None  # pylint:disable=invalid-name

    # Set up data directory.
    self.test_tmpdir = self.get_tempdir()
    self.test_data_dir = os.path.join(self.test_tmpdir, "test_data")
    shutil.copytree(TEST_DATA_DIR, self.test_data_dir)
    for root, dirs, _ in os.walk(self.test_data_dir):
      for d in dirs + [""]:
        os.chmod(os.path.join(root, d), 0o777)

    self._prepare_sources_and_tasks()

  def _prepare_sources_and_tasks(self):
    clear_tasks()
    # Prepare TfdsSource
    # Note we don't use mock.Mock since they fail to pickle.
    fake_tfds_paths = {
        "train": [
            {  # pylint:disable=g-complex-comprehension
                "filename": "train.tfrecord-%05d-of-00002" % i,
                "skip": 0,
                "take": -1
            }
            for i in range(2)],
        "validation": [
            {
                "filename": "validation.tfrecord-00000-of-00001",
                "skip": 0,
                "take": -1
            }],
    }
    def _load_shard(shard_instruction, shuffle_files, seed):
      del shuffle_files
      del seed
      fname = shard_instruction["filename"]
      if "train" in fname:
        ds = get_fake_dataset("train")
        if fname.endswith("00000-of-00002"):
          return ds.take(2)
        else:
          return ds.skip(2)
      else:
        return get_fake_dataset("validation")

    fake_tfds = FakeLazyTfds(
        name="fake:0.0.0",
        load=get_fake_dataset,
        load_shard=_load_shard,
        info=FakeTfdsInfo(splits={"train": None, "validation": None}),
        files=fake_tfds_paths.get,
        size=lambda x: 30 if x == "train" else 10)
    self._tfds_patcher = mock.patch(
        "t5.seqio.utils.LazyTfdsLoader",
        new=mock.Mock(return_value=fake_tfds))
    self._tfds_patcher.start()

    # Set up data directory.
    self.test_tmpdir = self.get_tempdir()
    self.test_data_dir = os.path.join(self.test_tmpdir, "test_data")
    shutil.copytree(TEST_DATA_DIR, self.test_data_dir)
    for root, dirs, _ in os.walk(self.test_data_dir):
      for d in dirs + [""]:
        os.chmod(os.path.join(root, d), 0o777)

    # Prepare uncached TextLineTask.
    self.tfds_source = dataset_providers.TfdsDataSource(
        tfds_name="fake:0.0.0",
        splits=("train", "validation")
    )
    self.add_task("tfds_task", source=self.tfds_source)

    # Prepare TextLineSource.
    _dump_fake_dataset(
        os.path.join(self.test_data_dir, "train.tsv"),
        _FAKE_DATASET["train"], [2, 1], _dump_examples_to_tsv)
    self.text_line_source = dataset_providers.TextLineDataSource(
        split_to_filepattern={
            "train": os.path.join(self.test_data_dir, "train.tsv*"),
        },
        skip_header_lines=1,
    )
    self.add_task(
        "text_line_task",
        source=self.text_line_source,
        preprocessors=(split_tsv_preprocessor,) + self.DEFAULT_PREPROCESSORS)

    # Prepare TFExampleSource.
    _dump_fake_dataset(
        os.path.join(self.test_data_dir, "train.tfrecord"),
        _FAKE_DATASET["train"], [2, 1], _dump_examples_to_tfrecord)
    self.tf_example_source = dataset_providers.TFExampleDataSource(
        split_to_filepattern={
            "train": os.path.join(self.test_data_dir, "train.tfrecord*"),
        },
        feature_description={
            "prefix": tf.io.FixedLenFeature([], tf.string),
            "suffix": tf.io.FixedLenFeature([], tf.string),
        }
    )
    self.add_task("tf_example_task", source=self.tf_example_source)

    # Prepare FunctionDataSource
    self.function_source = dataset_providers.FunctionDataSource(
        dataset_fn=get_fake_dataset,
        splits=["train", "validation"]
    )
    self.add_task("function_task", source=self.function_source)

    # Prepare Task that is tokenized and preprocessed before caching.
    self.add_task(
        "fully_processed_precache",
        source=self.function_source,
        preprocessors=(
            test_text_preprocessor,
            preprocessors.tokenize,
            token_preprocessor_no_sequence_length,
            dataset_providers.CacheDatasetPlaceholder(),
        )
    )

    # Prepare Task that is tokenized after caching.
    self.add_task(
        "tokenized_postcache",
        source=self.function_source,
        preprocessors=(
            test_text_preprocessor,
            dataset_providers.CacheDatasetPlaceholder(),
            preprocessors.tokenize,
            token_preprocessor_no_sequence_length,
        )
    )

    # Prepare Task with randomization.
    self.random_task = self.add_task(
        "random_task",
        source=self.function_source,
        preprocessors=(
            test_text_preprocessor,
            dataset_providers.CacheDatasetPlaceholder(),
            preprocessors.tokenize,
            random_token_preprocessor,
        )
    )

    self.uncached_task = self.add_task("uncached_task", source=self.tfds_source)

    # Prepare cached task.
    dataset_utils.set_global_cache_dirs([self.test_data_dir])
    self.cached_task_dir = os.path.join(self.test_data_dir, "cached_task")
    _dump_fake_dataset(
        os.path.join(self.cached_task_dir, "train.tfrecord"),
        _FAKE_TOKENIZED_DATASET["train"], [2, 1],
        _dump_examples_to_tfrecord)
    _dump_fake_dataset(
        os.path.join(self.cached_task_dir, "validation.tfrecord"),
        _FAKE_TOKENIZED_DATASET["validation"], [2],
        _dump_examples_to_tfrecord)
    self.cached_task = self.add_task("cached_task", source=self.tfds_source)

    # Prepare cached plaintext task.
    _dump_fake_dataset(
        os.path.join(
            self.test_data_dir, "cached_plaintext_task", "train.tfrecord"),
        _FAKE_PLAINTEXT_TOKENIZED_DATASET["train"], [2, 1],
        _dump_examples_to_tfrecord)
    self.cached_plaintext_task = self.add_task(
        "cached_plaintext_task",
        source=self.tfds_source,
        preprocessors=self.DEFAULT_PREPROCESSORS + (test_token_preprocessor,))

  def tearDown(self):
    super().tearDown()
    self._tfds_patcher.stop()
    tf.random.set_seed(None)

  def verify_task_matches_fake_datasets(  # pylint:disable=dangerous-default-value
      self,
      task_name,
      use_cached,
      token_preprocessed=False,
      splits=("train", "validation"),
      sequence_length=_DEFAULT_SEQUENCE_LENGTH,
      num_shards=None
  ):
    """Assert all splits for both tokenized datasets are correct."""
    task = TaskRegistry.get(task_name)
    for split in splits:
      get_dataset = functools.partial(
          task.get_dataset, sequence_length, split, use_cached=use_cached,
          shuffle=False)
      if num_shards:
        ds = get_dataset(shard_info=dataset_providers.ShardInfo(0, num_shards))
        for i in range(1, num_shards):
          ds = ds.concatenate(
              get_dataset(
                  shard_info=dataset_providers.ShardInfo(i, num_shards)))
      else:
        ds = get_dataset()
      _assert_compare_to_fake_dataset(
          ds,
          split,
          task.output_features,
          sequence_length,
          token_preprocessed=token_preprocessed,
      )


class FakeMixtureTest(FakeTaskTest):
  """TestCase that sets up fake cached and uncached tasks."""

  def setUp(self):
    super().setUp()
    clear_mixtures()
    MixtureRegistry.add(
        "uncached_mixture",
        [("uncached_task", 1.0)],
    )
    self.uncached_mixture = MixtureRegistry.get(
        "uncached_mixture")
    MixtureRegistry.add(
        "cached_mixture",
        [("cached_task", 1.0)],
    )
    self.cached_mixture = MixtureRegistry.get("cached_mixture")
    MixtureRegistry.add(
        "uncached_random_mixture",
        [("random_task", 1.0)],
    )
    self.uncached_mixture = MixtureRegistry.get(
        "uncached_random_mixture")
