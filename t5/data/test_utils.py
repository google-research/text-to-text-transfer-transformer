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
"""T5 test utilities."""

import collections
import copy
import os
import shutil
import sys

from absl import flags
from absl import logging
from absl.testing import absltest
import numpy as np
from t5.data import sentencepiece_vocabulary
from t5.data import utils as dataset_utils
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds

TaskRegistry = dataset_utils.TaskRegistry
MixtureRegistry = dataset_utils.MixtureRegistry

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

_SEQUENCE_LENGTH = {"inputs": 13, "targets": 13}

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
    "validation": [
        {
            "idx": 0, "idxs": (100,), "id": "a", "ids": ("a1", "a2"),
            "inputs": (3, 13, 7, 14, 15, 9, 4, 16, 12, 11, 8, 6),
            "inputs_plaintext": "complete: this",
            "targets": (3, 8, 6, 3, 5, 3, 25, 5, 9, 8, 21, 18, 8, 7, 22),
            "targets_plaintext": "is a validation",
        }, {
            "idx": 1, "idxs": (200, 201), "id": "b", "ids": ("b1",),
            "inputs": (3, 13, 7, 14, 15, 9, 4, 16, 12, 11, 18),
            "inputs_plaintext": "complete: that",
            "targets": (17, 5, 6, 3, 5, 22, 7, 24, 20, 4, 23, 3, 25, 5, 9, 8,
                        21, 18, 8, 7, 22),
            "targets_plaintext": "was another validation",
        }
    ]
}

# Text preprocessed and tokenized.
_FAKE_TOKEN_PREPROCESSED_DATASET = {
    "train": [
        {
            "inputs": (3, 13, 7, 14, 15, 9, 4, 50, 12, 11, 8, 6),
            "inputs_plaintext": "complete: this",
            "targets": (3, 8, 6, 3, 5, 10),
            "targets_plaintext": "is a test"
        }, {
            "inputs": (3, 13, 7, 14, 15, 9, 4, 50, 12, 11, 50),
            "inputs_plaintext": "complete: that",
            "targets": (17, 5, 6, 3, 5, 10),
            "targets_plaintext": "was a test"
        }, {
            "inputs": (3, 13, 7, 14, 15, 9, 4, 50, 12, 11, 7, 6, 4),
            "inputs_plaintext": "complete: those",
            "targets": (17, 4, 23, 4, 10, 6),
            "targets_plaintext": "were tests"
        },
    ],
    "validation": [
        {
            "idx": 0, "idxs": (100,), "id": "a", "ids": ("a1", "a2"),
            "inputs": (3, 13, 7, 14, 15, 9, 4, 50, 12, 11, 8, 6),
            "inputs_plaintext": "complete: this",
            "targets": (3, 8, 6, 3, 5, 3, 25, 5, 9, 8, 21, 18, 8, 7, 22),
            "targets_plaintext": "is a validation",
        }, {
            "idx": 1, "idxs": (200, 201), "id": "b", "ids": ("b1",),
            "inputs": (3, 13, 7, 14, 15, 9, 4, 50, 12, 11, 50),
            "inputs_plaintext": "complete: that",
            "targets": (17, 5, 6, 3, 5, 22, 7, 24, 20, 4, 23, 3, 25, 5, 9, 8,
                        21, 18, 8, 7, 22),
            "targets_plaintext": "was another validation",
        }
    ]
}

_FAKE_DATASETS = {
    "input": _FAKE_DATASET,
    "tokenized": _FAKE_TOKENIZED_DATASET,
    "token_preprocessed": _FAKE_TOKEN_PREPROCESSED_DATASET
}


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


def _assert_compare_to_fake_dataset(
    ds, split, features, token_preprocessed=False
):
  """Calls assertion to compare fake examples to actual dataaset."""
  fake_examples = copy.deepcopy(_FAKE_DATASETS[
      "token_preprocessed" if token_preprocessed else "tokenized"][split])

  for key, feat in features.items():
    for n, ex in enumerate(fake_examples):
      if feat.add_eos:
        fake_examples[n][key] = ex[key][:_SEQUENCE_LENGTH[key] - 1] + (1,)
      else:
        fake_examples[n][key] = ex[key][:_SEQUENCE_LENGTH[key]]

  expected_output_shapes = {
      "inputs": [None], "targets": [None],
      "inputs_plaintext": [], "targets_plaintext": []}
  if split == "validation":
    expected_output_shapes.update(
        {"id": [], "ids": [None], "idx": [], "idxs": [None]})
  _pyunit_proxy.assertDictEqual(
      expected_output_shapes,
      {k: v.as_list() for k, v in ds.output_shapes.items()})

  actual_examples = _get_comparable_examples_from_ds(ds)
  expected_examples = [
      tuple(sorted(ex.items())) for ex in fake_examples]
  _pyunit_proxy.assertCountEqual(expected_examples, actual_examples)


def verify_task_matches_fake_datasets(
    task, use_cached, token_preprocessed=False, splits=("train", "validation"),
):
  """Assert all splits for both tokenized datasets are correct."""
  for split in splits:
    _assert_compare_to_fake_dataset(
        task.get_dataset(
            _SEQUENCE_LENGTH, split, use_cached=use_cached, shuffle=False),
        split,
        task.output_features,
        token_preprocessed=token_preprocessed,
    )


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


def assert_dataset(dataset, expected):
  """Tests whether the entire dataset == expected or [expected]."""
  if not isinstance(expected, list):
    expected = [expected]
  dataset = list(tfds.as_numpy(dataset))
  _pyunit_proxy.assertEqual(len(dataset), len(expected))

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

  for actual_ex, expected_ex in zip(dataset, expected):
    _compare_dict(actual_ex, expected_ex)


def get_fake_dataset(split, shuffle_files=False):
  """Returns a tf.data.Dataset with fake data."""
  del shuffle_files  # Unused, to be compatible with TFDS API.
  output_types = {"prefix": tf.string, "suffix": tf.string}
  if split == "validation":
    output_types.update(
        {"idx": tf.int64, "idxs": tf.int64, "id": tf.string, "ids": tf.string})
  output_shapes = {k: [] for k in output_types}
  if split == "validation":
    output_shapes.update({"idxs": [None], "ids": [None]})

  return tf.data.Dataset.from_generator(
      lambda: _FAKE_DATASET[split], output_types, output_shapes)


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

  return dataset.map(my_fn)


def _split_tsv_preprocessor(dataset, field_names=("prefix", "suffix")):
  """Splits TSV into dictionary."""

  def parse_line(line):
    return dict(zip(
        field_names,
        tf.io.decode_csv(
            line, record_defaults=[""] * len(field_names),
            field_delim="\t", use_quote_delim=False)
    ))

  return dataset.map(parse_line)


def test_token_preprocessor(dataset, vocabulary, **unused_kwargs):
  """Change all occurrences of non-zero even numbered tokens in inputs to 50."""
  del vocabulary

  def my_fn(ex):
    inputs = ex["inputs"]
    res = ex.copy()
    res["inputs"] = tf.where_v2(
        tf.greater(inputs, 15),
        tf.constant(50, tf.int64),
        inputs)
    return res

  return dataset.map(my_fn)


def mock_vocabulary(encode_dict, vocab_size=None):
  vocab = mock.MagicMock()
  vocab.vocab_size = vocab_size
  vocab.encode = mock.MagicMock(side_effect=lambda x: encode_dict[x])
  vocab.encode_tf = mock.MagicMock(
      side_effect=lambda x: tf.constant(encode_dict[x]))
  return vocab


def sentencepiece_vocab(extra_ids=0):
  return sentencepiece_vocabulary.SentencePieceVocabulary(
      os.path.join(TEST_DATA_DIR, "sentencepiece", "sentencepiece.model"),
      extra_ids=extra_ids)


def add_tfds_task(
    name,
    tfds_name="fake:0.0.0",
    text_preprocessor=test_text_preprocessor,
    token_preprocessor=None,
    splits=None):
  TaskRegistry.add(
      name,
      dataset_utils.TfdsTask,
      tfds_name=tfds_name,
      text_preprocessor=text_preprocessor,
      token_preprocessor=token_preprocessor,
      sentencepiece_model_path=os.path.join(TEST_DATA_DIR, "sentencepiece",
                                            "sentencepiece.model"),
      metric_fns=[],
      splits=splits)


def add_task(
    name,
    dataset_fn,
    text_preprocessor=test_text_preprocessor,
    token_preprocessor=None,
    splits=("train", "validation"),
    **kwargs):
  TaskRegistry.add(
      name,
      dataset_fn=dataset_fn,
      splits=splits,
      text_preprocessor=text_preprocessor,
      token_preprocessor=token_preprocessor,
      sentencepiece_model_path=os.path.join(
          TEST_DATA_DIR, "sentencepiece", "sentencepiece.model"),
      metric_fns=[],
      **kwargs)


def clear_tasks():
  TaskRegistry._REGISTRY = {}  # pylint:disable=protected-access


def clear_mixtures():
  MixtureRegistry._REGISTRY = {}  # pylint:disable=protected-access


def mark_completed(cache_dir, task_name):
  dirname = os.path.join(cache_dir, task_name)
  if not tf.io.gfile.isdir(dirname):
    tf.io.gfile.mkdir(dirname)
  with tf.io.gfile.GFile(os.path.join(dirname, "COMPLETED"), "w") as f:
    f.write("")


# pylint:disable=invalid-name
FakeLazyTfds = collections.namedtuple(
    "FakeLazyTfds",
    ["name", "load", "load_shard", "info", "files", "size"])
FakeTfdsInfo = collections.namedtuple("FakeTfdsInfo", ["splits"])
# pylint:enable=invalid-name


class FakeTaskTest(absltest.TestCase):
  """TestCase that sets up fake cached and uncached tasks."""

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

    # Mock TFDS
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
    def _load_shard(shard_instruction):
      fname = shard_instruction["filename"]
      if "train" in fname:
        if fname.endswith("00000-of-00002"):
          return get_fake_dataset("train").take(2)
        else:
          return get_fake_dataset("train").skip(2)
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
        "t5.data.utils.LazyTfdsLoader", new=mock.Mock(return_value=fake_tfds))
    self._tfds_patcher.start()

    # Set up data directory.
    self.test_tmpdir = self.get_tempdir()
    self.test_data_dir = os.path.join(self.test_tmpdir, "test_data")
    shutil.copytree(TEST_DATA_DIR, self.test_data_dir)
    for root, dirs, _ in os.walk(self.test_data_dir):
      for d in dirs + [""]:
        os.chmod(os.path.join(root, d), 0o777)

    # Register a cached test Task.
    dataset_utils.set_global_cache_dirs([self.test_data_dir])
    clear_tasks()
    add_tfds_task("cached_task")

    # Prepare cached task.
    self.cached_task = TaskRegistry.get("cached_task")
    cached_task_dir = os.path.join(self.test_data_dir, "cached_task")
    _dump_fake_dataset(
        os.path.join(cached_task_dir, "train.tfrecord"),
        _FAKE_TOKENIZED_DATASET["train"], [2, 1], _dump_examples_to_tfrecord)
    _dump_fake_dataset(
        os.path.join(cached_task_dir, "validation.tfrecord"),
        _FAKE_TOKENIZED_DATASET["validation"], [2], _dump_examples_to_tfrecord)

    # Prepare uncached TfdsTask.
    add_tfds_task("uncached_task")
    self.uncached_task = TaskRegistry.get("uncached_task")

    # Prepare uncached TextLineTask.
    _dump_fake_dataset(
        os.path.join(self.test_data_dir, "train.tsv"),
        _FAKE_DATASET["train"], [2, 1], _dump_examples_to_tsv)
    TaskRegistry.add(
        "text_line_task",
        dataset_utils.TextLineTask,
        split_to_filepattern={
            "train": os.path.join(self.test_data_dir, "train.tsv*"),
        },
        skip_header_lines=1,
        text_preprocessor=[_split_tsv_preprocessor, test_text_preprocessor],
        sentencepiece_model_path=os.path.join(
            TEST_DATA_DIR, "sentencepiece", "sentencepiece.model"),
        metric_fns=[])
    self.text_line_task = TaskRegistry.get("text_line_task")

  def tearDown(self):
    super().tearDown()
    self._tfds_patcher.stop()


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
