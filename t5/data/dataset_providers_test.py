# Copyright 2022 The T5 Authors.
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

"""Tests for t5.data.dataset_providers."""
import os

from absl.testing import absltest
import seqio
from seqio import test_utils
from t5.data import dataset_providers
import tensorflow.compat.v2 as tf

tf.compat.v1.enable_eager_execution()

TaskRegistry = dataset_providers.TaskRegistry
MixtureRegistry = seqio.MixtureRegistry
mock = absltest.mock


def _add_t5_task(name,
                 cls,
                 text_preprocessor=(test_utils.test_text_preprocessor,),
                 output_features=None,
                 **kwargs):
  output_features = output_features or {
      "inputs": seqio.Feature(test_utils.sentencepiece_vocab()),
      "targets": seqio.Feature(test_utils.sentencepiece_vocab())
  }
  return TaskRegistry.add(
      name,
      cls,
      text_preprocessor=text_preprocessor,
      metric_fns=[],
      output_features=output_features,
      **kwargs)


class TasksTest(test_utils.FakeTaskTest):

  def test_tfds_task(self):
    _add_t5_task(
        "t5_tfds_task", dataset_providers.TfdsTask, tfds_name="fake:0.0.0")
    self.verify_task_matches_fake_datasets("t5_tfds_task", use_cached=False)

  def test_function_task(self):
    _add_t5_task(
        "t5_fn_task",
        dataset_providers.FunctionTask,
        splits=("train", "validation"),
        dataset_fn=test_utils.get_fake_dataset)
    self.verify_task_matches_fake_datasets("t5_fn_task", use_cached=False)

  def test_text_line_task(self):
    _add_t5_task(
        "t5_text_line_task",
        dataset_providers.TextLineTask,
        split_to_filepattern={
            "train": os.path.join(self.test_data_dir, "train.tsv*"),
        },
        skip_header_lines=1,
        text_preprocessor=(test_utils.split_tsv_preprocessor,
                           test_utils.test_text_preprocessor))
    self.verify_task_matches_fake_datasets(
        "t5_text_line_task", use_cached=False, splits=["train"])

  def test_tf_example_task(self):
    self.verify_task_matches_fake_datasets(
        "tf_example_task", use_cached=False, splits=["train"])

  def test_cached_task(self):
    TaskRegistry.remove("cached_task")
    _add_t5_task(
        "cached_task", dataset_providers.TfdsTask, tfds_name="fake:0.0.0")
    self.verify_task_matches_fake_datasets("cached_task", use_cached=True)

  def test_token_preprocessor(self):
    TaskRegistry.remove("cached_task")
    _add_t5_task(
        "cached_task",
        dataset_providers.TfdsTask,
        tfds_name="fake:0.0.0",
        token_preprocessor=(test_utils.test_token_preprocessor,))

    self.verify_task_matches_fake_datasets(
        "cached_task", use_cached=False, token_preprocessed=True)
    self.verify_task_matches_fake_datasets(
        "cached_task", use_cached=True, token_preprocessed=True)

  def test_optional_features(self):

    def _dummy_preprocessor(output):
      return lambda _: tf.data.Dataset.from_tensors(output)

    default_vocab = test_utils.sentencepiece_vocab()
    features = {
        "inputs": seqio.Feature(vocabulary=default_vocab, required=False),
        "targets": seqio.Feature(vocabulary=default_vocab, required=True),
    }

    task = _add_t5_task(
        "task_missing_optional_feature",
        dataset_providers.TfdsTask,
        tfds_name="fake:0.0.0",
        output_features=features,
        text_preprocessor=_dummy_preprocessor({"targets": "a"}))
    task.get_dataset({"targets": 13}, "train", use_cached=False)

    task = _add_t5_task(
        "task_missing_required_feature",
        dataset_providers.TfdsTask,
        tfds_name="fake:0.0.0",
        output_features=features,
        text_preprocessor=_dummy_preprocessor({"inputs": "a"}))
    with self.assertRaisesRegex(
        ValueError,
        "Task dataset is missing expected output feature after preprocessing: "
        "targets"):
      task.get_dataset({"inputs": 13}, "train", use_cached=False)

  def test_no_eos(self):
    default_vocab = test_utils.sentencepiece_vocab()
    features = {
        "inputs": seqio.Feature(add_eos=True, vocabulary=default_vocab),
        "targets": seqio.Feature(add_eos=False, vocabulary=default_vocab),
    }
    _add_t5_task(
        "task_no_eos",
        dataset_providers.TfdsTask,
        tfds_name="fake:0.0.0",
        output_features=features)
    self.verify_task_matches_fake_datasets("task_no_eos", use_cached=False)

  def test_task_registry_reset(self):
    """Ensure reset() clears seqio.TaskRegistry."""
    _add_t5_task(
        "t5_task_before_reset",
        dataset_providers.TFExampleTask,
        split_to_filepattern={},
        feature_description={})
    # Assert that task was added to both t5.data.TaskRegistry and
    # seqio.TaskRegistry.
    self.assertSameElements(TaskRegistry.names(), seqio.TaskRegistry.names())
    TaskRegistry.reset()
    _add_t5_task(
        "t5_task_after_reset",
        dataset_providers.TFExampleTask,
        split_to_filepattern={},
        feature_description={})
    # Assert that task was added to both t5.data.TaskRegistry and
    # seqio.TaskRegistry so that they don't diverge after reset() call.
    self.assertSameElements(TaskRegistry.names(), seqio.TaskRegistry.names())


if __name__ == "__main__":
  absltest.main()
