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

"""Tests for t5.data.cache_tasks_main."""

import os

from absl.testing import absltest
import apache_beam as beam
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import equal_to
import t5
from t5.data import cache_tasks_main
import tensorflow.compat.v2 as tf

tf.compat.v1.enable_eager_execution()


test_utils = t5.data.test_utils
TaskRegistry = t5.data.TaskRegistry


class ProcessTaskBeamTest(test_utils.FakeTaskTest):

  def test_get_info(self):
    input_examples = [{"targets": range(10), "inputs": "test"}]

    with TestPipeline() as p:
      pcoll = (p
               | beam.Create(input_examples)
               | cache_tasks_main.GetInfo(num_shards=3))

    assert_that(pcoll, equal_to(
        {
            "num_shards": 3,
            "features_dict": {
                "targets": {"dtype": "int32", "shape": [10]},
                "inputs": {"dtype": "string", "shape": []},
            }
        }))

  def validate_pipeline(self,
                        task_name,
                        expected_task_dir="cached_task",
                        token_preprocessed=False,
                        num_shards=2):
    self.assertTrue(TaskRegistry.get("cached_task").cache_dir)
    task = TaskRegistry.get(task_name)
    self.assertFalse(task.cache_dir)

    with TestPipeline() as p:
      output_dirs = cache_tasks_main.run_pipeline(
          p, ["cached_task", task_name], cache_dir=self.test_data_dir)

    actual_task_dir = os.path.join(self.test_data_dir, task_name)
    expected_task_dir = os.path.join(test_utils.TEST_DATA_DIR,
                                     expected_task_dir)
    expected_tfrecord_files = [
        "train.tfrecord-%05d-of-%05d" % (i, num_shards)
        for i in range(num_shards)
    ]
    expected_auxiliary_files = [
        "stats.train.json", "info.train.json"
    ]

    if "validation" in task.splits:
      expected_tfrecord_files.append("validation.tfrecord-00000-of-00001")
      expected_auxiliary_files.extend(
          ["stats.validation.json", "info.validation.json"])
    self.assertEqual([actual_task_dir], output_dirs)
    self.assertCountEqual(
        expected_tfrecord_files + expected_auxiliary_files,
        tf.io.gfile.listdir(actual_task_dir))

    for fname in expected_auxiliary_files:
      self.assertEqual(
          tf.io.gfile.GFile(os.path.join(expected_task_dir,
                                         fname)).read().replace(
                                             '"num_shards": 2',
                                             f'"num_shards": {num_shards}'),
          tf.io.gfile.GFile(os.path.join(actual_task_dir,
                                         fname)).read().replace(", ", ","))

    # Add COMPLETED file so that we can load `uncached_task`.
    test_utils.mark_completed(self.test_data_dir, task_name)

    # Load task.
    uncached_task = TaskRegistry.get(task_name)

    # Check datasets.
    test_utils.verify_task_matches_fake_datasets(
        uncached_task,
        use_cached=True,
        splits=task.splits,
        token_preprocessed=token_preprocessed)

  def test_tfds_pipeline(self):
    self.validate_pipeline("uncached_task", token_preprocessed=True)

  def test_text_line_pipeline(self):
    self.validate_pipeline("text_line_task")

  def test_general_pipeline(self):
    self.validate_pipeline("general_task", num_shards=1)

  def test_tf_example_pipeline(self):
    self.validate_pipeline("tf_example_task")

  def test_v3_pipeline(self):
    self.validate_pipeline("task_v3", num_shards=1, token_preprocessed=True)

  def test_v3_cache_before_tokenization_pipeline(self):
    self.validate_pipeline(
        "task_v3_tokenized_postcache",
        expected_task_dir="cached_untokenized_task",
        num_shards=1,
        token_preprocessed=True)

  def test_overwrite(self):
    with TestPipeline() as p:
      _ = cache_tasks_main.run_pipeline(
          p, ["uncached_task"], cache_dir=self.test_data_dir, overwrite=True)
    # Add COMPLETED file so that we can load `uncached_task`.
    test_utils.mark_completed(self.test_data_dir, "uncached_task")

    actual_task_dir = os.path.join(self.test_data_dir, "uncached_task")
    stat_old = tf.io.gfile.stat(
        os.path.join(actual_task_dir, "train.tfrecord-00000-of-00002"))

    with TestPipeline() as p:
      _ = cache_tasks_main.run_pipeline(
          p, ["uncached_task"], cache_dir=self.test_data_dir, overwrite=True)

    stat_new = tf.io.gfile.stat(
        os.path.join(actual_task_dir, "train.tfrecord-00000-of-00002"))

    self.assertGreater(stat_new.mtime_nsec, stat_old.mtime_nsec)


if __name__ == "__main__":
  absltest.main()
