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

"""Tests for dump_task."""

from unittest import mock

from absl.testing import flagsaver
from absl.testing import parameterized
import gin
import seqio
from t5.scripts import dump_task
import tensorflow as tf


def get_mocked_task(name: str):
  task = mock.Mock()
  task.name = name
  task.postprocess_fn.return_value = ["test"]
  task.get_dataset.return_value = [{
      "inputs":
          tf.constant([111, 222, 333, 444]),
      "negative_inputs":
          tf.constant([[111, 222, 333, 444], [111, 222, 333, 444]]),
      "targets":
          tf.constant([222])
  }]
  task.mock_vocab = mock.Mock()
  task.mock_vocab.decode_tf.return_value = ["test"]
  task.output_features = {
      "inputs":
          seqio.dataset_providers.Feature(task.mock_vocab),
      "negative_inputs":
          seqio.dataset_providers.Feature(task.mock_vocab, rank=2),
      "targets":
          seqio.dataset_providers.Feature(task.mock_vocab)
  }
  return task


class DumpTaskTest(parameterized.TestCase, tf.test.TestCase):

  def tearDown(self):
    gin.clear_config(clear_constants=True)
    super().tearDown()

  @parameterized.named_parameters(
      dict(testcase_name="512Tokens", value=512),
      dict(testcase_name="256Tokens", value=256))
  def test_sequence_length(self, value: int):
    sequence_lengths = dump_task.sequence_length(value)
    self.assertEqual({"inputs": value, "targets": value}, sequence_lengths)

  @parameterized.named_parameters(
      dict(testcase_name="Single delimiter", value="[CLS] some input"),
      dict(
          testcase_name="Multiple delimiter",
          value="[CLS] some input [SEP] other input"))
  @flagsaver.flagsaver(
      detokenize=True, pretty=True, delimiters=["\\[[A-Z]+\\]"])
  def test_pretty(self, value: str):
    prettied = dump_task.pretty(value)
    self.assertGreaterEqual(prettied.find("\u001b[1m"), 0)  # Bold applied
    self.assertGreaterEqual(prettied.find("\u001b[0m"), 0)  # Reset applied

  @parameterized.named_parameters(
      dict(testcase_name="task", detokenize=True, task="test_task"),
      dict(testcase_name="detokenize_task", detokenize=True, task="test_task"),
      dict(
          testcase_name="postprocess",
          detokenize=True,
          apply_postprocess_fn=True,
          task="test_task"),
      dict(
          testcase_name="mixture",
          detokenize=True,
          apply_postprocess_fn=True,
          mixture="test_mixture"))
  @flagsaver.flagsaver(
      format_string="inputs: {inputs}, negatives: {negative_inputs}, targets: {targets}",
      module_import=[])
  def test_main(self, **flags):
    mock_task = get_mocked_task(flags["task"] if "task" in
                                flags else flags["mixture"])
    self.enter_context(
        mock.patch.object(seqio.TaskRegistry, "get", return_value=mock_task))
    self.enter_context(
        mock.patch.object(seqio.MixtureRegistry, "get", return_value=mock_task))
    with flagsaver.flagsaver(**flags):
      dump_task.main(None)
      if "task" in flags:
        seqio.TaskRegistry.get.assert_called_once_with(flags["task"])
      if "mixture" in flags:
        seqio.MixtureRegistry.get.assert_called_once_with(flags["mixture"])
      mock_task.get_dataset.assert_called_once()
      if "detokenize" in flags:
        # Once per input, negative input and target
        self.assertEqual(mock_task.mock_vocab.decode_tf.call_count, 3)
        if "apply_postprocess_fn" in flags:
          # Once for target
          mock_task.postprocess_fn.assert_called_once()


if __name__ == "__main__":
  tf.test.main()
