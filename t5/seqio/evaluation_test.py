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

"""Tests for seqio.evaluation."""

import functools
import os
from typing import Callable, Sequence
from unittest import mock

import numpy as np
from t5.seqio import dataset_providers
from t5.seqio import evaluation
from t5.seqio import preprocessors
from t5.seqio import test_utils
from t5.seqio import utils
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

Evaluator = evaluation.Evaluator

# For faster testing.
tf.compat.v1.enable_eager_execution()


def _string_label_to_class_id_postprocessor(
    string_label, label_classes, default=-1, **unused_kwargs):
  """Returns index of string_label in label_classes or default if not found."""
  if string_label in label_classes:
    return label_classes.index(string_label)
  else:
    return default


def _sequence_accuracy_metric(targets, predictions):
  seq_acc = 100 * np.mean([p == t for p, t in zip(predictions, targets)])
  return {"sequence_accuracy": seq_acc}


def _accuracy_metric(targets, predictions):
  acc = 100 * np.mean(
      [np.all(p == t) for p, t in zip(predictions, targets)])
  return {"accuracy": acc}


def register_dummy_task(
    task_name: str,
    dataset_fn: Callable[[str, str], tf.data.Dataset],
    output_feature_names: Sequence[str] = ("inputs", "targets"),
    postprocess_fn=None,
    metrics_fn=None) -> None:
  """Register a dummy task for GetDatasetTest."""
  dataset_providers.TaskRegistry.add(
      task_name,
      source=dataset_providers.FunctionDataSource(
          dataset_fn=dataset_fn, splits=["train", "validation"]),
      preprocessors=[preprocessors.append_eos_after_trim],
      postprocess_fn=postprocess_fn,
      output_features={
          # Mock the sentencepiece vocabulary.
          feat: dataset_providers.Feature(mock.Mock(eos_id=True))
          for feat in output_feature_names
      },
      metric_fns=metrics_fn)


def get_mocked_task(
    name: str = "mocked_test",
    metric_fns: Sequence[Callable] = (_sequence_accuracy_metric,)) -> mock.Mock:
  task = mock.Mock()
  task.name = name
  task.metric_fns = list(metric_fns)
  # Identity postprocess function
  task.postprocess_fn = lambda d, example, is_target: d

  mock_vocab = mock.Mock()
  task.output_features = {"targets": dataset_providers.Feature(mock_vocab)}
  return task


class EvaluationTest(tf.test.TestCase):

  def assertDictClose(self, a, b, delta=None, places=None):
    self.assertCountEqual(a.keys(), b.keys())
    for k in a:
      try:
        self.assertAlmostEqual(a[k], b[k], delta=delta, places=places)
      except AssertionError as e:
        raise AssertionError(str(e) + " for key '%s'" % k)

  def test_get_valid_eval_tasks(self):
    task_no_metrics = mock.Mock(splits=("train", "validation"), metric_fns=[])
    task_no_split = mock.Mock(splits=("train"), metric_fns=[lambda x: x])
    valid_task = mock.Mock(
        splits=("train", "validation"), metric_fns=[lambda x: x])
    self.assertSequenceEqual(
        evaluation.get_valid_eval_tasks(
            [task_no_metrics, task_no_split, valid_task], "validation"),
        [valid_task])

  def test_get_targets_and_examples(self):
    # pylint:disable=g-long-lambda
    def _task_from_tensor_slices(name, tensor_slices, label_classes):
      return dataset_providers.Task(
          name,
          dataset_providers.FunctionDataSource(
              lambda split, shuffle_files:
              tf.data.Dataset.from_tensor_slices(tensor_slices),
              splits=("validation")),
          preprocessors=[utils.map_over_dataset(lambda ex: {
              "inputs": tf.range(ex["inputs_lengths"]),
              "targets": tf.range(ex["targets_lengths"]),
              "targets_pretokenized": ex["targets_pretokenized"],
          })],
          postprocess_fn=functools.partial(
              _string_label_to_class_id_postprocessor,
              label_classes=label_classes),
          output_features={"inputs": dataset_providers.Feature(mock.Mock()),
                           "targets": dataset_providers.Feature(mock.Mock())}
      )
    task1 = _task_from_tensor_slices(
        "task1",
        {
            "inputs_lengths": [3, 2],
            "targets_lengths": [2, 3],
            "targets_pretokenized": ["e6", "e5"],
        },
        ("e4", "e5", "e6"))
    task2 = _task_from_tensor_slices(
        "task2",
        {
            "inputs_lengths": [1],
            "targets_lengths": [4],
            "targets_pretokenized": ["e4"],
        },
        ("e2", "e3", "e4"))
    cached_targets, cached_task_datasets, max_sequence_length = (
        evaluation.get_targets_and_examples(
            [task1, task2],
            lambda t: t.get_dataset(
                split="validation", sequence_length=None, shuffle=False))
        )

    self.assertDictEqual(cached_targets, {"task1": [2, 1], "task2": [2]})
    self.assertDictEqual(max_sequence_length, {"inputs": 3, "targets": 4})
    self.assertCountEqual(cached_task_datasets.keys(), ["task1", "task2"])
    self.assertLen(cached_task_datasets["task1"], 2)
    self.assertLen(cached_task_datasets["task2"], 1)
    expected_task1_examples = [
        {"inputs": [0, 1, 2], "targets": [0, 1], "targets_pretokenized": "e6"},
        {"inputs": [0, 1], "targets": [0, 1, 2], "targets_pretokenized": "e5"}
    ]
    expected_task2_examples = [
        {"inputs": [0], "targets": [0, 1, 2, 3], "targets_pretokenized": "e4"},
    ]
    test_utils.assert_dataset(cached_task_datasets["task1"],
                              expected_task1_examples)
    test_utils.assert_dataset(cached_task_datasets["task2"],
                              expected_task2_examples)
    # pylint:enable=g-long-lambda

  def test_evaluate_single_task(self):
    task = get_mocked_task()
    id_to_vocab = {5: "e5", 6: "e6", 7: "e7"}
    mock_vocab = task.output_features["targets"].vocabulary
    # Define a dummy decoding logic.
    mock_vocab.decode = lambda ids: " ".join([id_to_vocab[i] for i in ids])

    def mock_init(self):
      self._cached_model_datasets = {task.name: tf.data.Dataset.range(3)}
      self._cached_task_datasets = {task.name: tf.data.Dataset.range(3)}
      self._cached_targets = {task.name: ["e5 e6", "e6", "e7"]}
      self._eval_tasks = [task]
      self._summary_writer = None

    with mock.patch.object(Evaluator, "__init__", new=mock_init):
      evaluator = Evaluator()
      # A dummy prediction function which always returns the same output.
      # The output tokens will be decoded into ["e5 e6", "e7", "e7"].
      predict_fn = lambda x: [(0, [5, 6]), (1, [7]), (2, [7])]
      _, all_metrics = evaluator.evaluate(
          compute_metrics=True, predict_fn=predict_fn)
      expected = {"sequence_accuracy": 2.0 / 3 * 100}
      self.assertDictClose(expected, all_metrics[task.name][0])

  def test_evaluate_non_string(self):
    task = get_mocked_task()
    mock_vocab = task.output_features["targets"].vocabulary
    # Identity decode function
    mock_vocab.decode = lambda ids: ids

    def mock_init(self):
      # Dummy datasets
      self._cached_model_datasets = {task.name: tf.data.Dataset.range(2)}
      # Dummy task datasets.
      self._cached_task_datasets = {task.name: tf.data.Dataset.range(2)}
      self._cached_targets = {task.name: [[5, 6], [6, 7]]}
      self._eval_tasks = [task]
      self._summary_writer = None

    with mock.patch.object(Evaluator, "__init__", new=mock_init):
      evaluator = Evaluator()
      # A dummy prediction function which always returns the same output.
      # The first example is correct but the second is not.
      predict_fn = lambda x: [(0, [5, 6]), (1, [6, 8])]
      _, all_metrics = evaluator.evaluate(
          compute_metrics=True, predict_fn=predict_fn)
      # expected = {"accuracy": 2.0 / 3 * 100}
      expected = {"sequence_accuracy": 50}
      self.assertDictClose(expected, all_metrics[task.name][0])

  def test_evaluate_single_task_with_postprocessor(self):
    task = get_mocked_task(metric_fns=[_accuracy_metric])
    task.postprocess_fn = functools.partial(
        _string_label_to_class_id_postprocessor,
        label_classes=["e5", "e6", "e7"])

    id_to_vocab = {5: "e5", 6: "e6", 7: "e7"}
    mock_vocab = task.output_features["targets"].vocabulary
    mock_vocab.decode = lambda ids: id_to_vocab[ids[0]]

    def mock_init(self):
      self._cached_model_datasets = {task.name: tf.data.Dataset.range(3)}
      self._cached_task_datasets = {task.name: tf.data.Dataset.range(3)}
      self._cached_targets = {task.name: [0, 1, 2]}
      self._eval_tasks = [task]
      self._summary_writer = None

    with mock.patch.object(Evaluator, "__init__", new=mock_init):
      evaluator = Evaluator()
      # The output tokens will be docoded to ["e5", "e6", "e7"] and
      # postprocessed to [0, 1, 2].
      predict_fn = lambda x: [(0, [5]), (1, [6]), (2, [7])]
      _, all_metrics = evaluator.evaluate(
          compute_metrics=True, predict_fn=predict_fn)
      expected = {"accuracy": 100}
      self.assertDictClose(expected, all_metrics[task.name][0])

  def test_evaluate_mixture(self):
    id_to_vocab = {5: "e5", 6: "e6", 7: "e7"}

    task1 = get_mocked_task()
    mock_vocab1 = task1.output_features["targets"].vocabulary
    mock_vocab1.decode = lambda ids: " ".join([id_to_vocab[i] for i in ids])

    task2 = get_mocked_task(metric_fns=[_accuracy_metric])
    task2.postprocess_fn = functools.partial(
        _string_label_to_class_id_postprocessor,
        label_classes=["e5", "e6", "e7"])
    mock_vocab2 = task2.output_features["targets"].vocabulary
    mock_vocab2.decode = lambda ids: id_to_vocab[ids[0]]

    mock_ds1 = tf.data.Dataset.range(2)
    mock_ds2 = tf.data.Dataset.range(3)

    def mock_init(self):
      self._cached_model_datasets = {
          task1.name: mock_ds1,
          task2.name: mock_ds2,
      }
      self._cached_task_datasets = {
          task1.name: tf.data.Dataset.range(2),
          task2.name: tf.data.Dataset.range(3),
      }
      self._cached_targets = {
          task1.name: ["e5 e6", "e6", "e7"],
          task2.name: [0, 1, 2]
      }
      self._eval_tasks = [task1, task2]
      self._summary_writer = None

    with mock.patch.object(Evaluator, "__init__", new=mock_init):
      def predict_fn(ds):
        if ds == mock_ds1:
          return [(0, [5, 6]), (1, [7]), (2, [7])]
        elif ds == mock_ds2:
          return [(0, [5]), (1, [6]), (2, [7])]

      evaluator = Evaluator()
      _, all_metrics = evaluator.evaluate(
          compute_metrics=True, predict_fn=predict_fn)
      expected = {
          task1.name: [{"sequence_accuracy": 2.0 / 3 * 100}],
          task2.name: [{"accuracy": 100}]
      }
      self.assertDictClose(expected[task1.name][0], all_metrics[task1.name][0])
      self.assertDictClose(expected[task2.name][0], all_metrics[task2.name][0])

  def test_short_inputs_targets(self):
    task_name = "short_inputs_targets"
    x = [{"inputs": [7, 8], "targets": [3, 9], "targets_pretokenized": "ex 1"}]
    dtypes = {
        "inputs": tf.int32,
        "targets": tf.int32,
        "targets_pretokenized": tf.string
    }
    shapes = {"inputs": [None], "targets": [None], "targets_pretokenized": []}
    ds = tf.data.Dataset.from_generator(
        lambda: x, output_types=dtypes, output_shapes=shapes)
    dataset_fn = lambda split, shuffle_files: ds
    register_dummy_task(
        task_name,
        dataset_fn=dataset_fn,
        metrics_fn=[_sequence_accuracy_metric])

    feature_converter = mock.Mock()
    sequence_lengths = {"inputs": 10, "targets": 8}
    _ = Evaluator(
        mixture_or_task_name=task_name,
        feature_converter=feature_converter,
        eval_split="validation",
        sequence_lengths=sequence_lengths)
    feature_converter.assert_called_with(mock.ANY, sequence_lengths)

  def test_no_sequence_lengths(self):
    task_name = "no_sequence_lengths"
    x = [{
        "inputs": [7, 8],
        "targets": [3, 9],
        "targets_pretokenized": "ex 1"
    }, {
        "inputs": [8, 4, 5, 6],
        "targets": [4],
        "targets_pretokenized": "ex 2"
    }]
    dtypes = {
        "inputs": tf.int32,
        "targets": tf.int32,
        "targets_pretokenized": tf.string
    }
    shapes = {"inputs": [None], "targets": [None], "targets_pretokenized": []}
    ds = tf.data.Dataset.from_generator(
        lambda: x, output_types=dtypes, output_shapes=shapes)
    dataset_fn = lambda split, shuffle_files: ds
    register_dummy_task(
        task_name,
        dataset_fn=dataset_fn,
        metrics_fn=[_sequence_accuracy_metric])

    feature_converter = mock.Mock()
    _ = Evaluator(
        mixture_or_task_name=task_name,
        feature_converter=feature_converter,
        eval_split="validation")
    # EOS tokens are added, which increases the lengths by 1.
    feature_converter.assert_called_with(mock.ANY, {"inputs": 5, "targets": 3})

  def test_caching(self):
    task_name = "caching"
    x = [{
        "inputs": [7, 8],
        "targets": [3, 9],
        "targets_pretokenized": "ex 1"
    }, {
        "inputs": [8, 4],
        "targets": [4],
        "targets_pretokenized": "ex 2"
    }]
    dtypes = {
        "inputs": tf.int32,
        "targets": tf.int32,
        "targets_pretokenized": tf.string
    }
    shapes = {"inputs": [None], "targets": [None], "targets_pretokenized": []}
    ds = tf.data.Dataset.from_generator(
        lambda: x, output_types=dtypes, output_shapes=shapes)
    dataset_fn = lambda split, shuffle_files: ds
    register_dummy_task(
        task_name,
        dataset_fn=dataset_fn,
        metrics_fn=[_sequence_accuracy_metric])

    # Feature converter that just pads "inputs" and "targets".
    feature_converter = lambda ds, length: utils.trim_and_pad_dataset(ds, {
        "inputs": 4,
        "targets": 4
    })
    evaluator = Evaluator(
        mixture_or_task_name=task_name,
        feature_converter=feature_converter,
        eval_split="validation")
    expected_task_examples = [{
        "inputs": [7, 8, 1],
        "targets": [3, 9, 1],
        "targets_pretokenized": b"ex 1"
    }, {
        "inputs": [8, 4, 1],
        "targets": [4, 1],
        "targets_pretokenized": b"ex 2"
    }]
    expected_examples = [{
        "inputs": [7, 8, 1, 0],
        "targets": [3, 9, 1, 0],
        "targets_pretokenized": b"ex 1"
    }, {
        "inputs": [8, 4, 1, 0],
        "targets": [4, 1, 0, 0],
        "targets_pretokenized": b"ex 2"
    }]

    test_utils.assert_dataset(
        evaluator._cached_task_datasets[task_name], expected_task_examples)

    # _cached_model_datasets are enumerated. Remove the index for assertion.
    eval_ds = evaluator._cached_model_datasets[task_name].map(lambda i, ds: ds)
    test_utils.assert_dataset(eval_ds, expected_examples)
    self.assertEqual(evaluator._cached_targets[task_name], ["ex 1", "ex 2"])

  def test_predict_fn_called_with_cached_model_datasets(self):
    eval_ds = tf.data.Dataset.range(10)
    task = get_mocked_task()

    def mock_init(self):
      self._cached_model_datasets = {task.name: eval_ds}
      self._cached_task_datasets = {task.name: tf.data.Dataset.range(3)}
      self._eval_tasks = [task]
      self._summary_writer = None

    with mock.patch.object(Evaluator, "__init__", new=mock_init):
      evaluator = Evaluator()
      predict_fn = mock.Mock(return_value=[(0, 1)])
      evaluator.evaluate(compute_metrics=False, predict_fn=predict_fn)
      predict_fn.assert_called_with(eval_ds)

  def test_order_preservation(self):
    task = get_mocked_task()
    id_to_vocab = {5: "e5", 6: "e6", 7: "e7"}
    mock_vocab = task.output_features["targets"].vocabulary
    mock_vocab.decode = lambda ids: id_to_vocab[ids[0]]

    ds = tf.data.Dataset.from_tensor_slices([[5], [6], [7]])

    def mock_init(self):
      self._cached_model_datasets = {task.name: ds.enumerate()}
      # Dummy task datasets
      self._cached_task_datasets = {task.name: tf.data.Dataset.range(3)}
      self._cached_targets = {task.name: ["e5", "e6", "e7"]}
      self._eval_tasks = [task]
      self._summary_writer = None

    with mock.patch.object(Evaluator, "__init__", new=mock_init):
      evaluator = Evaluator()

      # Dummy predict_fn where only the order is mixed.
      def mixing_order_predict_fn(ds: tf.data.Dataset) -> Sequence[bytes]:
        exs = list(tfds.as_numpy(ds))
        return [exs[2], exs[0], exs[1]]

      all_outputs, all_metrics = evaluator.evaluate(
          compute_metrics=True, predict_fn=mixing_order_predict_fn)
      expected_metric = {"sequence_accuracy": 100}
      expected_outputs = [np.array([5]), np.array([6]), np.array([7])]
      self.assertDictEqual(expected_metric, all_metrics[task.name][0])
      self.assertEqual(expected_outputs, all_outputs[task.name])

  def test_task_with_no_metrics_fn(self):
    task_name = "no_metrics_task"
    x = [{"targets_pretokenized": "ex 1"}]
    dtypes = {"targets_pretokenized": tf.string}
    shapes = {"targets_pretokenized": []}
    ds = tf.data.Dataset.from_generator(
        lambda: x, output_types=dtypes, output_shapes=shapes)
    dataset_fn = lambda split, shuffle_files: ds
    register_dummy_task(task_name, dataset_fn=dataset_fn, metrics_fn=[])
    evaluator = Evaluator(mixture_or_task_name=task_name)
    ret = evaluator.evaluate(compute_metrics=True, predict_fn=mock.Mock())
    self.assertEqual(({}, {}), ret)

  def test_log_eval_results(self):
    summary_dir = self.create_tempdir().full_path

    def mock_init(self):
      with tf.compat.v1.Graph().as_default():
        self._summary_writer = tf.compat.v1.summary.FileWriter(summary_dir)

    with mock.patch.object(Evaluator, "__init__", new=mock_init):
      evaluator = Evaluator()

    task_metrics = [{"rouge1": 50, "rouge2": 100}]
    evaluator._log_eval_results(
        task_metrics=task_metrics, step=1, task_name="log_eval_task")
    event_file = os.path.join(summary_dir, tf.io.gfile.listdir(summary_dir)[0])
    # First event is boilerplate
    serialized_events = list(tfds.as_numpy(
        tf.data.TFRecordDataset(event_file)))[1:]
    event1 = tf.compat.v1.Event.FromString(
        serialized_events[0]).summary.value[0]
    rouge1 = event1.simple_value
    tag_rouge1 = event1.tag
    event2 = tf.compat.v1.Event.FromString(
        serialized_events[1]).summary.value[0]
    rouge2 = event2.simple_value
    tag_rouge2 = event2.tag

    self.assertEqual(tag_rouge1, "eval/log_eval_task/rouge1")
    self.assertEqual(tag_rouge2, "eval/log_eval_task/rouge2")
    self.assertAlmostEqual(rouge1, 50, places=4)
    self.assertAlmostEqual(rouge2, 100, places=4)


if __name__ == "__main__":
  tf.test.main()
