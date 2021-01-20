# Copyright 2021 The T5 Authors.
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
# pylint:disable=g-bare-generic,g-long-lambda

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


def _sum_scores_metric(targets, scores):
  weights = [sum(ord(c) for c in t) for t in targets]
  return {"total_score": (np.array(scores) * np.array(weights)).sum()}


def register_dummy_task(
    task_name: str,
    dataset_fn: Callable[[str, str], tf.data.Dataset],
    output_feature_names: Sequence[str] = ("inputs", "targets"),
    preprocessor=preprocessors.append_eos,
    postprocess_fn=None,
    metrics_fn=None) -> None:
  """Register a dummy task for GetDatasetTest."""
  dataset_providers.TaskRegistry.add(
      task_name,
      source=dataset_providers.FunctionDataSource(
          dataset_fn=dataset_fn, splits=["train", "validation"]),
      preprocessors=[preprocessor],
      postprocess_fn=postprocess_fn,
      output_features={
          # Mock the sentencepiece vocabulary.
          feat: dataset_providers.Feature(mock.Mock(eos_id=True))
          for feat in output_feature_names
      },
      metric_fns=metrics_fn)


def get_mocked_task(
    name: str = "mocked_test",
    predict_metric_fns: Sequence[Callable] = (_sequence_accuracy_metric,),
    score_metric_fns: Sequence[Callable] = ()) -> mock.Mock:
  task = mock.Mock()
  task.name = name
  task.score_metric_fns = list(score_metric_fns)
  task.predict_metric_fns = list(predict_metric_fns)
  task.metric_fns = list(predict_metric_fns) + list(score_metric_fns)
  # Identity postprocess function
  task.postprocess_fn = lambda d, example, is_target: d

  mock_vocab = mock.Mock()
  task.output_features = {"targets": dataset_providers.Feature(mock_vocab)}
  return task


class EvaluationTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.uncalled_fn = mock.Mock()

  def tearDown(self):
    super().tearDown()
    self.uncalled_fn.assert_not_called()

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

    self.assertDictEqual({"task1": [2, 1], "task2": [2]}, cached_targets)
    self.assertDictEqual({"inputs": 3, "targets": 4}, max_sequence_length)
    self.assertCountEqual(["task1", "task2"], cached_task_datasets.keys())
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

  def _evaluate_single_task(self, task):
    id_to_vocab = {5: "e5", 6: "e6", 7: "e7"}
    mock_vocab = task.output_features["targets"].vocabulary
    # Define a dummy decoding logic.
    mock_vocab.decode = lambda ids: " ".join([id_to_vocab[i] for i in ids])

    def mock_init(self):
      self._cached_model_datasets = {task.name: tf.data.Dataset.range(3)}
      self._cached_task_datasets = {task.name: tf.data.Dataset.range(3)}
      self._cached_targets = {task.name: ["e5 e6", "e6", "e7"]}
      self._eval_tasks = [task]
      self._summary_dir = None
      self._summary_writers = {}

    with mock.patch.object(Evaluator, "__init__", new=mock_init):
      evaluator = Evaluator()
      # A dummy score function that always returns the same output.
      predict_fn = (
          lambda x: [(0, [5, 6]), (1, [7]), (2, [7])] if task.predict_metric_fns
          else self.uncalled_fn)
      score_fn = (
          lambda x: [(1, 1), (0, 2), (2, 3)] if task.score_metric_fns
          else self.uncalled_fn)
      all_metrics, _, _ = evaluator.evaluate(
          compute_metrics=True, predict_fn=predict_fn, score_fn=score_fn)
      return all_metrics

  def test_evaluate_single_task_predict(self):
    task = get_mocked_task(
        predict_metric_fns=[_sequence_accuracy_metric], score_metric_fns=[])
    all_metrics = self._evaluate_single_task(task)
    self.assertDictClose(
        {"sequence_accuracy": 2.0 / 3 * 100}, all_metrics[task.name])

  def test_evaluate_single_task_score(self):
    task = get_mocked_task(
        predict_metric_fns=[], score_metric_fns=[_sum_scores_metric])
    all_metrics = self._evaluate_single_task(task)
    self.assertDictClose({"total_score": 1305}, all_metrics[task.name])

  def test_evaluate_single_task_both(self):
    task = get_mocked_task(
        predict_metric_fns=[_sequence_accuracy_metric],
        score_metric_fns=[_sum_scores_metric])
    all_metrics = self._evaluate_single_task(task)
    expected = {"sequence_accuracy": 2.0 / 3 * 100, "total_score": 1305}
    self.assertDictClose(expected, all_metrics[task.name])

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
      self._summary_dir = None
      self._summary_writers = {}

    with mock.patch.object(Evaluator, "__init__", new=mock_init):
      evaluator = Evaluator()
      # A dummy prediction function that always returns the same output.
      # The first example is correct but the second is not.
      predict_fn = lambda x: [(0, [5, 6]), (1, [6, 8])]
      all_metrics, _, _ = evaluator.evaluate(
          compute_metrics=True, predict_fn=predict_fn,
          score_fn=self.uncalled_fn)
      # expected = {"accuracy": 2.0 / 3 * 100}
      expected = {"sequence_accuracy": 50}
      self.assertDictClose(expected, all_metrics[task.name])

  def test_evaluate_single_task_with_postprocessor(self):
    task = get_mocked_task(predict_metric_fns=[_accuracy_metric])
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
      self._summary_dir = None
      self._summary_writers = {}

    with mock.patch.object(Evaluator, "__init__", new=mock_init):
      evaluator = Evaluator()
      # The output tokens will be docoded to ["e5", "e6", "e7"] and
      # postprocessed to [0, 1, 2].
      predict_fn = lambda x: [(0, [5]), (1, [6]), (2, [7])]
      all_metrics, _, _ = evaluator.evaluate(
          compute_metrics=True, predict_fn=predict_fn,
          score_fn=self.uncalled_fn)
      expected = {"accuracy": 100}
      self.assertDictClose(expected, all_metrics[task.name])

  def test_evaluate_mixture(self):
    id_to_vocab = {5: "e5", 6: "e6", 7: "e7"}

    task1 = get_mocked_task(
        name="task1",
        score_metric_fns=[_sum_scores_metric])
    mock_vocab1 = task1.output_features["targets"].vocabulary
    mock_vocab1.decode = lambda ids: " ".join([id_to_vocab[i] for i in ids])

    task2 = get_mocked_task(
        name="task2",
        predict_metric_fns=[_accuracy_metric],
        score_metric_fns=[])
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
          task1.name: ["e5 e6", "e6"],
          task2.name: [0, 1, 2]
      }
      self._eval_tasks = [task1, task2]
      self._summary_dir = None
      self._summary_writers = {}

    with mock.patch.object(Evaluator, "__init__", new=mock_init):
      def predict_fn(ds):
        if ds == mock_ds1:
          return [(0, [5, 6]), (1, [7])]
        elif ds == mock_ds2:
          return [(0, [5]), (1, [6]), (2, [7])]

      def score_fn(ds):
        self.assertEqual(ds, mock_ds1)
        return [(0, 1), (1, 2)]

      evaluator = Evaluator()
      all_metrics, _, _ = evaluator.evaluate(
          compute_metrics=True, predict_fn=predict_fn, score_fn=score_fn)
      expected = {
          task1.name: {
              "sequence_accuracy": 50.0,
              "total_score": 651
          },
          task2.name: {"accuracy": 100}
      }
      self.assertDictClose(expected[task1.name], all_metrics[task1.name])
      self.assertDictClose(expected[task2.name], all_metrics[task2.name])

  def test_short_inputs_targets(self):
    task_name = "short_inputs_targets"
    ds = tf.data.Dataset.from_tensors(
        {"inputs": [7, 8], "targets": [3, 9], "targets_pretokenized": "ex 1"})
    dataset_fn = lambda split, shuffle_files: ds
    register_dummy_task(
        task_name,
        dataset_fn=dataset_fn,
        metrics_fn=[_sequence_accuracy_metric])

    feature_converter = mock.Mock(
        get_model_feature_lengths=lambda x: {k: v+1 for k, v in x.items()})
    sequence_length = {"inputs": 10, "targets": 8}
    evaluator = Evaluator(
        mixture_or_task_name=task_name,
        feature_converter=feature_converter,
        eval_split="validation",
        sequence_length=sequence_length)
    feature_converter.assert_called_with(mock.ANY, sequence_length)
    self.assertDictEqual(
        {"inputs": 11, "targets": 9},
        evaluator.model_feature_lengths)

  def test_no_sequence_length(self):
    task_name = "no_sequence_length"
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

    feature_converter = mock.Mock(
        get_model_feature_lengths=lambda x: {k: v+1 for k, v in x.items()})
    evaluator = Evaluator(
        mixture_or_task_name=task_name,
        feature_converter=feature_converter,
        eval_split="validation")
    # EOS tokens are added, which increases the lengths by 1.
    feature_converter.assert_called_with(mock.ANY, {"inputs": 5, "targets": 3})
    self.assertDictEqual(
        {"inputs": 6, "targets": 4},
        evaluator.model_feature_lengths)

  def test_requires_sequence_length(self):
    task_name = "requires_sequence_length"
    ds = tf.data.Dataset.from_tensors(
        {"inputs": [7, 8], "targets": [3, 9], "targets_pretokenized": "ex 1"})
    dataset_fn = lambda split, shuffle_files: ds

    def preprocessor_with_sequence_length(dataset, sequence_length):
      del sequence_length
      return dataset

    register_dummy_task(
        task_name,
        dataset_fn=dataset_fn,
        # has sequence_length arg
        preprocessor=preprocessor_with_sequence_length,
        metrics_fn=[_sequence_accuracy_metric])

    feature_converter = mock.Mock()

    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "Preprocessor 'preprocessor_with_sequence_length' in task "
        "'requires_sequence_length' has a `sequence_length` argument, making "
        "it incompatible with automatic sequence length detection. Pass a "
        "valid `sequence_length` to `Evaluator` and try again."):
      _ = Evaluator(
          mixture_or_task_name=task_name,
          feature_converter=feature_converter,
          eval_split="validation")

  def test_preprocessor_with_optional_sequence_length(self):
    task_name = "preprocessor_with_optional_sequence_length"
    ds = tf.data.Dataset.from_tensors(
        {"inputs": [7, 8], "targets": [3, 9], "targets_pretokenized": "ex 1"})
    dataset_fn = lambda split, shuffle_files: ds
    register_dummy_task(
        task_name,
        dataset_fn=dataset_fn,
        # `append_eos_after_trim` has an optional sequence_length arg
        preprocessor=preprocessors.append_eos_after_trim,
        metrics_fn=[_sequence_accuracy_metric])

    feature_converter = mock.Mock()
    # Should not raise ValueError
    _ = Evaluator(
        mixture_or_task_name=task_name,
        feature_converter=feature_converter,
        eval_split="validation")

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
    feature_converter = mock.Mock(
        get_model_feature_lengths=lambda x: {"inputs": 4, "targets": 4})
    feature_converter.side_effect = (
        lambda ds, length: utils.trim_and_pad_dataset(ds, {
            "inputs": 4,
            "targets": 4
        })
    )
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
    self.assertEqual(evaluator.cached_targets[task_name], ["ex 1", "ex 2"])
    self.assertDictEqual(
        evaluator.model_feature_lengths,
        {"inputs": 4, "targets": 4})

  def test_predict_fn_called_with_cached_model_datasets(self):
    eval_ds = tf.data.Dataset.range(10)
    task = get_mocked_task()

    def mock_init(self):
      self._cached_model_datasets = {task.name: eval_ds}
      self._cached_task_datasets = {task.name: tf.data.Dataset.range(3)}
      self._eval_tasks = [task]
      self._summary_dir = None
      self._summary_writers = {}

    with mock.patch.object(Evaluator, "__init__", new=mock_init):
      evaluator = Evaluator()
      predict_fn = mock.Mock(return_value=[(0, 1)])
      evaluator.evaluate(
          compute_metrics=False, predict_fn=predict_fn,
          score_fn=self.uncalled_fn)
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
      self._summary_dir = None
      self._summary_writers = {}

    with mock.patch.object(Evaluator, "__init__", new=mock_init):
      evaluator = Evaluator()

      # Dummy predict_fn where only the order is mixed.
      def mixing_order_predict_fn(ds: tf.data.Dataset) -> Sequence[bytes]:
        exs = list(tfds.as_numpy(ds))
        return [exs[2], exs[0], exs[1]]

      all_metrics, all_outputs, _ = evaluator.evaluate(
          compute_metrics=True, predict_fn=mixing_order_predict_fn,
          score_fn=self.uncalled_fn)
      expected_metric = {"sequence_accuracy": 100}
      expected_outputs = [np.array([5]), np.array([6]), np.array([7])]
      self.assertDictEqual(expected_metric, all_metrics[task.name])
      self.assertEqual(expected_outputs, all_outputs[task.name])

  def test_duplicate_metric(self):
    task = get_mocked_task(
        predict_metric_fns=[_accuracy_metric, _accuracy_metric])
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "Duplicate metric key 'accuracy' in Task 'mocked_test'."):
      self._evaluate_single_task(task)

    task = get_mocked_task(
        score_metric_fns=[_sum_scores_metric, _sum_scores_metric])
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "Duplicate metric key 'total_score' in Task 'mocked_test'."):
      self._evaluate_single_task(task)

    task = get_mocked_task(
        predict_metric_fns=[_accuracy_metric],
        score_metric_fns=[lambda *_: {"accuracy": 0}])
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "Duplicate metric key 'accuracy' in Task 'mocked_test'."):
      self._evaluate_single_task(task)

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
    ret = evaluator.evaluate(
        compute_metrics=True, predict_fn=mock.Mock(), score_fn=self.uncalled_fn)
    self.assertEqual(({}, {}, {}), ret)

  def test_log_eval_results(self):
    summary_dir = self.create_tempdir().full_path

    def mock_init(self):
      self._summary_dir = summary_dir
      self._summary_writers = {}

    with mock.patch.object(Evaluator, "__init__", new=mock_init):
      evaluator = Evaluator()

    task_metrics = {"rouge1": 50, "rouge2": 100}
    evaluator._log_eval_results(
        task_metrics=task_metrics, step=1, task_name="log_eval_task")
    task_summary_dir = os.path.join(summary_dir, "log_eval_task")
    event_file = os.path.join(
        task_summary_dir, tf.io.gfile.listdir(task_summary_dir)[0])
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

    self.assertEqual(tag_rouge1, "eval/rouge1")
    self.assertEqual(tag_rouge2, "eval/rouge2")
    self.assertAlmostEqual(rouge1, 50, places=4)
    self.assertAlmostEqual(rouge2, 100, places=4)


if __name__ == "__main__":
  tf.test.main()
