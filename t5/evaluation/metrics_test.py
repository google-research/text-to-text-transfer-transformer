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

# Lint as: python3
"""Tests for t5.evaluation.metrics."""

from absl.testing import absltest
import sklearn.metrics

from t5.evaluation import metrics
from t5.evaluation import test_utils


class MetricsTest(test_utils.BaseMetricsTest):

  def test_same_bleu(self):
    ref = "this is a string"
    self.assertDictClose(
        metrics.bleu([ref, ref], [ref, ref]),
        {"bleu": 100})

  def test_different_bleu(self):
    ref = "this is a string"
    self.assertDictClose(
        metrics.bleu([ref, ref], ["", ""]),
        {"bleu": 0})

  def test_multiple_references_bleu(self):
    ref = "this is a string"
    self.assertDictClose(
        metrics.bleu([["", ref], [ref, ""]], [ref, ref]),
        {"bleu": 100})

  def test_same_rouge(self):
    ref = "this is a string"
    self.assertDictClose(
        metrics.rouge([ref, ref], [ref, ref]),
        {"rouge1": 100, "rouge2": 100, "rougeLsum": 100})

  def test_different_rouge(self):
    ref = "this is a string"
    self.assertDictClose(
        metrics.rouge([ref, ref], ["", ""]),
        {"rouge1": 0, "rouge2": 0, "rougeLsum": 0})

  def test_same_squad(self):
    ref = "this is a string"
    self.assertDictClose(
        metrics.squad([["", ref], [ref, ref]], [ref, ref]), {
            "em": 100,
            "f1": 100,
        })

  def test_different_squad(self):
    ref = "this is a string"
    self.assertDictClose(
        metrics.squad([[ref, ref], [ref, ref]], ["", ""]), {
            "em": 0,
            "f1": 0
        })

  def test_squad_big(self):
    self.assertDictClose(
        metrics.squad(
            [
                ["big moose", "hippo"],
                ["correct1"],
                ["correct2.1", "correct2.2"],
                ["a", "b"],
            ],
            [
                "‘a big  Moose!‘",
                "wrong",
                "correct2.2",
                "c",
            ],
        ),
        {"em": 25., "f1": 35.},
        places=2
    )

  def test_squad_small(self):
    self.assertDictClose(
        metrics.squad([["abc abd", "$$$$"]], ["abd"]),
        {"f1": 100 * 2.0 / 3.0, "em": 0.},
    )

  def test_span_squad(self):
    ref = "a string"
    ans_span = "start:2 end:3"
    context = "this is a string! it has the answer."
    self.assertDictClose(
        metrics.span_squad(
            [{"answers": ["", ref], "context": context},
             {"answers": [ref, ref], "context": context}],
            [ans_span, ans_span]),
        {"em": 100, "f1": 100})

  def test_trivia_qa(self):
    self.assertDictClose(
        metrics.trivia_qa(
            [
                ["big moose", "hippo"],
                ["correct1"],
                ["correct2.1", "correct2.2"],
                ["a", "b"],
            ],
            [
                "‘a big  Moose!‘",
                "wrong",
                "correct2.2",
                "c",
            ],
        ),
        {"em": 50., "f1": 50.},
    )

  def test_span_squad_one_word(self):
    ref = "answer"
    ans_span = "start:1 end:1"
    context = "the answer"

    self.assertDictClose(
        metrics.span_squad([{
            "answers": [ref],
            "context": context
        }], [ans_span]), {"em": 100, "f1": 100})

  def test_span_squad_non_numbers(self):

    ref = "answer"
    ans_span = "start:test end:why"
    context = "the answer"

    self.assertDictClose(
        metrics.span_squad([{
            "answers": [ref],
            "context": context
        }], [ans_span]), {"em": 0, "f1": 0})

  def test_sequence_accuracy(self):
    s1 = "this is a string."
    s2 = "this is a completely different string."
    self.assertDictEqual(
        metrics.sequence_accuracy([s1, s2], [s1, s1]),
        {"sequence_accuracy": 50})

  def test_multiclass_f1(self):
    self.assertDictClose(
        metrics.mean_multiclass_f1(num_classes=3)([0, 1, 1, 2], [0, 0, 2, 2]),
        {"mean_3class_f1": 44.44444444444444})

  def test_exact_match(self):
    self.assertDictEqual(
        metrics.exact_match([0, 1], [0, 1]), {"exact_match": 100.0})
    self.assertDictEqual(
        metrics.exact_match([0, 1], [0, 2]), {"exact_match": 0.0})

  def test_pearson_corrcoef(self):
    self.assertDictClose(
        metrics.pearson_corrcoef([0, 2], [0, 1]),
        {"pearson_corrcoef": 100.0})

  def test_spearman_corrcoef(self):
    self.assertDictClose(
        metrics.spearman_corrcoef([0, 2, 1], [0, 1, 2]),
        {"spearman_corrcoef": 50.})

  def test_f1_score_with_invalid(self):
    self.assertDictClose(
        metrics.f1_score_with_invalid([0, 1, 1, 0], [0, 1, 2, 2]),
        {"f1": 50.})

  def test_accuracy(self):
    self.assertDictClose(
        metrics.accuracy([0, 0, 2, 1], [0, 1, 2, 1]),
        {"accuracy": 75.})

  def test_mean_group_metric(self):
    metric_fn = metrics.mean_group_metric(metrics.accuracy)
    self.assertDictClose(
        metric_fn(
            [{"group": "a", "value": 0},
             {"group": "a", "value": 1},
             {"group": "b", "value": 0}],
            [{"value": 0},
             {"value": 0},
             {"value": 1}]),
        {"accuracy": 25.})

  def test_multirc_f1_over_all_answers(self):
    metric_fn = metrics.multirc_f1_over_all_answers
    self.assertDictClose(
        metric_fn(
            [{"group": "a", "value": 1},
             {"group": "a", "value": 1},
             {"group": "b", "value": 0}],
            [{"value": 1},
             {"value": 0},
             {"value": 1}]),
        {"f1": 50.})

  def test_auc(self):
    self.assertDictClose(
        metrics.auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]),
        {"auc-roc": 0.75,
         "auc-pr": 0.8333},
        places=4,
    )

  def test_auc_non_binary(self):
    self.assertDictClose(
        metrics.auc([0.0, 0.2, 0.5, 0.7], [0.1, 0.4, 0.35, 0.8],
                    targets_threshold=0.5),
        {"auc-roc": 0.75,
         "auc-pr": 0.8333},
        places=4,
    )

  def test_sklearn_wrapper(self):
    mae_fn = metrics.sklearn_metrics_wrapper("mean_absolute_error")
    y_true = [[0.5, 1], [-1, 1], [7, -6]]
    y_pred = [[0, 2], [-1, 2], [8, -5]]
    self.assertDictClose(
        mae_fn(y_true, y_pred),
        {"mean_absolute_error": sklearn.metrics.mean_absolute_error(y_true,
                                                                    y_pred)})

    hamming_fn = metrics.sklearn_metrics_wrapper(
        "hamming_loss",
        metric_dict_str="hamming_100x",
        metric_post_process_fn=lambda x: 100 * x)
    y_true = [1, 2, 3, 4]
    y_pred = [2, 2, 3, 4]
    self.assertDictClose(
        hamming_fn(y_true, y_pred),
        {"hamming_100x": 100 * sklearn.metrics.hamming_loss(y_true, y_pred)})

    y_true = [0, 0, 2, 1]
    y_pred = [0, 1, 2, 1]
    matthews_corrcoef_fn = metrics.sklearn_metrics_wrapper(
        "matthews_corrcoef", metric_post_process_fn=lambda x: 100 * x)
    self.assertDictClose(
        matthews_corrcoef_fn(y_true, y_pred),
        {"matthews_corrcoef": 70.})

  def test_rank_classification_default_weights(self):

    # num_classes = 2
    self.assertDictClose(
        metrics.rank_classification(
            [
                # 0
                ((0, 0), True, 1.0),
                ((0, 1), False, 1.0),
                # 1
                ((1, 0), False, 1.0),
                ((1, 1), True, 1.0),
                # 0
                ((2, 0), True, 1.0),
                ((2, 1), False, 1.0),
                # 0
                ((3, 0), True, 1.0),
                ((3, 1), False, 1.0),
            ],
            [
                0.1, 0.5,
                1.0, 1.1,
                0.3, 0.1,
                0.6, 0.5
            ],
            num_classes=2),
        {
            "accuracy": 75.,
            "auc-pr": 50.0,
            "auc-roc": 66.6666667,
            "f1": 66.6666667,
        })

    # num_classes = 3
    self.assertDictClose(
        metrics.rank_classification(
            [
                # 1
                ((0, 0), False, 1.0),
                ((0, 1), True, 1.0),
                ((0, 2), False, 1.0),
                # 0
                ((1, 0), True, 1.0),
                ((1, 1), False, 1.0),
                ((1, 2), False, 1.0),
                # 2
                ((2, 0), False, 1.0),
                ((2, 1), False, 1.0),
                ((2, 2), True, 1.0)
            ],
            [
                0.1, 0.5, 0.0,
                -2, -1, -3,
                3.0, 3.1, 3.2
            ],
            num_classes=3),
        {
            "accuracy": 66.6666667,
            "mean_3class_f1": 55.5555556,
        })

    # num_classes = 3, multi-label
    self.assertDictClose(
        metrics.rank_classification(
            [
                # 1
                ((0, 0), False, 1.0),
                ((0, 1), True, 1.0),
                ((0, 2), False, 1.0),
                # 0, 2
                ((1, 0), True, 1.0),
                ((1, 1), False, 1.0),
                ((1, 2), True, 1.0),
                # 1, 2
                ((2, 0), False, 1.0),
                ((2, 1), True, 1.0),
                ((2, 2), True, 1.0)
            ],
            [
                0.1, 0.5, 0.0,
                -2, -1, -3,
                3.0, 3.1, 3.2
            ],
            num_classes=3),
        {
            "accuracy": 66.6666667,
        })

    # num_classes = None, multi-answer
    self.assertDictClose(
        metrics.rank_classification(
            [
                # 1
                ((0, 0), False, 1.0),
                ((0, 1), True, 1.0),
                # 0, 3
                ((1, 0), True, 1.0),
                ((1, 1), False, 1.0),
                ((1, 2), True, 1.0),
                # 0
                ((2, 0), True, 1.0)
            ],
            [
                0.1, 0.5,
                -2, -1, -3,
                3.0
            ],
            num_classes=None),
        {
            "accuracy": 66.6666667,
        })

  def test_rank_classification_custom_weights(self):
    # num_classes = 2
    self.assertDictClose(
        metrics.rank_classification(
            [
                # 0
                ((0, 0), True, 0.2),
                ((0, 1), False, 0.2),
                # 1
                ((1, 0), False, 1.0),
                ((1, 1), True, 1.0),
                # 0
                ((2, 0), True, 0.8),
                ((2, 1), False, 0.8),
                # 0
                ((3, 0), True, 0.5),
                ((3, 1), False, 0.5),
            ],
            [
                0.1, 0.5,
                1.0, 1.1,
                0.3, 0.1,
                0.6, 0.5
            ],
            num_classes=2),
        {
            "accuracy": 92.0,
            "auc-pr": 83.3333333,
            "auc-roc": 86.6666667,
            "f1": 90.9090909,
        })

    # num_classes = 3
    self.assertDictClose(
        metrics.rank_classification(
            [
                # 1
                ((0, 0), False, 0.2),
                ((0, 1), True, 0.2),
                ((0, 2), False, 0.2),
                # 0
                ((1, 0), True, 0.5),
                ((1, 1), False, 0.5),
                ((1, 2), False, 0.5),
                # 2
                ((2, 0), False, 1.0),
                ((2, 1), False, 1.0),
                ((2, 2), True, 1.0)
            ],
            [
                0.1, 0.5, 0.0,
                -2, -1, -3,
                3.0, 3.1, 3.2
            ],
            num_classes=3),
        {
            "accuracy": 70.5882353,
            "mean_3class_f1": 48.1481481,
        })

    # num_classes = None, multi-answer
    self.assertDictClose(
        metrics.rank_classification(
            [
                # 1
                ((0, 0), False, 0.2),
                ((0, 1), True, 0.2),
                # 0, 3
                ((1, 0), True, 0.5),
                ((1, 1), False, 0.5),
                ((1, 2), True, 0.5),
                # 1
                ((2, 0), True, 1.0)
            ],
            [
                0.1, 0.5,
                -2, -1, -3,
                3.0
            ],
            num_classes=None),
        {
            "accuracy": 70.5882353,
        })

  def test_rank_classification_shuffled(self):
    # num_classes = 2
    self.assertDictClose(
        metrics.rank_classification(
            [
                ((3, 0), True, 0.5),
                ((0, 0), True, 0.2),
                ((1, 0), False, 1.0),
                ((1, 1), True, 1.0),
                ((2, 0), True, 0.8),
                ((2, 1), False, 0.8),
                ((3, 1), False, 0.5),
                ((0, 1), False, 0.2),
            ],
            [
                0.6,
                0.1,
                1.0,
                1.1,
                0.3,
                0.1,
                0.5,
                0.5,
            ],
            num_classes=2),
        {
            "accuracy": 92.0,
            "auc-pr": 83.3333333,
            "auc-roc": 86.6666667,
            "f1": 90.9090909,
        })

    # num_classes = 3
    self.assertDictClose(
        metrics.rank_classification(
            [
                ((0, 0), False, 0.2),
                ((2, 1), False, 1.0),
                ((0, 1), True, 0.2),
                ((1, 0), True, 0.5),
                ((1, 1), False, 0.5),
                ((1, 2), False, 0.5),
                ((0, 2), False, 0.2),
                ((2, 0), False, 1.0),
                ((2, 2), True, 1.0)
            ],
            [
                0.1,
                3.1,
                0.5,
                -2,
                -1,
                -3,
                0.0,
                3.0,
                3.2
            ],
            num_classes=3),
        {
            "accuracy": 70.5882353,
            "mean_3class_f1": 48.1481481,
        })

    # num_classes = None, multi-answer
    self.assertDictClose(
        metrics.rank_classification(
            [
                ((0, 0), False, 0.2),
                ((2, 0), True, 1.0),
                ((0, 1), True, 0.2),
                ((1, 2), True, 0.5),
                ((1, 0), True, 0.5),
                ((1, 1), False, 0.5),
            ],
            [
                0.1,
                3.0,
                0.5,
                -3,
                -2,
                -1,
            ],
            num_classes=None),
        {
            "accuracy": 70.5882353,
        })

  def test_rank_classification_raise(self):
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "`targets` should contain 3 elements but has 2."):
      metrics.rank_classification(
          [
              ((0, 0), True),
              ((0, 1), True),
          ],
          [
              0.1, 0.5
          ],
          num_classes=2)

    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "The first element of `targets` ('idx') should be 2-dimensional. Got "
        "0."):
      metrics.rank_classification(
          [
              (0, True, 1.0),
              (0, True, 1.0),
          ],
          [
              0.1, 0.5
          ],
          num_classes=2)

  def test_coqa_tokenize(self):
    self.assertEqual(metrics._coqa_tokenize("Maru the cat"), ["maru", "cat"])
    self.assertEqual(metrics._coqa_tokenize("Maru  cat"), ["maru", "cat"])
    self.assertEqual(metrics._coqa_tokenize("Maru the cat."), ["maru", "cat"])

  def test_sequence_f1(self):
    self.assertEqual(metrics._sequence_f1([], []), 1.0)
    self.assertEqual(metrics._sequence_f1([], ["cat"]), 0.0)
    self.assertEqual(metrics._sequence_f1(["cat"], []), 0.0)
    self.assertEqual(metrics._sequence_f1(["dog"], ["cat"]), 0.0)
    self.assertAlmostEqual(metrics._sequence_f1(["cat", "dog"], ["cat"]), 2 / 3)
    self.assertAlmostEqual(metrics._sequence_f1(["cat"], ["cat", "dog"]), 2 / 3)

  def test_coqa_f1(self):
    self.assertDictClose(
        metrics.coqa_f1([["jump box"], ["maru"]], ["jump", "cat"]),
        {"f1": 1 / 3})
    self.assertDictClose(
        metrics.coqa_f1([["jump the box"], ["maru"]], ["jump", "cat"]),
        {"f1": 1 / 3})


if __name__ == "__main__":
  absltest.main()
