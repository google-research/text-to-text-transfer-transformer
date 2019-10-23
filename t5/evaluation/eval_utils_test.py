# Copyright 2019 The T5 Authors.
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

"""Tests for t5.evaluation.eval_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

from absl.testing import absltest
import pandas as pd
from t5.evaluation import eval_utils
import tensorflow.compat.v1 as tf


class EvalUtilsTest(absltest.TestCase):

  def test_parse_events_files(self):
    tb_summary_dir = self.create_tempdir()
    summary_writer = tf.summary.FileWriter(tb_summary_dir.full_path)
    tags = [
        "eval/foo_task/accuracy",
        "eval/foo_task/accuracy",
        "loss",
    ]
    values = [1., 2., 3.]
    steps = [20, 30, 40]
    for tag, value, step in zip(tags, values, steps):
      summary = tf.Summary()
      summary.value.add(tag=tag, simple_value=value)
      summary_writer.add_summary(summary, step)
    summary_writer.flush()
    events = eval_utils.parse_events_files(tb_summary_dir.full_path)
    self.assertDictEqual(
        events,
        {
            "eval/foo_task/accuracy": [(20, 1.), (30, 2.)],
            "loss": [(40, 3.)],
        },
    )

  def test_get_eval_metric_values(self):
    events = {
        "eval/foo_task/accuracy": [(20, 1.), (30, 2.)],
        "eval/bar_task/sequence_accuracy": [(10, 3.)],
        "loss": [(40, 3.)],
    }
    eval_values = eval_utils.get_eval_metric_values(events)
    self.assertDictEqual(
        eval_values,
        {
            "foo_task/accuracy": [(20, 1.), (30, 2.)],
            "bar_task/sequence_accuracy": [(10, 3.)],
        }
    )

  def test_glue_average(self):
    score_names = [
        "glue_cola_v002/matthews_corrcoef",
        "glue_sst2_v002/accuracy",
        "glue_mrpc_v002/f1",
        "glue_mrpc_v002/accuracy",
        "glue_stsb_v002/pearson_corrcoef",
        "glue_stsb_v002/spearman_corrcoef",
        "glue_qqp_v002/f1",
        "glue_qqp_v002/accuracy",
        "glue_mnli_matched_v002/accuracy",
        "glue_mnli_mismatched_v002/accuracy",
        "glue_qnli_v002/accuracy",
        "glue_rte_v002/accuracy",
        "super_glue_boolq_v102/accuracy",
        "super_glue_cb_v102/mean_3class_f1",
        "super_glue_cb_v102/accuracy",
        "super_glue_copa_v102/accuracy",
        "super_glue_multirc_v102/f1",
        "super_glue_multirc_v102/exact_match",
        "super_glue_record_v102/f1",
        "super_glue_record_v102/em",
        "super_glue_rte_v102/accuracy",
        "super_glue_wic_v102/accuracy",
        "super_glue_wsc_v102_simple_eval/accuracy",
        "super_glue_average",
        "random/accuracy",
        "glue_average",
    ]
    scores = {k: [(20, n), (30, n*2)] for n, k in enumerate(score_names)}
    scores = eval_utils.compute_avg_glue(scores)
    expected_glue = (
        0 + 1 + (2 + 3)/2. + (4 + 5)/2. + (6 + 7)/2. + (8 + 9)/2. + 10 + 11
    )/8.
    expected_glue_average = [(20, expected_glue), (30, expected_glue * 2)]
    self.assertEqual(scores["glue_average"], expected_glue_average)
    expected_super = (
        12 + (13 + 14)/2. + 15 + (16 + 17)/2. + (18 + 19)/2. + 20 + 21 + 22
    )/8.
    expected_super_average = [(20, expected_super), (30, expected_super * 2)]
    self.assertEqual(scores["super_glue_average"], expected_super_average)
    # Test that keys don't get added when GLUE scores are not computed
    scores = {k: [(20, n), (30, n*2)] for n, k in enumerate(score_names)}
    del scores["glue_cola_v002/matthews_corrcoef"]
    del scores["glue_average"]
    scores = eval_utils.compute_avg_glue(scores)
    self.assertNoCommonElements(scores.keys(), ["glue_average"])

  def test_metric_group_max(self):
    df = pd.DataFrame(
        collections.OrderedDict([
            ("ABC Accuracy", [1., 2., 3., 4.]),
            ("DEF Exact Match", [0., 10., 3., 0.]),
            ("DEF Accuracy", [4., 7., 8., 0.]),
        ]),
        index=[10, 20, 30, 40],
    )
    metric_names = collections.OrderedDict([
        ("metric1", eval_utils.Metric("ABC Accuracy")),
        ("metric2", eval_utils.Metric("DEF Accuracy", "DEF")),
        ("metric3", eval_utils.Metric("DEF Exact Match", "DEF")),
    ])
    metric_max, metric_max_step = eval_utils.metric_group_max(df, metric_names)
    self.assertTrue(metric_max.keys().equals(df.columns))
    self.assertListEqual(list(metric_max.values), [4., 10., 7.])
    self.assertTrue(metric_max_step.keys().equals(df.columns))
    self.assertSequenceEqual(list(metric_max_step.values), [40, 20, 20])

  def test_log_csv(self):
    with self.assertRaises(ValueError):
      eval_utils.log_csv({"foo_task/unknown_metric": [(10, 30.)]})
    metric_keys = list(eval_utils.METRIC_NAMES.keys())
    metric_names = list(eval_utils.METRIC_NAMES.values())
    scores = {
        metric_keys[0]: [(20, 1.), (30, 2.)],
        metric_keys[1]: [(10, 3.)],
        metric_keys[2]: [(10, 4.)],
    }
    output_file = os.path.join(self.create_tempdir().full_path, "results.csv")
    eval_utils.log_csv(scores, output_file=output_file)
    with tf.gfile.Open(output_file) as f:
      output = f.read()
    expected = """step,{},{},{}
10,,3.000,4.000
20,1.000,,
30,2.000,,
max,2.000,3.000,4.000
step,30,10,10""".format(*[m.name for m in metric_names[:3]])
    self.assertEqual(output, expected)


if __name__ == "__main__":
  tf.disable_v2_behavior()
  absltest.main()
