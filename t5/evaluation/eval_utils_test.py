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

"""Tests for t5.evaluation.eval_utils."""

import collections
import os

from absl.testing import absltest
import numpy as np
import pandas as pd
import seqio
from t5.evaluation import eval_utils
import tensorflow.compat.v1 as tf


class EvalUtilsTest(absltest.TestCase):

  def test_parse_events_files(self):
    tb_summary_dir = self.create_tempdir()
    with tf.Graph().as_default():
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

  def test_parse_events_files_seqio(self):
    tb_summary_dir = self.create_tempdir()
    metrics = [{"accuracy": seqio.metrics.Scalar(1.)},
               {"accuracy": seqio.metrics.Scalar(2.)}]
    steps = [20, 30]

    logger = seqio.TensorBoardLoggerV1(tb_summary_dir.full_path)
    for metric, step in zip(metrics, steps):
      logger(task_name="foo_task", metrics=metric, step=step,
             dataset=tf.data.Dataset.range(0), inferences={}, targets=[])

    events = eval_utils.parse_events_files(
        os.path.join(tb_summary_dir.full_path, "foo_task"),
        seqio_summaries=True)

    self.assertDictEqual(
        events,
        {
            "eval/accuracy": [(20, 1.), (30, 2.)],
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

  def test_get_eval_metric_values_seqio(self):
    events = {
        "eval/accuracy": [(20, 1.), (30, 2.)],
        "eval/sequence_accuracy": [(10, 3.)],
        "loss": [(40, 3.)],
    }
    eval_values = eval_utils.get_eval_metric_values(
        events, task_name="foo_task")
    self.assertDictEqual(
        eval_values,
        {
            "foo_task/accuracy": [(20, 1.), (30, 2.)],
            "foo_task/sequence_accuracy": [(10, 3.)],
        }
    )

  def test_glue_average(self):
    mets = eval_utils.METRIC_NAMES.items()
    glue_metric_names = [
        v.name for k, v in mets if k.startswith("glue") and "average" not in k
    ]
    super_glue_metric_names = [
        v.name for k, v in mets if k.startswith("super") and "average" not in k
    ]
    extra_metric_names = ["Fake metric", "Average GLUE Score"]
    columns = glue_metric_names + super_glue_metric_names + extra_metric_names
    n_total_metrics = len(columns)
    df = pd.DataFrame(
        [np.arange(n_total_metrics), 2*np.arange(n_total_metrics)],
        columns=columns,
    )
    df = eval_utils.compute_avg_glue(df)
    expected_glue = (
        0 + 1 + (2 + 3)/2. + (4 + 5)/2. + (6 + 7)/2. + (8 + 9)/2. + 10 + 11
    )/8.
    self.assertSequenceAlmostEqual(
        df["Average GLUE Score"], [expected_glue, 2*expected_glue]
    )
    expected_super = (
        12 + (13 + 14)/2. + 15 + (16 + 17)/2. + (18 + 19)/2. + 20 + 21 + 22
    )/8.
    self.assertSequenceAlmostEqual(
        df["Average SuperGLUE Score"], [expected_super, 2*expected_super]
    )
    del df["CoLA"]
    del df["Average GLUE Score"]
    df = eval_utils.compute_avg_glue(df)
    self.assertNoCommonElements(df.columns, ["Average GLUE Score"])

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
    metric_names = list(eval_utils.METRIC_NAMES.values())
    df = pd.DataFrame(
        collections.OrderedDict([
            (metric_names[0].name, [np.nan, 1., 2.]),
            (metric_names[1].name, [3., np.nan, np.nan]),
            (metric_names[2].name, [4., np.nan, np.nan]),
        ]),
        index=[10, 20, 30],
    )
    df.index.name = "step"
    output_file = os.path.join(self.create_tempdir().full_path, "results.csv")
    eval_utils.log_csv(df, output_file=output_file)
    with tf.io.gfile.GFile(output_file) as f:
      output = f.read()
    expected = """step,{},{},{}
10,,3.000,4.000
20,1.000,,
30,2.000,,
max,2.000,3.000,4.000
step,30,10,10""".format(*[m.name for m in metric_names[:3]])
    self.assertEqual(output, expected)


if __name__ == "__main__":
  absltest.main()
