# Copyright 2023 The T5 Authors.
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

"""Tests for equivalent scores between different t5.evaluation.metrics."""

import json
import os

from absl.testing import absltest
from t5.evaluation import metrics
from t5.evaluation import test_utils

# Delta for matching rouge values between the different scorers.
_DELTA = 0.5

_TESTDATA_PREFIX = os.path.join(os.path.dirname(__file__), "testdata")

_LARGE_TARGETS_FILE = os.path.join(_TESTDATA_PREFIX, "target_large.txt")

_LARGE_PREDICTIONS_FILE = os.path.join(_TESTDATA_PREFIX, "prediction_large.txt")

_EXPECTED_RESULTS_FILE = os.path.join(
    _TESTDATA_PREFIX, "expected_bootstrap_results.json"
)


class ScoringTest(test_utils.BaseMetricsTest):

  def setUp(self):
    super(ScoringTest, self).setUp()
    with open(_LARGE_TARGETS_FILE, "r") as f:
      self.targets = f.readlines()
    with open(_LARGE_PREDICTIONS_FILE, "r") as f:
      self.predictions = f.readlines()
    with open(_EXPECTED_RESULTS_FILE, "r") as f:
      self.expected_bootstrap_result = json.load(f)

  def test_rouge_variants(self):
    mean_result = metrics.rouge_mean(self.targets, self.predictions)
    self.assertDictClose(
        mean_result, self.expected_bootstrap_result, delta=_DELTA
    )


if __name__ == "__main__":
  absltest.main()
