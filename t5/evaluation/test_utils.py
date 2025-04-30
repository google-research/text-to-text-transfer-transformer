# Copyright 2024 The T5 Authors.
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

"""Testing utilities for the evaluation package."""

from absl.testing import absltest


class BaseMetricsTest(absltest.TestCase):

  def assertDictClose(self, a, b, delta=None, places=None):
    self.assertCountEqual(a.keys(), b.keys())
    for k in a:
      try:
        self.assertAlmostEqual(a[k], b[k], delta=delta, places=places)
      except AssertionError as e:
        raise AssertionError(str(e) + " for key '%s'" % k)

  def assertDictContainsSubset(self, expected_subset, actual_set):
    self.assertContainsSubset(expected_subset.keys(), actual_set.keys())
    for k in expected_subset:
      try:
        self.assertEqual(expected_subset[k], actual_set[k])
      except AssertionError as e:
        raise AssertionError(str(e) + " for key '%s'" % k) from None
