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

"""Tests for t5.data.postprocessors."""

from absl.testing import absltest
import numpy as np
from t5.data import postprocessors


class PostprocessorsTest(absltest.TestCase):

  def test_string_to_float(self):
    self.assertEqual(postprocessors.string_to_float("10"), 10.)
    self.assertEqual(postprocessors.string_to_float("10."), 10.)
    self.assertEqual(postprocessors.string_to_float("asdf"), -1.)
    self.assertEqual(postprocessors.string_to_float("asdf", -2.), -2.)

  def test_lower_text(self):
    self.assertEqual(postprocessors.lower_text("TeST"), "test")

  def test_string_label_to_class_id(self):
    cls = ["one", "two"]
    self.assertEqual(postprocessors.string_label_to_class_id("one", cls), 0)
    self.assertEqual(postprocessors.string_label_to_class_id("two", cls), 1)
    self.assertEqual(postprocessors.string_label_to_class_id("foo", cls), -1)
    self.assertEqual(postprocessors.string_label_to_class_id("foo", cls, 2), 2)

  def test_multirc(self):
    self.assertDictEqual(
        postprocessors.multirc(
            "False",
            example={
                "idx/question": 0,
                "idx/answer": 1
            },
            is_target=True), {
                "group": 0,
                "value": 0
            })
    self.assertDictEqual(
        postprocessors.multirc("True", is_target=False), {"value": 1})

  def test_record(self):
    self.assertDictEqual(
        postprocessors.record(
            "answer",
            example={"answers": [b"a1", b"a2"],
                     "idx/passage": 1,
                     "idx/query": 2},
            is_target=True),
        {"value": ["a1", "a2"], "group": (1, 2)}
    )
    self.assertDictEqual(
        postprocessors.record("answer", is_target=False),
        {"value": "answer"}
    )

  def test_qa(self):
    self.assertEqual(
        postprocessors.qa(
            "answer", example={"answers": [b"a1", b"a2"]}, is_target=True),
        ["a1", "a2"])
    self.assertEqual(postprocessors.qa("answer", is_target=False), "answer")

  def test_span_qa(self):
    self.assertEqual(
        postprocessors.span_qa(
            "answer",
            example={
                "answers": [b"a1", b"a2"],
                "context": b"Full context"
            },
            is_target=True), {
                "answers": ["a1", "a2"],
                "context": "Full context"
            })

    self.assertEqual(
        postprocessors.span_qa("answer", is_target=False), "answer")

  def test_wsc_simple(self):
    self.assertEqual(
        postprocessors.wsc_simple("blah", example={"label": 1}, is_target=True),
        1)
    self.assertEqual(
        postprocessors.wsc_simple(
            "blah", example={"label": -1}, is_target=True), -1)

    self.assertEqual(
        postprocessors.wsc_simple(
            "potato", example={"targets_pretokenized": b"turnip"},
            is_target=False), 0)
    self.assertEqual(
        postprocessors.wsc_simple(
            "turnip", example={"targets_pretokenized": b"turnip"},
            is_target=False), 1)
    self.assertEqual(
        postprocessors.wsc_simple(
            "the cat", example={"targets_pretokenized": b"cat"},
            is_target=False),
        1)
    self.assertEqual(
        postprocessors.wsc_simple(
            "Bob's hat", example={"targets_pretokenized": b"Bob"},
            is_target=False), 0)
    self.assertEqual(
        postprocessors.wsc_simple(
            "Bob's hat",
            example={"targets_pretokenized": b"Bob's hat"},
            is_target=False), 1)
    self.assertEqual(
        postprocessors.wsc_simple(
            "potato", example={"targets_pretokenized": b"Potato"},
            is_target=False), 1)
    self.assertEqual(
        postprocessors.wsc_simple(
            "a potato",
            example={"targets_pretokenized": b"my potato"},
            is_target=False), 1)
    self.assertEqual(
        postprocessors.wsc_simple(
            "fuzzy bunny",
            example={"targets_pretokenized": b"fuzzy hungry bunny"},
            is_target=False), 1)
    self.assertEqual(
        postprocessors.wsc_simple(
            "",
            example={"targets_pretokenized": b"cat"},
            is_target=False), -1)
    self.assertEqual(
        postprocessors.wsc_simple(
            "a this",
            example={"targets_pretokenized": b"cat"},
            is_target=False), -1)

  def test_rank_classification(self):
    self.assertEqual(postprocessors.rank_classification(-13.4), -13.4)

  def test_rank_classification_is_target(self):
    # The example does not have weight feature.
    self.assertEqual(
        postprocessors.rank_classification(
            "blah",
            example={
                "is_correct": False,
                "idx": np.array([10, 1]),
                "targets": [1, 2, 3],
                "passthrough": "ps_value",
            },
            is_target=True), ((10, 1), False, 1, 3))

    self.assertEqual(
        postprocessors.rank_classification(
            "blah",
            example={
                "is_correct": False,
                "idx": np.array([10, 1]),
                "targets": [1, 2, 3],
                "passthrough": "ps_value",
            },
            is_target=True,
            passthrough_feature_keys=["passthrough"]),
        ((10, 1), False, 1, 3, "ps_value"))

    # The example has weight feature.
    self.assertEqual(
        postprocessors.rank_classification(
            "blah",
            example={
                "is_correct": False,
                "idx": np.array([10, 1]),
                "targets": [1, 2, 3],
                "weight": 0,
                "passthrough": ["pt1", "pt2"],
            },
            is_target=True), ((10, 1), False, 0, 3))

    self.assertEqual(
        postprocessors.rank_classification(
            "blah",
            example={
                "is_correct": False,
                "idx": np.array([10, 1]),
                "targets": [1, 2, 3],
                "weight": 0,
                "passthrough": ["pt1", "pt2"],
            },
            is_target=True,
            passthrough_feature_keys=["passthrough"]),
        ((10, 1), False, 0, 3, ["pt1", "pt2"]))

if __name__ == "__main__":
  absltest.main()
