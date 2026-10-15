# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Read the value belonging to each tag in a TensorBoard event."""

import os

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from t5.evaluation import eval_utils
import tensorflow.compat.v1 as tf


class EventValuesTest(parameterized.TestCase):

  def make_value(self, tag, value, tensor):
    if tensor:
      return tf.Summary.Value(
          tag=tag, tensor=tf.make_tensor_proto(value, dtype=tf.float64)
      )
    return tf.Summary.Value(tag=tag, simple_value=value)

  def write_events(self, path, events, header=True):
    with tf.io.TFRecordWriter(path) as writer:
      if header:
        writer.write(tf.Event(file_version='brain.Event:2').SerializeToString())
      for event in events:
        writer.write(event.SerializeToString())

  @parameterized.product(first_tensor=[False, True], second_tensor=[False, True])
  def test_multiple_tags_retain_their_own_values(self, first_tensor, second_tensor):
    directory = self.create_tempdir().full_path
    values = [
        self.make_value('eval/accuracy', 0.25, first_tensor),
        self.make_value('eval/loss', 3.5, second_tensor),
        self.make_value('eval/zero', 0.0, first_tensor),
    ]
    self.write_events(
        os.path.join(directory, 'events.mixed'),
        [
            tf.Event(step=7, summary=tf.Summary(value=values)),
            tf.Event(step=8, summary=tf.Summary(value=list(reversed(values)))),
        ],
    )
    actual = eval_utils.parse_events_files(directory, seqio_summaries=True)
    self.assertEqual(set(actual), {'eval/accuracy', 'eval/loss', 'eval/zero'})
    for tag, expected in [
        ('eval/accuracy', 0.25),
        ('eval/loss', 3.5),
        ('eval/zero', 0.0),
    ]:
      self.assertEqual([x.step for x in actual[tag]], [7, 8])
      np.testing.assert_array_equal([x.value for x in actual[tag]], [expected] * 2)

  @parameterized.product(tensor=[False, True], header=[False, True])
  def test_single_value_with_optional_header(self, tensor, header):
    directory = self.create_tempdir().full_path
    self.write_events(
        os.path.join(directory, 'events.single'),
        [
            tf.Event(
                step=2,
                summary=tf.Summary(value=[self.make_value('eval/first', -0.5, tensor)]),
            ),
            tf.Event(
                step=3,
                summary=tf.Summary(
                    value=[self.make_value('eval/second', 0.75, tensor)]
                ),
            ),
        ],
        header=header,
    )
    actual = eval_utils.parse_events_files(directory, seqio_summaries=True)
    self.assertEqual(actual['eval/first'], [(2, -0.5)])
    self.assertEqual(actual['eval/second'], [(3, 0.75)])

  @parameterized.parameters(False, True)
  def test_metadata_only_and_empty_files(self, seqio_summaries):
    directory = self.create_tempdir().full_path
    self.write_events(os.path.join(directory, 'events.metadata'), [tf.Event(step=4)])
    self.write_events(os.path.join(directory, 'events.empty'), [], header=False)
    self.assertEqual(
        eval_utils.parse_events_files(directory, seqio_summaries=seqio_summaries), {}
    )

  @parameterized.parameters(False, True)
  def test_legacy_scalar_events_and_multiple_files(self, seqio_summaries):
    directory = self.create_tempdir().full_path
    for name, step, amount in [('one', 1, 2.0), ('two', 3, -4.0)]:
      self.write_events(
          os.path.join(directory, 'events.' + name),
          [
              tf.Event(
                  step=step,
                  summary=tf.Summary(value=[self.make_value(name, amount, False)]),
              )
          ],
      )
    self.write_events(
        os.path.join(directory, 'ignored'),
        [tf.Event(summary=tf.Summary(value=[self.make_value('ignored', 99, False)]))],
    )
    self.assertEqual(
        eval_utils.parse_events_files(directory, seqio_summaries=seqio_summaries),
        {'one': [(1, 2.0)], 'two': [(3, -4.0)]},
    )

  def test_tensor_dtype_and_shape_are_preserved(self):
    directory = self.create_tempdir().full_path
    self.write_events(
        os.path.join(directory, 'events.tensor'),
        [
            tf.Event(
                step=6,
                summary=tf.Summary(
                    value=[
                        self.make_value('scalar', 0.125, True),
                        self.make_value('vector', [1.5, -2.5], True),
                    ]
                ),
            )
        ],
    )
    actual = eval_utils.parse_events_files(directory, seqio_summaries=True)
    self.assertEqual(actual['vector'][0].value.dtype, np.float64)
    np.testing.assert_array_equal(actual['vector'][0].value, [1.5, -2.5])
    self.assertEqual(actual['scalar'][0].value.shape, ())

  @parameterized.parameters(False, True)
  def test_truncated_tail_keeps_complete_events(self, seqio_summaries):
    directory = self.create_tempdir().full_path
    path = os.path.join(directory, 'events.truncated')
    self.write_events(
        path,
        [
            tf.Event(
                step=9,
                summary=tf.Summary(
                    value=[self.make_value('complete', 5.0, seqio_summaries)]
                ),
            )
        ],
    )
    with open(path, 'ab') as file:
      file.write(b'broken-record')
    self.assertEqual(
        eval_utils.parse_events_files(directory, seqio_summaries=seqio_summaries),
        {'complete': [(9, 5.0)]},
    )


if __name__ == '__main__':
  absltest.main()
