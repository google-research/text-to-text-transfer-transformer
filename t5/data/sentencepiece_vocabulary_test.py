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

"""Tests for t5.data.sentencepiece_vocabulary."""

from absl.testing import absltest
from t5.data import test_utils
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
tf.enable_eager_execution()

mock = absltest.mock

_UNK_STRING = b" \xe2\x81\x87 "
_TEST_STRING = b"this is a test"
_TEST_TOKENS = (11, 8, 6, 3, 8, 6, 3, 5, 10)


class SentencepieceVocabularyTest(absltest.TestCase):

  def test_vocab(self):
    vocab = test_utils.sentencepiece_vocab()
    self.assertEqual(26, vocab.vocab_size)
    self.assertSequenceEqual(_TEST_TOKENS, vocab.encode(_TEST_STRING))
    self.assertEqual(
        _TEST_STRING,
        tf.compat.as_bytes(vocab.decode(_TEST_TOKENS)))
    self.assertEqual(
        _TEST_TOKENS,
        tuple(vocab.encode_tf(_TEST_STRING).numpy()))
    self.assertEqual(
        _TEST_STRING,
        vocab.decode_tf(_TEST_TOKENS).numpy())

  def test_extra_ids(self):
    vocab = test_utils.sentencepiece_vocab(extra_ids=10)
    self.assertEqual(36, vocab.vocab_size)
    self.assertEqual("v", vocab.decode([25]))
    self.assertEqual(_UNK_STRING, tf.compat.as_bytes(vocab.decode([35])))
    self.assertEqual(_UNK_STRING, vocab.decode_tf([35]).numpy())


if __name__ == "__main__":
  absltest.main()
