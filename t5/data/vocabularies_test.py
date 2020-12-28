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

"""Tests for t5.data.vocabularies."""

from absl.testing import absltest
from t5.data import test_utils
from t5.data import vocabularies
import tensorflow.compat.v2 as tf

tf.compat.v1.enable_eager_execution()

mock = absltest.mock


def _decode_tf(vocab, tokens):
  return vocab.decode_tf(tf.constant(tokens, tf.int32)).numpy().decode("UTF-8")


class VocabularyTest(absltest.TestCase):

  TEST_STR = "Testing."
  TEST_IDS = [84, 101, 115, 116, 105, 110, 103, 46]

  class AsciiVocab(vocabularies.Vocabulary):

    @property
    def _base_vocab_size(self):
      return 128

    def _encode(self, s):
      return [ord(c) for c in s]

    def _decode(self, ids):
      return "".join("<eos>" if id == 1 else chr(id) for id in ids if id > 0)

    def _encode_tf(self, s):
      return tf.strings.unicode_decode(s, "UTF-8")

    def _decode_tf(self, ids):
      s = tf.strings.unicode_encode(ids, "UTF-8")
      s = tf.strings.regex_replace(s, chr(0), "")
      s = tf.strings.regex_replace(s, chr(1), "<eos>")
      return s

  def test_properties(self):
    test_vocab = self.AsciiVocab(use_eos=False, use_unk=True, extra_ids=10)
    self.assertEqual(test_vocab.extra_ids, 10)
    self.assertEqual(test_vocab.pad_id, 0)
    self.assertIsNone(test_vocab.eos_id)
    self.assertEqual(test_vocab.unk_id, 2)
    self.assertEqual(test_vocab.vocab_size, 128 + 10)

    test_vocab = self.AsciiVocab(use_eos=True, use_unk=False)
    self.assertEqual(test_vocab.extra_ids, 0)
    self.assertEqual(test_vocab.pad_id, 0)
    self.assertEqual(test_vocab.eos_id, 1)
    self.assertIsNone(test_vocab.unk_id)
    self.assertEqual(test_vocab.vocab_size, 128)

  def test_encode(self):
    test_vocab = self.AsciiVocab()
    self.assertSequenceEqual(test_vocab.encode(self.TEST_STR), self.TEST_IDS)
    self.assertSequenceEqual(
        tuple(test_vocab.encode_tf(self.TEST_STR).numpy()),
        self.TEST_IDS)

  def test_decode_unk_and_eos(self):
    test_vocab = self.AsciiVocab(use_eos=True, use_unk=True)
    test_ids = [161] + self.TEST_IDS + [127, 191, 1, 0, 10]
    test_str = "\x02" + self.TEST_STR + "\x7f\x02<eos>"
    self.assertEqual(test_vocab.decode(test_ids), test_str)
    self.assertEqual(_decode_tf(test_vocab, test_ids), test_str)

  def test_decode_unk_only(self):
    test_vocab = self.AsciiVocab(use_eos=False, use_unk=True, extra_ids=35)
    test_ids = [161] + self.TEST_IDS + [127, 191, 1, 33, 1]
    test_str = "\x02" + self.TEST_STR + "\x7f\x02<eos>!<eos>"
    self.assertEqual(test_vocab.decode(test_ids), test_str)
    self.assertEqual(_decode_tf(test_vocab, test_ids), test_str)

  def test_decode_eos_only(self):
    test_vocab = self.AsciiVocab(use_eos=True, use_unk=False)
    test_ids = [161] + self.TEST_IDS + [127, 191, 1, 33, 1]
    test_str = "¡" + self.TEST_STR + "\x7f¿<eos>"
    self.assertEqual(test_vocab.decode(test_ids), test_str)
    self.assertEqual(_decode_tf(test_vocab, test_ids), test_str)

    test_ids = [161] + self.TEST_IDS + [127, 191]
    test_str = "¡" + self.TEST_STR + "\x7f¿"
    self.assertEqual(test_vocab.decode(test_ids), test_str)
    self.assertEqual(_decode_tf(test_vocab, test_ids), test_str)

    test_ids = [1] + self.TEST_IDS
    test_str = "<eos>"
    self.assertEqual(test_vocab.decode(test_ids), test_str)
    self.assertEqual(_decode_tf(test_vocab, test_ids), test_str)

  def test_decode_no_unk_or_eos(self):
    test_vocab = self.AsciiVocab(use_eos=False, use_unk=False)
    test_ids = [161] + self.TEST_IDS +  [127, 191, 1, 33, 1]
    test_str = "¡" + self.TEST_STR + "\x7f¿<eos>!<eos>"
    self.assertEqual(test_vocab.decode(test_ids), test_str)
    self.assertEqual(_decode_tf(test_vocab, test_ids), test_str)

  def test_decode_tf_batch(self):
    test_vocab = self.AsciiVocab(use_eos=True, use_unk=True)
    test_ids = (
        [161] + self.TEST_IDS +  [127, 191, 1, 33, 1],
        [161] + self.TEST_IDS +  [1, 191, 1, 33, 1],
    )
    test_str = (
        "\x02" + self.TEST_STR + "\x7f\x02<eos>",
        "\x02" + self.TEST_STR + "<eos>",
    )
    decoded = [
        dec.decode("UTF-8") for dec in
        test_vocab.decode_tf(tf.constant(test_ids, tf.int32)).numpy()
    ]
    self.assertSequenceEqual(decoded, test_str)


class SentencepieceVocabularyTest(absltest.TestCase):

  TEST_STRING = "this is a test"
  TEST_TOKENS = (11, 8, 6, 3, 8, 6, 3, 5, 10)
  UNK_STRING = " ⁇ "

  def test_vocab(self):
    vocab = test_utils.sentencepiece_vocab()
    self.assertEqual(26, vocab.vocab_size)
    self.assertSequenceEqual(self.TEST_TOKENS, vocab.encode(self.TEST_STRING))
    self.assertEqual(self.TEST_STRING, vocab.decode(self.TEST_TOKENS))
    self.assertSequenceEqual(
        self.TEST_TOKENS,
        tuple(vocab.encode_tf(self.TEST_STRING).numpy()))
    self.assertEqual(self.TEST_STRING, _decode_tf(vocab, self.TEST_TOKENS))

  def test_extra_ids(self):
    vocab = test_utils.sentencepiece_vocab(extra_ids=10)
    self.assertEqual(36, vocab.vocab_size)
    self.assertEqual("v", vocab.decode([25]))
    self.assertEqual(self.UNK_STRING, vocab.decode([35]))
    self.assertEqual(self.UNK_STRING, _decode_tf(vocab, [35]))


  def test_equal(self):
    vocab1 = test_utils.sentencepiece_vocab()
    vocab2 = test_utils.sentencepiece_vocab()
    self.assertEqual(vocab1, vocab2)

  def test_not_equal(self):
    vocab1 = test_utils.sentencepiece_vocab()
    vocab2 = test_utils.sentencepiece_vocab(10)
    self.assertNotEqual(vocab1, vocab2)


class ByteVocabularyTest(absltest.TestCase):

  TEST_STRING = "this is a test"
  TEST_BYTE_IDS = (
      119, 107, 108, 118, 35, 108, 118, 35, 100, 35, 119, 104, 118, 119)

  def test_vocab(self):
    vocab = vocabularies.ByteVocabulary()
    self.assertEqual(259, vocab.vocab_size)
    self.assertSequenceEqual(self.TEST_BYTE_IDS, vocab.encode(self.TEST_STRING))
    self.assertEqual(self.TEST_STRING, vocab.decode(self.TEST_BYTE_IDS))
    self.assertEqual(
        self.TEST_BYTE_IDS,
        tuple(vocab.encode_tf(self.TEST_STRING).numpy()))
    self.assertEqual(self.TEST_STRING, _decode_tf(vocab, self.TEST_BYTE_IDS))

  def test_extra_ids(self):
    vocab = vocabularies.ByteVocabulary(extra_ids=10)
    self.assertEqual(269, vocab.vocab_size)
    self.assertEqual("a", vocab.decode([100]))
    self.assertEqual("", vocab.decode([268]))

  def test_out_of_vocab(self):
    vocab = vocabularies.ByteVocabulary()
    self.assertEqual(259, vocab.vocab_size)
    self.assertEqual("", vocab.decode([260]))

  def test_equal(self):
    vocab1 = vocabularies.ByteVocabulary()
    vocab2 = vocabularies.ByteVocabulary()
    self.assertEqual(vocab1, vocab2)

  def test_not_equal(self):
    vocab1 = vocabularies.ByteVocabulary()
    vocab2 = vocabularies.ByteVocabulary(10)
    self.assertNotEqual(vocab1, vocab2)

if __name__ == "__main__":
  absltest.main()
