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

# Lint as: python3
"""Abstract Vocabulary."""

import abc
import gin
import tensorflow.compat.v1 as tf


@gin.configurable
class Vocabulary(object):
  """Base class for all vocabularies."""

  def __init__(self, extra_ids=0):
    self._extra_ids = extra_ids

  @property
  def vocab_size(self):
    raise NotImplementedError

  @abc.abstractmethod
  def encode(self, s):
    raise NotImplementedError

  @abc.abstractmethod
  def decode(self, ids):
    raise NotImplementedError

  @abc.abstractmethod
  def encode_tf(self, s):
    raise NotImplementedError

  @abc.abstractmethod
  def decode_tf(self, ids):
    raise NotImplementedError

  @property
  def extra_ids(self):
    return self._extra_ids


@gin.configurable
class ByteVocabulary(Vocabulary):
  """Byte level vocabulary.

  Build mappings between Unicode characters and IDs. Encode/decode
  Unicode characeters/IDs based on UTF-8. Reserve ID=0 is for padding,
  ID=1 for EOS, and ID=2 for UNK.
  """

  def __init__(self, extra_ids=None):
    """Create a ByteVocabulary.

    Optionally, specify a number of extra ids to add to the end of the
    vocabulary for use as sentinels.

    Args:
      extra_ids: an optional integer
    """
    self._byte_size = 256
    # The special tokens: 0=PAD, 1=EOS,and 2=UNK
    self._num_special_tokens = 3
    kwargs = {"extra_ids": extra_ids} if extra_ids is not None else {}
    super().__init__(**kwargs)

  def _convert_strings_to_ids(self, s):
    """Convert a python string to integers based on UTF-8 encoding.

    Args:
      s: a string
    Returns:
      ids: a list of integers
    """
    return list(s.encode("utf-8"))

  def _convert_ids_to_strings(self, ids):
    """Convert ids to a python string based on UTF-8 encoding.

    Args:
      ids: a list of integers
    Returns:
      s: a string
    """
    return bytes(ids).decode("utf-8", errors="ignore")

  def _filter_non_string_ids(self, ids):
    """Filter special token ids and extra ids if there are any.

    Args:
      ids: a list of integers
    Returns:
      ids: a list of integers
    """
    lower_bound = self._num_special_tokens
    upper_bound = self._byte_size + self._num_special_tokens
    return [id for id in ids if lower_bound <= id < upper_bound]

  @property
  def vocab_size(self):
    """Number of ids.

    Returns:
      an integer, the vocabulary size
    """
    return self._num_special_tokens + self._byte_size + self._extra_ids

  def encode(self, s):
    """Encode a python string as a list of integers.

    To keep the first few ids for special tokens, increase ids by the number
    of special tokens.

    Args:
      s: a string
    Returns:
      a list of integers (not terminated by EOS)
    """
    ids = self._convert_strings_to_ids(s)
    return [i + self._num_special_tokens for i in ids]

  def decode(self, ids):
    """Decode a list of integers to a python string.

    The special tokens of PAD, EOS, and UNK will not be represented in the
    output string. This is different from the sentencepiece_vocabulary, where
    UNK will show up as a '?' character.

    Args:
      ids: a list of integers (not terminated by EOS)
    Returns:
      a string
    """

    ids = self._filter_non_string_ids(ids)
    ids = [i - self._num_special_tokens for i in ids]
    return self._convert_ids_to_strings(ids)

  def encode_tf(self, s):
    """Encode a tf.Scalar string to a tf.Tensor.

    Args:
      s: a tf.Scalar with dtype tf.string
    Returns:
      a 1d tf.Tensor with dtype tf.int32
    """
    tf_ids = tf.io.decode_raw(s, tf.uint8) + self._num_special_tokens
    return tf.dtypes.cast(tf_ids, tf.int32)

  def decode_tf(self, ids):
    """Decode in TensorFlow.

    Args:
      ids: a 1d tf.Tensor with dtype tf.int32
    Returns:
      a tf Scalar with dtype tf.string
    """
    return tf.py_function(func=self.decode, inp=[ids], Tout=tf.string)

  def __eq__(self, other):
    their_extra_ids = other.extra_ids
    return self.extra_ids == their_extra_ids
