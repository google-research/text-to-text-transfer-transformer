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
"""Vocabularies."""

import abc
import hashlib
from typing import Iterable, List, Optional, Sequence

import tensorflow.compat.v2 as tf
import tensorflow_text as tf_text

import sentencepiece as sentencepiece_processor


PAD_ID = 0
EOS_ID = 1
UNK_ID = 2


class Vocabulary(metaclass=abc.ABCMeta):
  """Abstract class for all vocabularies.

  Subclasses must implement methods for converting between strings and tokens
  both in pure python (`_encode`/`_decode`) and in TensorFlow
  (`_encode_tf`/`_decode_tf`).

  Subclasses are responsible for reserving PAD_ID=0 as well as EOS_ID and UNK_ID
  if `use_eos` and `use_unk` are True, respectively.

  `_base_vocab_size` should account for PAD, EOS, and UNK but not `extra_ids`.
  """

  def __init__(
      self,
      extra_ids: int = 0,
      use_eos: bool = True,
      use_unk: bool = True):
    """Vocabulary constructor.

    Args:
      extra_ids: The number of extra IDs to reserve.
      use_eos: Whether to stop decoding at EOS_ID=1.
      use_unk: Whether to replace tokens out of range with UNK_ID=2.
    """
    self._extra_ids = extra_ids
    self._use_eos = use_eos
    self._use_unk = use_unk

  @property
  def eos_id(self) -> Optional[int]:
    return EOS_ID if self._use_eos else None

  @property
  def pad_id(self) -> int:
    return PAD_ID

  @property
  def unk_id(self) -> Optional[int]:
    return UNK_ID if self._use_unk else None

  @property
  def extra_ids(self) -> int:
    return self._extra_ids

  @property
  def vocab_size(self) -> int:
    """Vocabulary size, including extra ids."""
    return self._base_vocab_size + self.extra_ids

  @abc.abstractproperty
  def _base_vocab_size(self) -> int:
    """Vocabulary size, excluding extra ids but including PAD/EOS/UNK."""
    raise NotImplementedError

  @abc.abstractmethod
  def _encode(self, s: str) -> Sequence[int]:
    raise NotImplementedError

  def encode(self, s: str) -> Sequence[int]:
    """Tokenizes string to an int sequence, without adding EOS."""
    return self._encode(s)

  @abc.abstractmethod
  def _decode(self, ids):
    raise NotImplementedError

  def decode(self, ids: Iterable[int]):
    """Detokenizes int32 iterable to a string, up through first EOS."""
    clean_ids = list(ids)

    if self.unk_id is not None:
      clean_ids = [
          self.unk_id if i >= self._base_vocab_size else i
          for i in clean_ids
      ]

    if self.eos_id is not None and self.eos_id in clean_ids:
      clean_ids = clean_ids[:clean_ids.index(self.eos_id) + 1]

    return self._decode(clean_ids)

  @abc.abstractmethod
  def _encode_tf(self, s: tf.Tensor) -> tf.Tensor:
    raise NotImplementedError

  def encode_tf(self, s: tf.Tensor) -> tf.Tensor:
    """Tokenizes string Scalar to an int32 Tensor, without adding EOS."""
    return self._encode_tf(s)

  @abc.abstractmethod
  def _decode_tf(self, ids: tf.Tensor) -> tf.Tensor:
    raise NotImplementedError

  def decode_tf(self, ids: tf.Tensor) -> tf.Tensor:
    """Detokenizes int32 N-D Tensor to string (N-1)-D Tensor, through first EOS.
    """
    clean_ids = ids

    if self.unk_id is not None:
      clean_ids = tf.where(
          tf.less(clean_ids, self._base_vocab_size), clean_ids, self.unk_id)

    if self.eos_id is not None:
      # Replace everything after the first EOS_ID with PAD_ID.
      after_eos = tf.cumsum(
          tf.cast(tf.equal(clean_ids, self.eos_id), tf.int32),
          exclusive=True, axis=-1)
      clean_ids = tf.where(tf.cast(after_eos, tf.bool), self.pad_id, clean_ids)

    return self._decode_tf(clean_ids)


class SentencePieceVocabulary(Vocabulary):
  """Wrapper for nlp/sentencepiece encoder.

  Assumes the model was built using flags to reserve ID=0 for padding, ID=1 for
  EOS, and ID=2 for UNK.

  """

  def __init__(self, sentencepiece_model_file, extra_ids=None):
    """Create a SentencePieceVocabulary.

    Optionally, specify a number of extra ids to add to the end of the
    vocabulary for use as sentinels.

    Args:
      sentencepiece_model_file: a string
      extra_ids: an optional integer
    """
    self._sentencepiece_model_file = sentencepiece_model_file
    self._tokenizer = None
    self._sp_model = None
    super().__init__(use_eos=True, use_unk=True, extra_ids=extra_ids)

  def _load_model(self):
    """Load SPM and Python tokenizer."""
    # Handle cases where SP can't load the file, but gfile can.
    with tf.io.gfile.GFile(self._sentencepiece_model_file, "rb") as f:
      self._sp_model = f.read()
    # Load Python tokenizer and ensure the EOS and PAD IDs are correct.
    # TODO(adarob): Add support for arbitrary EOS and PAD IDs.
    self._tokenizer = sentencepiece_processor.SentencePieceProcessor()
    self._tokenizer.LoadFromSerializedProto(self._sp_model)
    if self._tokenizer.pad_id() != 0:
      raise ValueError(
          f"Vocabulary PAD ID must be 0, got {self._tokenizer.pad_id()}")
    if self._tokenizer.eos_id() != 1:
      raise ValueError(
          f"Vocabulary EOS ID must be 1, got {self._tokenizer.eos_id()}")
    if self._tokenizer.unk_id() != 2:
      raise ValueError(
          f"Vocabulary UNK ID must be 2, got {self._tokenizer.unk_id()}")

  @property
  def sp_model(self):
    """Retrieve the SPM."""
    if self._sp_model is None:
      self._load_model()
    return self._sp_model

  @property
  def sentencepiece_model_file(self):
    return self._sentencepiece_model_file

  @property
  def tokenizer(self):
    """Returns the Python tokenizer."""
    if not self._tokenizer:
      self._load_model()
    return self._tokenizer

  @property
  def tf_tokenizer(self):
    """Instantiate and return a TF tokenizer."""
    return tf_text.SentencepieceTokenizer(model=self.sp_model)


  @property
  def _base_vocab_size(self):
    """Number of ids (including 0=PAD, 1=EOS, and 2=UNK).

    Returns:
      an integer, the vocabulary size
    """
    return self.tokenizer.GetPieceSize()

  def _encode(self, s):
    """Encode a python string as a list of integers.

    Args:
      s: a string
    Returns:
      a list of integers (not terminated by EOS)
    """
    return self.tokenizer.EncodeAsIds(s)

  def _decode(self, ids):
    """Decode a list of integers to a python string.

    Args:
      ids: a list of integers (not terminated by EOS)
    Returns:
      a string
    """
    # convert all the extra ids (sentinels) to UNK=2
    ids = [
        self.tokenizer.unk_id() if i >= self.tokenizer.GetPieceSize()
        else i for i in ids]
    return self.tokenizer.DecodeIds(ids)

  def _encode_tf(self, s):
    """Encode a tf.Scalar string to a tf.Tensor.

    This will be necessary for on-the-fly tokenization.

    Args:
      s: a tf.Scalar with dtype tf.string
    Returns:
      a 1d tf.Tensor with dtype tf.int32
    """
    return self.tf_tokenizer.tokenize(s)

  def _decode_tf(self, ids):
    """Decode in TensorFlow.

    Args:
      ids: a 1d tf.Tensor with dtype tf.int32
    Returns:
      a tf Scalar with dtype tf.string
    """
    return self.tf_tokenizer.detokenize(ids)

  def __eq__(self, other):
    try:
      their_md5 = hashlib.md5(other.sp_model).hexdigest()
      their_extra_ids = other.extra_ids
    # If other has no sp_model/extra_ids attribute, we can't test for equality
    except AttributeError:
      return False
    our_md5 = hashlib.md5(self.sp_model).hexdigest()
    return our_md5 == their_md5 and self.extra_ids == their_extra_ids


class ByteVocabulary(Vocabulary):
  """Byte level vocabulary.

  Build mappings between Unicode characters and IDs. Encode/decode
  Unicode characeters/IDs based on UTF-8. Reserve ID=0 is for padding,
  ID=1 for EOS, and ID=2 for UNK.
  """

  def __init__(self, extra_ids: int = 0):
    """Create a ByteVocabulary.

    Optionally, specify a number of extra ids to add to the end of the
    vocabulary for use as sentinels.

    Args:
      extra_ids: an optional integer
    """
    self._byte_size = 256
    # The special tokens: 0=PAD, 1=EOS,and 2=UNK
    self._num_special_tokens = 3
    super().__init__(use_eos=True, use_unk=True, extra_ids=extra_ids)

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
  def _base_vocab_size(self):
    """Number of ids.

    Returns:
      an integer, the vocabulary size
    """
    return self._num_special_tokens + self._byte_size

  def _encode(self, s):
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

  def _decode(self, ids):
    """Decode a list of integers to a python string.

    The special tokens of PAD, EOS, and UNK will not be represented in the
    output string. This is different from the SentencePieceVocabulary, where
    UNK will show up as a '?' character.

    Args:
      ids: a list of integers (not terminated by EOS)
    Returns:
      a string
    """

    ids = self._filter_non_string_ids(ids)
    ids = [i - self._num_special_tokens for i in ids]
    return self._convert_ids_to_strings(ids)

  def _encode_tf(self, s):
    """Encode a tf.Scalar string to a tf.Tensor.

    Args:
      s: a tf.Scalar with dtype tf.string
    Returns:
      a 1d tf.Tensor with dtype tf.int32
    """
    tf_ids = tf.io.decode_raw(s, tf.uint8) + self._num_special_tokens
    return tf.dtypes.cast(tf_ids, tf.int32)

  def _decode_tf(self, ids):
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
