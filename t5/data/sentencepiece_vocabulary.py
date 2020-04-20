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
"""SentencePieceVocabulary."""

# from t5.data.vocabularies import SentencePieceVocabulary  # pylint: disable=unused-import
import abc
import hashlib

import gin
from t5.data import vocabularies
import tensorflow.compat.v1 as tf
import tensorflow_text as tf_text

import sentencepiece as sentencepiece_processor


@gin.configurable
class SentencePieceVocabulary(vocabularies.Vocabulary):
  """Wrapper for nlp/sentencepiece encoder.

  Assumes the model was built using flags in `build_sentencepiece_model.sh`,
  which reserve ID=0 is for padding, ID=1 for EOS, and ID=2 for UNK.
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
    # Pass extra_ids if it is specified, otherwise, allow it to be
    # gin-configured through the base class
    kwargs = {"extra_ids": extra_ids} if extra_ids is not None else {}
    super().__init__(**kwargs)

  @property
  def sp_model(self):
    """Retrieve the SPM."""
    if self._sp_model is None:
      # Handle cases where SP can't load the file, but gfile can.
      with tf.gfile.GFile(self._sentencepiece_model_file, "rb") as f:
        self._sp_model = f.read()
    return self._sp_model

  @property
  def sentencepiece_model_file(self):
    return self._sentencepiece_model_file

  @property
  def tokenizer(self):
    """Instantiate and return a tokenizer."""
    if self._tokenizer is None:
      self._tokenizer = sentencepiece_processor.SentencePieceProcessor()
      self._tokenizer.LoadFromSerializedProto(self.sp_model)
    return self._tokenizer

  @property
  def tf_tokenizer(self):
    """Instantiate and return a TF tokenizer."""
    return tf_text.SentencepieceTokenizer(model=self.sp_model)

  @property
  def vocab_size(self):
    """Number of ids (including 0=PAD, 1=EOS, and 2=UNK).

    Returns:
      an integer, the vocabulary size
    """
    return self.tokenizer.GetPieceSize() + self._extra_ids

  def encode(self, s):
    """Encode a python string as a list of integers.

    Args:
      s: a string
    Returns:
      a list of integers (not terminated by EOS)
    """
    return self.tokenizer.EncodeAsIds(s)

  def decode(self, ids):
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

  def encode_tf(self, s):
    """Encode a tf.Scalar string to a tf.Tensor.

    This will be necessary for on-the-fly tokenization.

    Args:
      s: a tf.Scalar with dtype tf.string
    Returns:
      a 1d tf.Tensor with dtype tf.int32
    """
    return self.tf_tokenizer.tokenize(s)

  def decode_tf(self, ids):
    """Decode in TensorFlow.

    Args:
      ids: a 1d tf.Tensor with dtype tf.int32
    Returns:
      a tf Scalar with dtype tf.string
    """
    ids = tf.where_v2(
        tf.less(ids, self.tokenizer.GetPieceSize()),
        ids, self.tokenizer.unk_id())

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
