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

"""Preprocessors for SeqIO Tasks."""

from t5.seqio import utils
import tensorflow.compat.v2 as tf


def tokenize(dataset, output_features, copy_pretokenized=True):
  """Encode specified string features.

  Passes through non-string features unchanged. Optionally passes through copy
  of original features with "_pretokenized" suffix added to the key.

  Args:
    dataset: a tf.data.Dataset
    output_features: a dict of Feature objects; their vocabulary attribute will
      be used to tokenize the specified features.
    copy_pretokenized: bool, whether to pass through copies of original features
      with "_pretokenized" suffix added to the key.

  Returns:
    a tf.data.Dataset
  """

  def my_fn(features):
    """Encode all specified feature that are strings and return a dictionary.

    Args:
      features: a dictionary

    Returns:
      a dictionary
    """
    ret = {}
    for k, v in features.items():
      if k in output_features:
        if copy_pretokenized:
          ret[f'{k}_pretokenized'] = v
        v = output_features[k].vocabulary.encode_tf(v)
      ret[k] = v
    return ret

  return dataset.map(my_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)


@utils.map_over_dataset
def print_dataset(ex):
  """tf.Print dataset fields for debugging purposes."""
  return {k: tf.Print(v, [v], k + ': ') for k, v in ex.items()}
