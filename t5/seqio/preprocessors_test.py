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

"""Tests for seqio.preprocessors."""

from absl.testing import absltest
from t5.seqio import dataset_providers
from t5.seqio import preprocessors
from t5.seqio import test_utils
import tensorflow.compat.v2 as tf

assert_dataset = test_utils.assert_dataset
Feature = dataset_providers.Feature


class PreprocessorsTest(absltest.TestCase):

  def test_tokenize(self):
    og_dataset = tf.data.Dataset.from_tensors({
        'prefix': 'This is',
        'suffix': 'a test.'
    })
    output_features = {
        'prefix': Feature(test_utils.MockVocabulary({'This is': [0, 1]})),
        'suffix': Feature(test_utils.MockVocabulary({'a test.': [2, 3]})),
    }

    assert_dataset(
        preprocessors.tokenize(og_dataset, output_features=output_features), {
            'prefix': [0, 1],
            'prefix_pretokenized': 'This is',
            'suffix': [2, 3],
            'suffix_pretokenized': 'a test.'
        })
    assert_dataset(
        preprocessors.tokenize(
            og_dataset, output_features=output_features,
            copy_pretokenized=False),
        {
            'prefix': [0, 1],
            'suffix': [2, 3]
        })


if __name__ == '__main__':
  absltest.main()
