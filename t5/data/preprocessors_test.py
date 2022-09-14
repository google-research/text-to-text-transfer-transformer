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

"""Tests for from t5.preprocessors."""

import functools

from absl.testing import absltest
import gin
import seqio
from seqio import test_utils
from t5.data import preprocessors as prep
import tensorflow.compat.v2 as tf

tf.compat.v1.enable_eager_execution()

mock = absltest.mock
assert_dataset = test_utils.assert_dataset


class PreprocessorsTest(tf.test.TestCase):

  def test_regular_noise_mask(self):
    length = 800
    span_length = 2
    noise_density = 0.25
    noise_mask = prep.regular_noise_mask(
        length=length,
        noise_density=noise_density,
        seeds=[(0, 1), (2, 3)],
        min_span_length=span_length,
        max_span_length=span_length)
    num_masked = tf.reduce_sum(tf.cast(noise_mask, tf.int32))
    self.assertEqual(self.evaluate(num_masked), length * noise_density)

  def test_random_prefix_noise_mask(self):
    for _ in range(100):
      length = 10
      noise_density = 0.5
      noise_mask = prep.random_prefix_noise_mask(
          length=length,
          noise_density=noise_density,
          seeds=[(0, 1)])
      first = noise_mask[0]
      last = noise_mask[-1]
      self.assertTrue(self.evaluate(first))
      self.assertFalse(self.evaluate(last))

  def test_random_spans_helper(self):
    input_length = 64
    noise_density = 0.20
    mean_noise_span_lengths = [2.0, 1.0]
    expected_outputs = [(70, 22), (63, 27)]
    for mean_length, expected_output in zip(mean_noise_span_lengths,
                                            expected_outputs):
      output = prep.random_spans_helper(input_length, noise_density,
                                        mean_length, 1, 1)
      self.assertAllEqual(output, expected_output)

  def test_random_spans_noise_mask(self):
    length = 32
    noise_density = 0.25
    mean_noise_span_length = 2.0
    # there should be 4 noise spans with a total length of 8.
    noise_mask = prep.random_spans_noise_mask(
        length, noise_density, [(1, 2), (3, 4)], mean_noise_span_length)
    output = self.evaluate(tf.cast(noise_mask, tf.int32))
    expected_output = [
        0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1]
    self.assertAllEqual(output, expected_output)

  def test_random_spans_noise_mask_with_roll(self):
    """Test random_spans_noise_mask with roll on a fixed sample+seed."""
    # Generate a random mask with and without rolling.
    noise_mask_values = []
    for random_roll in (False, True):
      noise_mask = prep.random_spans_noise_mask(length=32,
                                                noise_density=0.25,
                                                seeds=[(1, 2), (3, 4)],
                                                mean_noise_span_length=3,
                                                random_roll=random_roll)
      noise_mask_values += [self.evaluate(tf.cast(noise_mask, tf.int32))]

    # Assert the only difference between the masks is a roll, shifted by 20.
    mask_no_roll = noise_mask_values[0]
    mask_yes_roll = tf.roll(noise_mask_values[1], shift=20, axis=0)
    self.assertAllEqual(mask_yes_roll, mask_no_roll)

  def test_random_spans_noise_mask_with_roll_avg(self):
    """Test that the empirical mask density is close to the desired density."""
    noise_density = 0.15
    total_masked = 0
    total_lengths = 0
    # Sample 50 random masks and count how many positions are masked.
    for i in range(50):
      span_len = 3
      length = 16 + (i % span_len)  # Vary mask length, keeping it short.
      noise_mask = prep.random_spans_noise_mask(length=length,
                                                noise_density=noise_density,
                                                seeds=[(1+i, 2), (3+i, 4)],
                                                mean_noise_span_length=span_len,
                                                random_roll=True)
      output = self.evaluate(tf.cast(noise_mask, tf.int32))
      total_masked += output.sum()
      total_lengths += length

    empirical_ratio = total_masked / total_lengths
    self.assertAllClose(empirical_ratio, noise_density, atol=0.01)

  def test_random_spans_noise_mask_no_corruption(self):
    length = 32
    noise_density = 0.0
    mean_noise_span_length = 2.0
    noise_mask = prep.random_spans_noise_mask(length, noise_density, [(1, 2),
                                                                      (3, 4)],
                                              mean_noise_span_length)
    output = self.evaluate(tf.cast(noise_mask, tf.int32))
    expected_output = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0
    ]
    self.assertAllEqual(output, expected_output)

  def test_noise_token_to_sentinel(self):
    vocabulary = test_utils.MockVocabulary({'foo': [10]}, vocab_size=1000)
    tokens = tf.constant([10, 11, 12, 13, 14, 15])
    noise_mask = tf.constant([True, True, False, False, True, False])
    expected_output = [999, 999, 12, 13, 999, 15]
    output = self.evaluate(prep.noise_token_to_sentinel(
        tokens, noise_mask, vocabulary, ()))
    self.assertAllEqual(output, expected_output)

  def test_noise_span_to_sentinel(self):
    vocabulary = test_utils.MockVocabulary({'foo': [10]}, vocab_size=1000)
    tokens = tf.constant([10, 11, 12, 13, 14, 15])
    noise_mask = tf.constant([True, True, False, False, True, False])
    expected_output = [999, 12, 13, 999, 15]
    output = self.evaluate(prep.noise_span_to_sentinel(
        tokens, noise_mask, vocabulary, ()))
    self.assertAllEqual(output, expected_output)

  def test_nonnoise_span_to_sentinel(self):
    vocabulary = test_utils.MockVocabulary({'foo': [10]}, vocab_size=1000)
    tokens = tf.constant([10, 11, 12, 13, 14, 15])
    noise_mask = tf.constant([True, True, False, False, True, False])
    expected_output = [10, 11, 999, 14, 999]
    output = self.evaluate(prep.nonnoise_span_to_sentinel(
        tokens, noise_mask, vocabulary, ()))
    self.assertAllEqual(output, expected_output)

  def test_noise_span_to_unique_sentinel(self):
    vocabulary = test_utils.MockVocabulary({'foo': [10]}, vocab_size=1000)
    tokens = tf.constant([10, 11, 12, 13, 14, 15])
    noise_mask = tf.constant([True, True, False, False, True, False])
    expected_output = [999, 12, 13, 998, 15]
    output = self.evaluate(prep.noise_span_to_unique_sentinel(
        tokens, noise_mask, vocabulary, ()))
    self.assertAllEqual(output, expected_output)

  def test_drop_noise_tokens(self):
    vocabulary = test_utils.MockVocabulary({'foo': [10]}, vocab_size=1000)
    tokens = tf.constant([10, 11, 12, 13, 14, 15])
    noise_mask = tf.constant([True, True, False, False, True, False])
    expected_output = [12, 13, 15]
    output = self.evaluate(prep.drop_noise_tokens(
        tokens, noise_mask, vocabulary, ()))
    self.assertAllEqual(output, expected_output)

  def test_drop_nonnoise_tokens(self):
    vocabulary = test_utils.MockVocabulary({'foo': [10]}, vocab_size=1000)
    tokens = tf.constant([10, 11, 12, 13, 14, 15])
    noise_mask = tf.constant([True, True, False, False, True, False])
    expected_output = [10, 11, 14]
    output = self.evaluate(prep.drop_nonnoise_tokens(
        tokens, noise_mask, vocabulary, ()))
    self.assertAllEqual(output, expected_output)

  def test_permute_noise_tokens(self):
    tf.random.set_seed(55)
    vocabulary = test_utils.MockVocabulary({'foo': [10]}, vocab_size=1000)
    tokens = tf.constant([10, 11, 12, 13, 14, 15])
    noise_mask = tf.constant([True, True, False, False, True, False])
    expected_output = [10, 14, 12, 13, 11, 15]
    output = self.evaluate(prep.permute_noise_tokens(
        tokens, noise_mask, vocabulary, [(0, 1)]))
    self.assertAllEqual(output, expected_output)
    tf.random.set_seed(None)

  def test_noise_token_to_gathered_token(self):
    vocabulary = test_utils.MockVocabulary({'foo': [10]}, vocab_size=1000)
    tokens = tf.constant([10, 11, 12, 13, 14, 15])
    noise_mask = tf.constant([True, True, False, False, True, False])
    expected_output = [13, 14, 12, 13, 10, 15]
    output = self.evaluate(prep.noise_token_to_gathered_token(
        tokens, noise_mask, vocabulary, [(55, 56)]))
    self.assertAllEqual(output, expected_output)

  def test_noise_token_to_random_token(self):
    vocabulary = test_utils.MockVocabulary({'foo': [10]}, vocab_size=1000)
    tokens = tf.constant([10, 11, 12, 13, 14, 15])
    noise_mask = tf.constant([True, True, False, False, True, False])
    expected_output = [961, 553, 12, 13, 60, 15]

    output = self.evaluate(prep.noise_token_to_random_token(
        tokens, noise_mask, vocabulary, seeds=[(55, 56)]))
    self.assertAllEqual(output, expected_output)

  def test_noise_token_to_random_token_or_sentinel(self):
    vocabulary = test_utils.MockVocabulary({'foo': [10]}, vocab_size=1000)
    tokens = tf.constant(list(range(10)))
    noise_mask = tf.constant(
        [True, True, False, False, True, False, True, True, True, True])
    expected_output = [999, 348, 2, 3, 108, 5, 999, 999, 999, 999]
    output = self.evaluate(prep.noise_token_to_random_token_or_sentinel(
        tokens, noise_mask, vocabulary,
        seeds=[(55, 56), (57, 58)], random_prob=0.2))
    self.assertAllEqual(output, expected_output)

  def test_translate(self):
    og_dataset = tf.data.Dataset.from_tensor_slices(
        {'en': ['That is good.'], 'de': ['Das ist gut.']}
    )

    dataset = prep.translate(og_dataset, 'en', 'de')
    assert_dataset(
        dataset,
        {
            'inputs': 'translate English to German: That is good.',
            'targets': 'Das ist gut.',
        }
    )

  def test_summarize(self):
    og_dataset = tf.data.Dataset.from_tensor_slices(
        {'article': ['An article.'], 'highlights': ['A summary.']}
    )

    dataset = prep.summarize(og_dataset, 'article', 'highlights')
    assert_dataset(
        dataset,
        {'inputs': 'summarize: An article.', 'targets': 'A summary.'},
    )

  def assertStringEqual(self, a, b):
    self.assertTrue(tf.equal(a, b), '%s != %s' % (a, b))

  def test_pad_punctuation(self):
    self.assertStringEqual(
        ' " This is a string with " punctuation ( 1845 - 1986 ) " . ',
        prep._pad_punctuation(
            '"This  is a string with "punctuation (1845-1986) ".'))

  def test_pad_punctuation_i18n(self):
    self.assertStringEqual(
        ' " Introducción ( la vídeo ) " . ',
        prep._pad_punctuation('"Introducción (la vídeo)".'))

  def test_span_answer(self):
    self.assertStringEqual(
        'start: 2 end: 3',
        prep._span_answer(tf.constant('Called the Denver Broncos.'),
                          tf.constant('Denver Broncos')))
    # Not found.
    self.assertStringEqual(
        '',
        prep._span_answer(tf.constant('Called the Denver Broncos.'),
                          tf.constant('Denver Bronscos')))

  def test_squad(self):
    og_dataset = tf.data.Dataset.from_tensors({
        'id': 'testid',
        'context': 'Some context.',
        'question': 'A question?',
        'answers': {
            'text': ['The answer.', 'Another answer.'],
        }
    })

    dataset = prep.squad(og_dataset)
    assert_dataset(
        dataset, {
            'id': 'testid',
            'inputs': 'question: A question ? context: Some context . ',
            'targets': 'The answer . ',
            'context': 'Some context . ',
            'question': 'A question ? ',
            'answers': ['The answer . ', 'Another answer . '],
        })

  def test_pad_nonspaced_languages(self):
    dataset = tf.data.Dataset.from_tensor_slices(
        {'text': ['Hello there. 你好吗？']})
    dataset = prep.pad_nonspaced_languages(dataset)
    assert_dataset(
        dataset,
        {
            'text': 'Hello there. 你 好 吗 ？',
        })

  def test_triviaqa(self):
    answers = ['key', 'keys']
    contexts = [
        'The answer to all questions is the key.',
        'The answer to all questions are the keys.'
    ]
    og_dataset = tf.data.Dataset.from_tensors(
        {
            'question': 'What is the answer?',
            'entity_pages': {
                'wiki_context': contexts
            },
            'answer': {
                'normalized_aliases': answers,
                'normalized_value': 'key'
            }
        }
    )

    dataset = prep.trivia_qa(og_dataset)
    assert_dataset(dataset, [{
        'inputs':
            'question: What is the answer ? context: The answer to all questions is the key . ',
        'targets':
            'key'
    }, {
        'inputs':
            'question: What is the answer ? context: The answer to all questions are the keys . ',
        'targets':
            'key'
    }, {
        'inputs':
            'question: What is the answer ? context: The answer to all questions are the keys . ',
        'targets':
            'keys'
    }])

  def test_squad_span_space_tokenized(self):
    answers = ['the answer', 'answer']
    d = tf.data.Dataset.from_tensors(
        {
            'id': 'a',
            'context': 'context with the answer.',
            'question': 'Say what?',
            'answers': {
                'text': answers,
            },
        },)
    og_dataset = d.concatenate(
        tf.data.Dataset.from_tensors(
            {  # Filter this out because answer is not in context.
                'id': 'b',
                'context': 'context without answers.',
                'question': 'Say what?',
                'answers': {
                    'text': answers,
                }
            }))

    dataset = prep.squad_span_space_tokenized(og_dataset)
    assert_dataset(
        dataset, {
            'id':
                'a',
            'inputs':
                'question: Say what ? context: context with the answer . ',
            'targets':
                'start: 2 end: 3',
            'context':
                'context with the answer . ',
            'question':
                'Say what ? ',
            'answers':
                answers,
        })

  def test_glue(self):
    test_idx = 10
    input_data = {
        'q1': 'How so?',
        'q2': 'Why not?',
        'q3': 'Who?',
        'idx': test_idx,
        'label': 0,
    }
    og_dataset = tf.data.Dataset.from_tensors(input_data)
    benchmark_name = 'qqp'
    label_names = ['not_duplicate', 'duplicate']

    dataset = prep.glue(og_dataset, benchmark_name, label_names)
    assert_dataset(
        dataset,
        {
            'inputs': 'qqp q1: How so? q2: Why not? q3: Who?',
            'targets': 'not_duplicate',
            'idx': test_idx,
        },
    )

    # Test `feature_names` argument.
    dataset = prep.glue(
        og_dataset, benchmark_name, label_names, feature_names=['q3', 'q1'])
    assert_dataset(
        dataset,
        {
            'inputs': 'qqp q3: Who? q1: How so?',
            'targets': 'not_duplicate',
            'idx': test_idx,
        },
    )

    # Test target is <unk> when label is -1
    input_data['label'] = -1
    og_dataset = tf.data.Dataset.from_tensors(input_data)
    dataset = prep.glue(og_dataset, benchmark_name, label_names)
    assert_dataset(
        dataset,
        {
            'inputs': 'qqp q1: How so? q2: Why not? q3: Who?',
            'targets': '<unk>',
            'idx': test_idx,
        },
    )

    # Test id_key argument
    input_data = {
        'q1': 'How so?',
        'q2': 'Why not?',
        'q3': 'Who?',
        'uid': test_idx,
        'label': 0,
    }
    og_dataset = tf.data.Dataset.from_tensors(input_data)
    dataset = prep.glue(og_dataset, benchmark_name, label_names,
                        feature_names=['q1', 'q2', 'q3'], id_key='uid')
    assert_dataset(
        dataset,
        {
            'inputs': 'qqp q1: How so? q2: Why not? q3: Who?',
            'targets': 'not_duplicate',
            'idx': test_idx,
        },
    )

  def test_multirc(self):
    og_dataset = tf.data.Dataset.from_tensors({
        'paragraph':
            '<b>Sent 1: </b>Once upon a time, there was a squirrel named Joey.<br><b>Sent 2: </b>Joey loved to go outside and play with his cousin Jimmy.',
        'question':
            'Why was Joey surprised the morning he woke up for breakfast?',
        'answer':
            'There was only pie to eat',
        'label': 1,
        'idx': {
            'paragraph': 5,
            'question': 1,
            'answer': 3
        }
    })

    dataset = prep.glue(
        og_dataset,
        'multirc',
        label_names=['False', 'True'],
        feature_names=('question', 'answer', 'paragraph'),
    )
    assert_dataset(
        dataset,
        {
            'inputs':
                'multirc question: Why was Joey surprised the morning he woke up for breakfast? answer: There was only pie to eat paragraph: Sent 1: Once upon a time, there was a squirrel named Joey. Sent 2: Joey loved to go outside and play with his cousin Jimmy.',
            'targets': 'True',
            'idx/paragraph': 5,
            'idx/question': 1,
            'idx/answer': 3,
        },
    )

  def test_stsb(self):
    test_idx = 10
    og_dataset = tf.data.Dataset.from_tensors(
        {
            'sentence1': ['Big news.'],
            'sentence2': ['No idea.'],
            'label': [2.8],
            'idx': test_idx,
        },
    )

    dataset = prep.stsb(og_dataset)
    assert_dataset(
        dataset,
        {
            'inputs': 'stsb sentence1: Big news. sentence2: No idea.',
            'targets': '2.8',
            'idx': test_idx,
        },
    )

    # Test when floating point label is not in [0., 0.2, ..., 4.8, 5.0]
    og_dataset = tf.data.Dataset.from_tensor_slices(
        {
            'sentence1': ['Big news.'],
            'sentence2': ['No idea.'],
            'label': [1.66],
            'idx': [test_idx],
        }
    )
    dataset = prep.stsb(og_dataset)
    assert_dataset(
        dataset,
        {
            'inputs': 'stsb sentence1: Big news. sentence2: No idea.',
            'targets': '1.6',
            'idx': test_idx,
        },
    )

  def test_multi_translate(self):
    languages = ['en', 'de', 'fr']
    translations = ['That is good.', 'Das ist gut.', 'Ca c\'est bon.']
    og_dataset = tf.data.Dataset.from_tensors(
        {'translations': {'language': languages, 'translation': translations}}
    )

    dataset = prep.multi_translate(og_dataset, 'en', 'de')
    assert_dataset(
        dataset,
        {
            'inputs': 'translate English to German: That is good.',
            'targets': 'Das ist gut.',
        }
    )

    # Test that it skips over the whole (single-entry) dataset when we ask for
    # a language which is not in the language list
    dataset = prep.multi_translate(og_dataset, 'en', 'sk')
    assert_dataset(dataset, [])

  def test_fill_in_the_blank(self):
    num_tries = 1000
    original = 'This is a long test with lots of words to see if it works ok.'
    dataset = tf.data.Dataset.from_tensor_slices(
        {'text': [original] * num_tries})
    dataset = prep.fill_in_the_blank(dataset)
    for data in test_utils.dataset_as_text(dataset):
      # Remove the prefix from the start of the input string
      self.assertTrue(data['inputs'].startswith('fill: '))
      inp = data['inputs'].replace('fill: ', '')
      # Split output into chunks according to X locations.
      out_split = data['targets'].split('X')
      # Make sure that there is at least one blank
      self.assertGreater(len(out_split), 1)
      # Remove leading/trailing whitespace and any empty chunks
      out_split = [o.strip() for o in out_split if o]
      # Replace 'X' with entries from out_split by popping from the front
      reconstructed = ''.join(
          [i if i != 'X' else out_split.pop(0) for i in inp])
      self.assertEqual(reconstructed, original)

  def test_fill_in_the_blank_sized(self):
    def _validate_data(data, valid_bins, og_length=15):
      # Remove the prefix from the start of the input string
      self.assertTrue(data['inputs'].startswith('fill: '))
      inp = data['inputs'].replace('fill: ', '')
      # Split input into chunks according to blank locations.
      inp_split = inp.split('_')
      # Make sure that there is exactly one blank (could be at beginning/end).
      self.assertLen(inp_split, 3)
      # Make sure reconstruction is accurate.
      reconstructed = ''.join([inp_split[0], data['targets']] + inp_split[2:])
      self.assertEqual(reconstructed, original)
      # Make sure blank size is correctly chosen.
      blank_bin = int(inp_split[1])
      self.assertIn(blank_bin, valid_bins)
      blank_size = len(data['targets'].split())
      self.assertGreaterEqual(blank_size, min(og_length, valid_bins[0]))
      self.assertLessEqual(blank_size, valid_bins[-1])
      return blank_size, blank_bin

    num_tries = 250
    original = 'This is a long test with lots of words to see if it works ok.'
    dataset = tf.data.Dataset.from_tensor_slices(
        {'text': [original] * num_tries})
    dataset = prep.fill_in_the_blank_sized(dataset, [1, 4])
    num_outputs = 0
    for data in test_utils.dataset_as_text(dataset):
      blank_size, blank_bin = _validate_data(data, [1, 4])
      if blank_size <= 2:
        self.assertEqual(blank_bin, 1)
      else:
        self.assertEqual(blank_bin, 4)
      num_outputs += 1
    self.assertEqual(num_tries, num_outputs)

    # Check case where bin size is larger than text.
    dataset = tf.data.Dataset.from_tensor_slices(
        {'text': [original] * num_tries})
    dataset = prep.fill_in_the_blank_sized(dataset, [1024])
    self.assertEmpty(list(test_utils.dataset_as_text(dataset)))

  def test_random_split_text(self):
    num_tries = 10
    original = '%s' % list(range(100))
    dataset = tf.data.Dataset.from_tensor_slices(
        {'text': [original] * num_tries})
    dataset = prep.random_split_text(dataset)
    out = []
    for data in test_utils.dataset_as_text(dataset):
      out.append(data['text'])
    reconstructed = ' '.join(out)
    ref = ' '.join([original] * num_tries)
    self.assertEqual(reconstructed, ref)

  def test_split_tokens(self):
    original = list(range(2, 102))
    og_dataset = tf.data.Dataset.from_tensors({'targets': original})

    # Verify splits with no max segments.
    def _verify_split(length, n_expected_outputs):
      ds = prep.split_tokens(
          og_dataset, unused_vocabulary=None, max_tokens_per_segment=length)
      outputs = list(test_utils.dataset_as_text(ds))
      self.assertLen(outputs, n_expected_outputs)
      reconstructed = []
      for ex in outputs[:-1]:
        t = ex['targets']
        self.assertLen(t, length)
        reconstructed.extend(t)
      final_t = outputs[-1]['targets']
      self.assertLessEqual(len(final_t), length)
      reconstructed.extend(final_t)
      self.assertEqual(reconstructed, original)
    _verify_split(25, 4)
    _verify_split(30, 4)
    _verify_split(100, 1)
    _verify_split(1000, 1)

  def test_split_tokens_rank2(self):
    original = list([i, 2 * i] for i in range(2, 102))
    og_dataset = tf.data.Dataset.from_tensors({'targets': original})

    # Verify splits with no max segments.
    def _verify_split(length, n_expected_outputs):
      ds = prep.split_tokens(
          og_dataset, unused_vocabulary=None, max_tokens_per_segment=length)
      outputs = list(test_utils.dataset_as_text(ds))
      self.assertLen(outputs, n_expected_outputs)
      reconstructed = []
      for ex in outputs[:-1]:
        t = ex['targets']
        self.assertLen(t, length)
        reconstructed.extend(t)
      final_t = outputs[-1]['targets']
      self.assertLessEqual(len(final_t), length)
      reconstructed.extend(final_t)
      self.assertAllEqual(reconstructed, original)
    _verify_split(25, 4)
    _verify_split(30, 4)
    _verify_split(100, 1)
    _verify_split(1000, 1)

  def test_split_padding_tokens(self):
    original = [0] * 100
    og_dataset = tf.data.Dataset.from_tensors({'targets': original})

    # Verify splits with no max segments.
    def _verify_split(length, n_expected_outputs):
      ds = prep.split_tokens(
          og_dataset, unused_vocabulary=None, max_tokens_per_segment=length)
      outputs = list(test_utils.dataset_as_text(ds))
      self.assertLen(outputs, n_expected_outputs)
      reconstructed = []
      for ex in outputs[:-1]:
        t = ex['targets']
        self.assertLen(t, length)
        reconstructed.extend(t)
      final_t = outputs[-1]['targets']
      self.assertLessEqual(len(final_t), length)
      reconstructed.extend(final_t)
      self.assertEqual(reconstructed, original)
    _verify_split(25, 4)
    _verify_split(30, 4)
    _verify_split(100, 1)
    _verify_split(1000, 1)

  def test_split_tokens_additional_features_passthrough(self):
    original = list(range(2, 102))
    original_aux = list(range(4, 104))
    original_aux2d = list([i, 2 * i] for i in range(6, 106))
    original_passthrough = list(range(20))
    og_dataset = tf.data.Dataset.from_tensors({
        'targets': original,
        'aux': original_aux,
        'aux2d': original_aux2d,
        'passthrough': original_passthrough
    })
    # Verify splits with no max segments.
    def _verify_split(length, n_expected_outputs):
      ds = prep.split_tokens(
          og_dataset, unused_vocabulary=None, max_tokens_per_segment=length,
          additional_feature_keys=['aux', 'aux2d'],
          passthrough_feature_keys=['passthrough'])
      outputs = list(test_utils.dataset_as_text(ds))
      self.assertLen(outputs, n_expected_outputs)
      reconstructed = []
      reconstructed_aux = []
      reconstructed_aux2d = []
      for ex in outputs[:-1]:
        t = ex['targets']
        self.assertLen(t, length)
        reconstructed.extend(t)

        a = ex['aux']
        self.assertLen(a, length)
        reconstructed_aux.extend(a)

        a2d = ex['aux2d']
        self.assertLen(a2d, length)
        reconstructed_aux2d.extend(a2d)
      final_t = outputs[-1]['targets']
      self.assertLessEqual(len(final_t), length)
      reconstructed.extend(final_t)
      self.assertEqual(reconstructed, original)

      final_a = outputs[-1]['aux']
      self.assertLessEqual(len(final_a), length)
      reconstructed_aux.extend(final_a)
      self.assertEqual(reconstructed_aux, original_aux)

      final_a2d = outputs[-1]['aux2d']
      self.assertLessEqual(len(final_a2d), length)
      reconstructed_aux2d.extend(final_a2d)
      self.assertAllEqual(reconstructed_aux2d, original_aux2d)

      for ex in outputs:
        self.assertAllEqual(original_passthrough, ex['passthrough'])
    _verify_split(25, 4)
    _verify_split(30, 4)
    _verify_split(100, 1)
    _verify_split(1000, 1)

  def test_split_tokens_additional_features_passthrough_rank0(self):
    original = list(range(2, 102))
    original_aux = list(range(4, 104))
    original_passthrough = 1234
    og_dataset = tf.data.Dataset.from_tensors({
        'targets': original,
        'aux': original_aux,
        'passthrough': original_passthrough
    })
    # Verify splits with no max segments.
    def _verify_split(length, n_expected_outputs):
      ds = prep.split_tokens(
          og_dataset, unused_vocabulary=None, max_tokens_per_segment=length,
          additional_feature_keys=['aux'],
          passthrough_feature_keys=['passthrough'])
      outputs = list(test_utils.dataset_as_text(ds))
      self.assertLen(outputs, n_expected_outputs)
      reconstructed = []
      reconstructed_aux = []
      for ex in outputs[:-1]:
        t = ex['targets']
        self.assertLen(t, length)
        reconstructed.extend(t)

        a = ex['aux']
        self.assertLen(a, length)
        reconstructed_aux.extend(a)
      final_t = outputs[-1]['targets']
      self.assertLessEqual(len(final_t), length)
      reconstructed.extend(final_t)
      self.assertEqual(reconstructed, original)

      final_a = outputs[-1]['aux']
      self.assertLessEqual(len(final_a), length)
      reconstructed_aux.extend(final_a)
      self.assertEqual(reconstructed_aux, original_aux)

      for ex in outputs:
        self.assertAllEqual(original_passthrough, ex['passthrough'])
    _verify_split(25, 4)
    _verify_split(30, 4)
    _verify_split(100, 1)
    _verify_split(1000, 1)

  def test_split_tokens_to_targets_length(self):
    original = list(range(2, 102))
    og_dataset = tf.data.Dataset.from_tensors({'targets': original})
    sequence_length = {'targets': 4}
    eos_features = {
        'targets': seqio.Feature(
            vocabulary=seqio.PassThroughVocabulary(102),
            add_eos=True)
    }
    no_eos_features = {
        'targets': seqio.Feature(
            vocabulary=seqio.PassThroughVocabulary(102),
            add_eos=False)
    }

    ds = prep.split_tokens_to_targets_length(
        og_dataset,
        sequence_length=sequence_length,
        output_features=eos_features)
    eos_outputs = list(ds.as_numpy_iterator())
    # Outputs should be length 3 to leave room for adding EOS.
    self.assertLen(eos_outputs[0]['targets'], 3)

    ds = prep.split_tokens_to_targets_length(
        og_dataset,
        sequence_length=sequence_length,
        output_features=no_eos_features)
    no_eos_outputs = list(ds.as_numpy_iterator())
    self.assertLen(no_eos_outputs[0]['targets'], 4)

  def test_trim_tokens_at_front(self):
    sequence_length = {'inputs': 4}
    inputs = tf.data.Dataset.from_tensors(
        {'inputs': tf.constant([10, 11, 12, 13, 14, 15])})
    output = prep.trim_tokens_at_front(inputs, sequence_length=sequence_length)

    expected_output = [{'inputs': tf.constant([13, 14, 15])}]
    test_utils.assert_dataset(output, expected_output)

  def test_split_text_to_words(self):
    og_dataset = tf.data.Dataset.from_tensor_slices({
        'text': [
            'Here\'s a sentence. Here is another; it has a semicolon.',
            'I\'m 2-words.',
            'There_are_no_spaces_here_so_technically_this_is_one_word.',
            '',
        ]
    })
    dataset = prep.split_text_to_words(og_dataset)
    test_utils.assert_dataset(dataset, [
        {
            'words': ['Here\'s', 'a', 'sentence.', 'Here', 'is', 'another;',
                      'it', 'has', 'a', 'semicolon.',],
            'text': 'Here\'s a sentence. Here is another; it has a semicolon.'
        },
        {
            'words': ['I\'m', '2-words.'],
            'text': 'I\'m 2-words.'
        },
    ])

  def test_definite_pronoun_resolution_simple(self):
    # Test where the pronoun is in the middle of the sentence. Also test the
    # case where the string pronoun is a substring of another word in the
    # sentence.
    og_dataset = tf.data.Dataset.from_tensors({
        'sentence': 'Mitchell asked Tom if he could lend some money.',
        'pronoun': 'he',
        'candidates': ['Mitchell', 'Tom'],
        'label': 1,
    })
    dataset = prep.definite_pronoun_resolution_simple(og_dataset)
    assert_dataset(
        dataset, {
            'inputs':
                'wsc: Mitchell asked Tom if *he* could lend some money.',
            'targets':
                'Tom',
        })

    # Test multiple word pronouns. The Definite Pronoun Resolution Dataset is
    # weird.
    og_dataset = tf.data.Dataset.from_tensors({
        'sentence':
            'Bill beat Tom at Scrabble because that newbie had all the luck.',
        'pronoun':
            'that newbie',
        'candidates': ['Bill', 'Tom'],
        'label':
            0,
    })
    dataset = prep.definite_pronoun_resolution_simple(og_dataset)
    assert_dataset(
        dataset, {
            'inputs':
                'wsc: Bill beat Tom at Scrabble because *that newbie* had all the luck.',
            'targets':
                'Bill',
        })

    # Test pronoun at end of sentence.
    og_dataset = tf.data.Dataset.from_tensors({
        'sentence':
            'Carl borrowed a book from Richard, but the book was unreadable to him.',
        'pronoun':
            'him',
        'candidates': ['Carl', 'Richard'],
        'label':
            0,
    })
    dataset = prep.definite_pronoun_resolution_simple(og_dataset)
    assert_dataset(
        dataset, {
            'inputs':
                'wsc: Carl borrowed a book from Richard, but the book was unreadable to *him*.',
            'targets':
                'Carl',
        })

  def test_wsc_simple(self):
    og_dataset = tf.data.Dataset.from_tensors({
        'text': 'Mitchell asked Tom if he could lend some money.',
        'span1_text': 'Tom',
        'span2_text': 'he',
        'span2_index': 4,
        'idx': 1,
    })

    dataset = prep.wsc_simple(og_dataset, correct_referent_only=False)
    assert_dataset(
        dataset, {
            'inputs': 'wsc: Mitchell asked Tom if *he* could lend some money.',
            'targets': 'Tom',
            'label': 0,
            'idx': 1,
        })

    # Test including only examples with the correct referent.
    og_dataset = tf.data.Dataset.from_tensor_slices({
        'text': [
            'Mitchell asked Tom if he could lend some money.',
            'Mitchell asked Tom if he could lend some money.',
        ],
        'span1_text': [
            'Tom',
            'Mitchell',
        ],
        'span2_text': [
            'he',
            'he',
        ],
        'span2_index': [4, 4],
        'label': [1, 0],
        'idx': [1, 2]
    })
    dataset = prep.wsc_simple(og_dataset, correct_referent_only=True)
    assert_dataset(dataset, [{
        'inputs': 'wsc: Mitchell asked Tom if *he* could lend some money.',
        'targets': 'Tom',
        'label': True,
        'idx': 1,
    }])

  def test_wnli_simple(self):
    og_dataset = tf.data.Dataset.from_tensor_slices({
        'sentence1': [
            'Lily spoke to Donna breaking her silence.',
            'The fish ate the worm. It was tasty.',
            'Edward dropped adhesive tape onto his window sill, and when he pulled the tape off, some of the glue was stuck on it.',
            "Al stole Bob's wallet and car, and then he was driving it really fast to get away.",
        ],
        'sentence2': [
            "Lily spoke to Donna breaking Donna's silence.",
            'The worm was tasty.',
            'Some of the glue was stuck on the window sill.',
            'He was driving the car really fast to get away.',
        ],
        'idx': [1, 2, 3, 4],
        'label': [1, 0, 0, 1],
    })
    dataset = prep.wnli_simple(og_dataset)
    assert_dataset(dataset, [
        {
            'inputs': 'wsc: Lily spoke to Donna breaking *her* silence.',
            'targets': 'Donna',
            'premise': 'Lily spoke to Donna breaking her silence.',
            'hypothesis': "Lily spoke to Donna breaking Donna's silence.",
            'label': 1,
            'idx': 1,
        },
        {
            'inputs': 'wsc: The fish ate the worm. *It* was tasty.',
            'targets': 'The worm',
            'premise': 'The fish ate the worm. It was tasty.',
            'hypothesis': 'The worm was tasty.',
            'label': 0,
            'idx': 2,
        },
        {
            'inputs':
                'wsc: Edward dropped adhesive tape onto his window sill, and when he pulled the tape off, some of the glue was stuck on *it* .',
            'targets':
                'the window sill',
            'premise':
                'Edward dropped adhesive tape onto his window sill, and when he pulled the tape off, some of the glue was stuck on it.',
            'hypothesis':
                'Some of the glue was stuck on the window sill.',
            'label':
                0,
            'idx':
                3,
        },
        {
            'inputs':
                "wsc: Al stole Bob's wallet and car, and then he was driving *it* really fast to get away.",
            'targets':
                'the car',
            'premise':
                "Al stole Bob's wallet and car, and then he was driving it really fast to get away.",
            'hypothesis':
                'He was driving the car really fast to get away.',
            'label':
                1,
            'idx':
                4,
        },
    ])

  def test_next_sentence_prediction(self):

    og_dataset = tf.data.Dataset.from_tensor_slices({
        'text': [
            'This is the first sentence. This is the second sentence.',
            'This is the third sentence. This is the fourth sentence.',
        ]
    })

    # Test neighboring sentences.
    dataset = prep.next_sentence_prediction(
        og_dataset, label_sentences=False, p_neighbors=1, buffer_size=1)
    assert_dataset(
        dataset,
        [
            {
                'inputs':
                    'nsp: This is the first sentence. This is the second sentence.',
                'targets':
                    'next',
            },
            {
                'inputs':
                    'nsp: This is the third sentence. This is the fourth sentence.',
                'targets':
                    'next',
            },
        ],
    )

    # Test non-neighboring sentences.
    dataset = prep.next_sentence_prediction(
        og_dataset, label_sentences=False, p_neighbors=0, buffer_size=1)
    assert_dataset(
        dataset,
        [
            {
                'inputs':
                    'nsp: This is the first sentence. This is the fourth sentence.',
                'targets':
                    'not_next',
            },
            {
                'inputs':
                    'nsp: This is the third sentence. This is the second sentence.',
                'targets':
                    'not_next',
            },
        ],
    )

    # Test labeling sentences.
    dataset = prep.next_sentence_prediction(
        og_dataset, label_sentences=True, p_neighbors=1, buffer_size=1)
    assert_dataset(
        dataset,
        [
            {
                'inputs':
                    'nsp: sentence1: This is the first sentence. sentence2: This is the second sentence.',
                'targets':
                    'next',
            },
            {
                'inputs':
                    'nsp: sentence1: This is the third sentence. sentence2: This is the fourth sentence.',
                'targets':
                    'next',
            },
        ],
    )

  def test_lm(self):
    dataset = tf.data.Dataset.from_tensor_slices({'text': ['That is good.']})
    dataset = prep.lm(dataset)
    assert_dataset(dataset, {'inputs': '', 'targets': 'That is good.'})

  def test_triviaqa_truncate_text(self):

    vocab = test_utils.sentencepiece_vocab()

    def tokenize_and_prepare_dataset(inputs, targets):
      tokenized_inputs = vocab.encode(inputs)
      tokenized_targets = vocab.encode(targets)

      dataset = tf.data.Dataset.from_tensors({
          'inputs': tokenized_inputs,
          'targets': tokenized_targets,
      })

      return dataset, tokenized_targets

    inputs = 'This is a very very long string which must contain the answer.'
    targets = 'long string'

    og_dataset, tokenized_targets = tokenize_and_prepare_dataset(
        inputs, targets)

    for _ in range(0, 10):
      dataset = prep.trivia_qa_truncate_inputs(
          og_dataset, output_features=None, sequence_length={'inputs': 20})

      for data in test_utils.dataset_as_text(dataset):
        self.assertLen(data['inputs'], 20)
        self.assertContainsSubset(tokenized_targets, data['inputs'])

    # Dummy input which exists in the vocab to be able to compare strings after
    # decoding.
    inputs = 'w h d n r t v'
    targets = 'h d'

    og_dataset, _ = tokenize_and_prepare_dataset(inputs, targets)

    for _ in range(0, 5):
      dataset = prep.trivia_qa_truncate_inputs(
          og_dataset, output_features=None, sequence_length={'inputs': 5})

      for data in test_utils.dataset_as_text(dataset):
        self.assertLen(data['inputs'], 5)
        truncated_inputs = vocab.decode(data['inputs'].tolist())
        new_targets = vocab.decode(data['targets'].tolist())
        self.assertRegex(truncated_inputs, '.*' + targets + '.*')
        self.assertEqual(targets, new_targets)

  def test_triviaqa_truncate(self):

    sequence_length = {
        'inputs': 10,
    }

    # Answer starts from the 0th position of the inputs.
    dataset = tf.data.Dataset.from_tensors({
        'inputs': tf.range(0, 30),
        'targets': tf.range(0, 5)
    })

    dataset = prep.trivia_qa_truncate_inputs(
        dataset, output_features=None, sequence_length=sequence_length)

    assert_dataset(dataset, {
        'inputs': tf.range(0, 10),
        'targets': tf.range(0, 5)
    })

    # Answer is in the last n elements of the targets.
    dataset = tf.data.Dataset.from_tensors({
        'inputs': tf.range(0, 30),
        'targets': tf.range(27, 30)
    })

    dataset = prep.trivia_qa_truncate_inputs(
        dataset, output_features=None, sequence_length=sequence_length)

    assert_dataset(dataset, {
        'inputs': tf.range(20, 30),
        'targets': tf.range(27, 30)
    })

    # Answer is not in inputs. Example is droped from the dataset.
    no_overlap_dataset = tf.data.Dataset.from_tensors({
        'inputs': tf.range(0, 30),
        'targets': tf.range(27, 32)
    })

    dataset = prep.trivia_qa_truncate_inputs(
        no_overlap_dataset,
        output_features=None,
        sequence_length=sequence_length
    )

    i = 0
    for data in test_utils.dataset_as_text(dataset):
      i = i + 1

    self.assertEqual(i, 0)

    # Answer is in the middle of the inputs.
    for _ in range(0, 10):
      og_dataset = tf.data.Dataset.from_tensors({
          'inputs': tf.range(0, 30),
          'targets': tf.range(10, 15),
      })

      dataset = prep.trivia_qa_truncate_inputs(
          og_dataset, output_features=None,
          sequence_length=sequence_length
      )
      for data in test_utils.dataset_as_text(dataset):
        self.assertContainsSubset(data['targets'], data['inputs'])
        self.assertLen(data['inputs'], 10)

  def test_record(self):
    og_dataset = tf.data.Dataset.from_tensors({
        'query': 'It was @placeholder.',
        'entities': ['A', 'B', 'C'],
        'passage': [
            'This is the passage\n@highlight\nAnother sentence.\n@highlight\nThird sentence.'
        ],
        'answers': ['A', 'C'],
        'idx': {
            'passage': 1,
            'query': 2,
        },
    })

    dataset = prep.record(og_dataset)
    assert_dataset(dataset, [
        {
            'inputs':
                'record query: It was @placeholder. entities: A, B, C passage: This is the passage. Another sentence. Third sentence.',
            'targets': 'A',
            'idx/passage': 1,
            'idx/query': 2,
            'answers': ['A', 'C'],
        },
        {
            'inputs':
                'record query: It was @placeholder. entities: A, B, C passage: This is the passage. Another sentence. Third sentence.',
            'targets': 'C',
            'idx/passage': 1,
            'idx/query': 2,
            'answers': ['A', 'C'],
        },
    ])

    # Test a dataset without answers, as would appear in the test set
    og_test_dataset = tf.data.Dataset.from_tensors({
        'query': 'It was @placeholder.',
        'entities': ['A', 'B', 'C'],
        'passage': [
            'This is the passage\n@highlight\nAnother sentence.\n@highlight\nThird sentence.'
        ],
        'answers': tf.constant([], dtype=tf.string),
        'idx': {
            'passage': 1,
            'query': 2,
        },
    })

    # Test all answers.
    dataset = prep.record(og_test_dataset)
    assert_dataset(dataset, [
        {
            'inputs':
                'record query: It was @placeholder. entities: A, B, C passage: This is the passage. Another sentence. Third sentence.',
            'targets': '<unk>',
            'idx/passage': 1,
            'idx/query': 2,
            'answers': []
        },
    ])

  def test_parse_tsv(self):
    og_dataset = tf.data.Dataset.from_tensor_slices(['a\tb', 'c\td'])
    dataset = prep.parse_tsv(og_dataset, field_names=['f1', 'f2'])
    assert_dataset(dataset, [{'f1': 'a', 'f2': 'b'}, {'f1': 'c', 'f2': 'd'}])

  def test_parse_tsv_select_cols(self):
    og_dataset = tf.data.Dataset.from_tensor_slices(
        ['0\ta\tb\te', '1\tc\td\tf'])
    dataset = prep.parse_tsv(
        og_dataset, field_names=['f1', 'f2', 'f3'], field_columns=[1, 2, 3])
    assert_dataset(dataset, [{
        'f1': 'a',
        'f2': 'b',
        'f3': 'e'
    }, {
        'f1': 'c',
        'f2': 'd',
        'f3': 'f'
    }])

  def test_denoise_no_corruption(self):
    vocab = test_utils.sentencepiece_vocab()
    target_tokens = vocab.encode('The quick brown fox.')

    # This is what it encodes to.
    self.assertEqual(
        target_tokens,
        [3, 2, 20, 4, 3, 2, 8, 13, 2, 3, 2, 23, 7, 19, 22, 3, 2, 7, 2])

    og_dataset = tf.data.Dataset.from_tensor_slices({
        'targets': [target_tokens],
    })

    output_features = {
        'targets': seqio.Feature(vocab),
    }

    # Using noise density 0.0 to avoid corruption
    with seqio.map_seed_manager(42):
      denoised_dataset = prep.denoise(
          og_dataset,
          output_features,
          noise_density=0.0,
          noise_mask_fn=prep.random_spans_noise_mask,
          inputs_fn=prep.noise_span_to_unique_sentinel,
          targets_fn=prep.nonnoise_span_to_unique_sentinel,
          input_feature_key='text_tokens')

    # Nothing gets corrupted
    assert_dataset(denoised_dataset, [
        {
            'text_tokens':
                [3, 2, 20, 4, 3, 2, 8, 13, 2, 3, 2, 23, 7, 19, 22, 3, 2, 7, 2],
            'targets': [25],
        },
    ])

  def test_denoise(self):
    vocab = test_utils.sentencepiece_vocab()
    target_tokens = vocab.encode('The quick brown fox.')

    # This is what it encodes to.
    self.assertEqual(
        target_tokens,
        [3, 2, 20, 4, 3, 2, 8, 13, 2, 3, 2, 23, 7, 19, 22, 3, 2, 7, 2])

    og_dataset = tf.data.Dataset.from_tensor_slices({
        'targets': [target_tokens],
    })

    output_features = {
        'targets': seqio.Feature(vocab),
    }

    # These are the parameters of denoise in the operative config of 'base'.
    # Except noise_density, bumped up from 0.15 to 0.3 in order to demonstrate
    # multiple corrupted spans.
    with seqio.map_seed_manager(42):
      denoised_dataset = prep.denoise(
          og_dataset,
          output_features,
          noise_density=0.3,
          noise_mask_fn=prep.random_spans_noise_mask,
          inputs_fn=prep.noise_span_to_unique_sentinel,
          targets_fn=prep.nonnoise_span_to_unique_sentinel,
          input_feature_key='text_tokens')

    # The two spans corrupted, [2] and [22, 3, 2, 7, 2], are replaced by unique
    # sentinels 25 and 24 respectively.
    assert_dataset(
        denoised_dataset,
        [
            {
                'text_tokens': [
                    3,
                    2,
                    20,
                    4,
                    3,
                    2,
                    8,
                    13,  # unchanged
                    25,  # replace [2]
                    3,
                    2,
                    23,
                    7,
                    19,  # unchanged
                    24,  # replaced [22, 3, 2, 7, 2]
                ],
                'targets': [25, 2, 24, 22, 3, 2, 7, 2],
            },
        ])

  def test_denoise_nested_decorators(self):
    """Test whether gin and utils.map_over_dataset decorators are compatible."""
    bindings = """
      preprocessors.unsupervised.preprocessors = [@preprocessors.denoise]
      preprocessors.denoise.noise_density = 0.15
      preprocessors.denoise.noise_mask_fn = @preprocessors.iid_noise_mask
      preprocessors.denoise.inputs_fn = @noise_token_to_sentinel
    """
    gin.parse_config(bindings)
    og_dataset = tf.data.Dataset.from_tensor_slices({'targets': [1, 2, 3]})
    output_features = {
        'targets': seqio.Feature(test_utils.sentencepiece_vocab())
    }
    # Test denoise function when it is used as a gin-configurable of another
    # gin-configurable, prep.unsupervised.
    dataset = prep.unsupervised(og_dataset, output_features=output_features)
    self.assertIsInstance(dataset, tf.data.Dataset)

  def test_prefix_lm(self):
    vocab = test_utils.sentencepiece_vocab()
    # Create list of length 99 because prefix_lm will split to 1 less than the
    # max length of 100 to leave room for EOS token.
    inp = list(range(1, 100))
    og_dataset = tf.data.Dataset.from_tensor_slices({'targets': [inp]})
    og_dataset = og_dataset.repeat(100)
    output_features = {
        'targets': seqio.Feature(vocab),
        'inputs': seqio.Feature(vocab),
    }
    output_dataset = prep.prefix_lm(
        og_dataset,
        {'inputs': 100, 'targets': 100},
        output_features,
    )
    input_lengths = set()
    for ex in output_dataset.as_numpy_iterator():
      self.assertListEqual(
          ex['inputs'].tolist() + ex['targets'].tolist(), inp
      )
      input_lengths.add(len(ex['inputs']))
    self.assertGreater(len(input_lengths), 1)

  def test_rank_classification(self):
    dataset = tf.data.Dataset.from_tensors({
        'left': 'the sky is blue',
        'right': 'cats are so cute',
        'label_idx': 1,
    })
    preprocessor = functools.partial(
        prep.rank_classification,
        dataset,
        inputs_fn=lambda features: [features['right'], features['left']],
        targets_fn=lambda features: ['class 0', 'class 1'],
        is_correct_fn=lambda features: tf.one_hot(features['label_idx'], 2))

    test_utils.assert_dataset(
        preprocessor(mode='train'),
        [
            {
                'idx': [0, 1],
                'inputs': 'the sky is blue',
                'targets': 'class 1',
                'is_correct': True,
            }
        ])

    test_utils.assert_dataset(
        preprocessor(mode='eval'),
        [
            {
                'idx': [0, 0],
                'inputs': 'cats are so cute',
                'targets': 'class 0',
                'is_correct': False,
            },
            {
                'idx': [0, 1],
                'inputs': 'the sky is blue',
                'targets': 'class 1',
                'is_correct': True,
            }
        ])

    test_utils.assert_dataset(
        preprocessor(mode='fewshot_eval'),
        [
            {
                'idx': [[0, 0], [0, 1]],
                'inputs': ['cats are so cute', 'the sky is blue'],
                'targets': ['class 0', 'class 1'],
                'is_correct': [False, True]
            },
        ])

  def test_rank_classification_multilabel(self):
    dataset = tf.data.Dataset.from_tensors({
        'left': 'the sky is blue',
        'right': 'cats are so cute',
    })

    preprocessor = functools.partial(
        prep.rank_classification,
        dataset,
        inputs_fn=lambda features: [features['right'], features['left'], 'X'],
        targets_fn=lambda features: ['class 0', 'class 1', 'class 2'],
        is_correct_fn=lambda features: [False, True, True])

    test_utils.assert_dataset(
        preprocessor(mode='train'),
        [
            {
                'idx': [0, 1],
                'inputs': 'the sky is blue',
                'targets': 'class 1',
                'is_correct': True,
            },
            {
                'idx': [0, 2],
                'inputs': 'X',
                'targets': 'class 2',
                'is_correct': True,
            },
        ])

    test_utils.assert_dataset(
        preprocessor(mode='eval'),
        [
            {
                'idx': [0, 0],
                'inputs': 'cats are so cute',
                'targets': 'class 0',
                'is_correct': False,
            },
            {
                'idx': [0, 1],
                'inputs': 'the sky is blue',
                'targets': 'class 1',
                'is_correct': True,
            },
            {
                'idx': [0, 2],
                'inputs': 'X',
                'targets': 'class 2',
                'is_correct': True,
            },
        ])

    test_utils.assert_dataset(
        preprocessor(mode='fewshot_eval'),
        [
            {
                'idx': [[0, 0], [0, 1], [0, 2]],
                'inputs': ['cats are so cute', 'the sky is blue', 'X'],
                'targets': ['class 0', 'class 1', 'class 2'],
                'is_correct': [False, True, True]
            },
        ])

  def test_rank_classification_with_weight(self):
    dataset = tf.data.Dataset.from_tensors({
        'left': 'the sky is blue',
        'right': 'cats are so cute',
        'label_idx': 1,
        'weight': 1.0,
    })
    preprocessor = functools.partial(
        prep.rank_classification,
        dataset,
        inputs_fn=lambda features: [features['right'], features['left']],
        targets_fn=lambda features: ['class 0', 'class 1'],
        is_correct_fn=lambda features: [False, True],
        weight_fn=lambda features: features['weight'])

    test_utils.assert_dataset(
        preprocessor(mode='train'), [{
            'idx': [0, 1],
            'inputs': 'the sky is blue',
            'targets': 'class 1',
            'is_correct': True,
            'weight': 1.0,
        }])

    test_utils.assert_dataset(
        preprocessor(mode='eval'), [{
            'idx': [0, 0],
            'inputs': 'cats are so cute',
            'targets': 'class 0',
            'is_correct': False,
            'weight': 1.0,
        }, {
            'idx': [0, 1],
            'inputs': 'the sky is blue',
            'targets': 'class 1',
            'is_correct': True,
            'weight': 1.0,
        }])

    test_utils.assert_dataset(
        preprocessor(mode='fewshot_eval'), [
            {
                'idx': [[0, 0], [0, 1]],
                'inputs': ['cats are so cute', 'the sky is blue'],
                'targets': ['class 0', 'class 1'],
                'is_correct': [False, True],
                'weight': [1, 1],
            },
        ])

  def test_rank_classification_with_passthrough_feature_keys(self):
    def dataset_generator():
      yield {
          'left': 'the sky is blue',
          'right': 'cats are so cute',
          'label_idx': 1,
          'weight': 1.0,
          'options': ['class 0', 'class 1'],
          'starburst_allow_pass': [0.1, 0.2],
          'context_allow_pass': 'the sun is out',
          'multicontext_allow_pass': ['the sun is out', 'so i am out'],
          'starburst_not_allow_pass': [0.9, 0.8]
      }

    dataset = tf.data.Dataset.from_generator(
        dataset_generator,
        output_signature={
            'left': tf.TensorSpec(shape=(), dtype=tf.string),
            'right': tf.TensorSpec(shape=(), dtype=tf.string),
            'label_idx': tf.TensorSpec(shape=(), dtype=tf.int32),
            'weight': tf.TensorSpec(shape=(), dtype=tf.float32),
            'options': tf.TensorSpec(shape=(None,), dtype=tf.string),
            'starburst_allow_pass': tf.TensorSpec(shape=(2,),
                                                  dtype=tf.float32),
            'context_allow_pass': tf.TensorSpec(shape=(), dtype=tf.string),
            'multicontext_allow_pass': tf.TensorSpec(shape=(None,),
                                                     dtype=tf.string),
            'starburst_not_allow_pass': tf.TensorSpec(shape=(2,),
                                                      dtype=tf.float32)
        })

    preprocessor = functools.partial(
        prep.rank_classification,
        dataset,
        inputs_fn=lambda features: [features['right'], features['left']],
        targets_fn=lambda features: features['options'],
        is_correct_fn=lambda features: [False, True],
        weight_fn=lambda features: features['weight'],
        passthrough_feature_keys=['starburst_allow_pass',
                                  'context_allow_pass',
                                  'multicontext_allow_pass'])

    test_utils.assert_dataset(
        preprocessor(mode='train'), [{
            'idx': [0, 1],
            'inputs': 'the sky is blue',
            'targets': 'class 1',
            'is_correct': True,
            'weight': 1.0,
            'starburst_allow_pass': [0.1, 0.2],
            'context_allow_pass': 'the sun is out',
            'multicontext_allow_pass': ['the sun is out', 'so i am out']
        }])

    test_utils.assert_dataset(
        preprocessor(mode='eval'), [{
            'idx': [0, 0],
            'inputs': 'cats are so cute',
            'targets': 'class 0',
            'is_correct': False,
            'weight': 1.0,
            'starburst_allow_pass': [0.1, 0.2],
            'context_allow_pass': 'the sun is out',
            'multicontext_allow_pass': ['the sun is out', 'so i am out']
        }, {
            'idx': [0, 1],
            'inputs': 'the sky is blue',
            'targets': 'class 1',
            'is_correct': True,
            'weight': 1.0,
            'starburst_allow_pass': [0.1, 0.2],
            'context_allow_pass': 'the sun is out',
            'multicontext_allow_pass': ['the sun is out', 'so i am out']
        }])

    test_utils.assert_dataset(
        preprocessor(mode='fewshot_eval'), [
            {
                'idx': [[0, 0], [0, 1]],
                'inputs': ['cats are so cute', 'the sky is blue'],
                'targets': ['class 0', 'class 1'],
                'is_correct': [False, True],
                'weight': [1, 1],
                'starburst_allow_pass': [[0.1, 0.2], [0.1, 0.2]],
                'context_allow_pass': ['the sun is out', 'the sun is out'],
                'multicontext_allow_pass': [
                    ['the sun is out', 'so i am out'],
                    ['the sun is out', 'so i am out']]
            },
        ])

  def test_rank_classification_errors(self):
    dataset = tf.data.Dataset.from_tensors({
        'left': 'the sky is blue',
        'right': 'cats are so cute',
    })

    with self.assertRaisesRegex(
        tf.errors.InvalidArgumentError,
        '.*`inputs_fn`, `targets_fn`, and `is_correct_fn` must return the same '
        'size tensors.*'):
      list(prep.rank_classification(
          dataset,
          inputs_fn=lambda features: tf.stack([features['right']]),
          targets_fn=lambda features: tf.stack(['class 0', 'class 1']),
          is_correct_fn=lambda features: [False, True, True]))

    with self.assertRaisesRegex(
        tf.errors.InvalidArgumentError,
        '.*`inputs_fn`, `targets_fn`, and `is_correct_fn` must return the same '
        'size tensors.*'):
      list(prep.rank_classification(
          dataset,
          inputs_fn=
          lambda features: tf.stack([features['right'], features['left']]),
          targets_fn=lambda features: tf.stack(['class 0', 'class 1']),
          is_correct_fn=lambda features: [False, True, True]))

  def test_rank_classification_formatter(self):
    input_examples = [
        {
            'premise': 'The farmland needed irrigation.',
            'question': 'effect',
            'choice1': 'a canal was constructed',
            'choice2': 'the crops grew tall',
            'label': 0,
        },
        {
            'premise': 'I decided to stay home last night.',
            'question': 'cause',
            'choice1': 'I wanted to see people',
            'choice2': 'I was too tired',
            'label': 1,
        },
    ]

    input_ds = tf.data.Dataset.from_generator(
        lambda: (x for x in input_examples),
        output_types={
            'premise': tf.string,
            'question': tf.string,
            'choice1': tf.string,
            'choice2': tf.string,
            'label': tf.int32,
        },
        output_shapes={
            'premise': [],
            'question': [],
            'choice1': [],
            'choice2': [],
            'label': [],
        })

    # all options
    dataset = prep.rank_classification_formatter(
        input_ds,
        inputs_formats='{premise} What is the {question}? X',
        targets_formats=['I think {choice1}.', 'I think {choice2}.'],
        mode='eval')

    test_utils.assert_dataset(
        dataset,
        [
            {
                'idx': [0, 0],
                'inputs':
                    'The farmland needed irrigation. What is the effect? X',
                'targets': 'I think a canal was constructed.',
                'is_correct': True
            },
            {
                'idx': [0, 1],
                'inputs':
                    'The farmland needed irrigation. What is the effect? X',
                'targets': 'I think the crops grew tall.',
                'is_correct': False
            },
            {
                'idx': [1, 0],
                'inputs':
                    'I decided to stay home last night. What is the cause? X',
                'targets': 'I think I wanted to see people.',
                'is_correct': False
            },
            {
                'idx': [1, 1],
                'inputs':
                    'I decided to stay home last night. What is the cause? X',
                'targets': 'I think I was too tired.',
                'is_correct': True
            },
        ])

    # Reverse inputs and targets for supporting the use case when there is
    # one target, but multiple inputs to select from.
    dataset = prep.rank_classification_formatter(
        input_ds,
        inputs_formats=['I think {choice1}.', 'I think {choice2}.'],
        targets_formats='{premise} What is the {question}? X',
        mode='eval')

    test_utils.assert_dataset(
        dataset,
        [
            {
                'idx': [0, 0],
                'targets':
                    'The farmland needed irrigation. What is the effect? X',
                'inputs': 'I think a canal was constructed.',
                'is_correct': True
            },
            {
                'idx': [0, 1],
                'targets':
                    'The farmland needed irrigation. What is the effect? X',
                'inputs': 'I think the crops grew tall.',
                'is_correct': False
            },
            {
                'idx': [1, 0],
                'targets':
                    'I decided to stay home last night. What is the cause? X',
                'inputs': 'I think I wanted to see people.',
                'is_correct': False
            },
            {
                'idx': [1, 1],
                'targets':
                    'I decided to stay home last night. What is the cause? X',
                'inputs': 'I think I was too tired.',
                'is_correct': True
            },
        ])

    # train mode
    dataset = prep.rank_classification_formatter(
        input_ds,
        inputs_formats='{premise} What is the {question}? X',
        targets_formats=['I think {choice1}.', 'I think {choice2}.'],
        mode='train')

    test_utils.assert_dataset(
        dataset,
        [
            {
                'idx': [0, 0],
                'inputs':
                    'The farmland needed irrigation. What is the effect? X',
                'targets': 'I think a canal was constructed.',
                'is_correct': True
            },
            {
                'idx': [1, 1],
                'inputs':
                    'I decided to stay home last night. What is the cause? X',
                'targets': 'I think I was too tired.',
                'is_correct': True
            },
        ])

    # fewshot_eval mode
    dataset = prep.rank_classification_formatter(
        input_ds,
        inputs_formats='{premise} What is the {question}? X',
        targets_formats=['I think {choice1}.', 'I think {choice2}.'],
        mode='fewshot_eval')

    test_utils.assert_dataset(
        dataset,
        [
            {
                'idx': [[0, 0], [0, 1]],
                'inputs': [
                    'The farmland needed irrigation. What is the effect? X',
                    'The farmland needed irrigation. What is the effect? X',
                ],
                'targets': [
                    'I think a canal was constructed.',
                    'I think the crops grew tall.',
                ],
                'is_correct': [True, False]
            },
            {
                'idx': [[1, 0], [1, 1]],
                'inputs': [
                    'I decided to stay home last night. What is the cause? X',
                    'I decided to stay home last night. What is the cause? X',
                ],
                'targets': [
                    'I think I wanted to see people.',
                    'I think I was too tired.',
                ],
                'is_correct': [False, True]
            },
        ])

  def test_nested_key_rank_classification_formatter(self):
    input_ds = tf.data.Dataset.from_tensors({
        'answerKey': 0,
        'fact1': 'creating paper requires cutting down trees',
        'question': {
            'choice_A': 'forests',
            'choice_B': 'canyons',
            'sub_question': {
                'stem': 'What is the ultimate source of greeting cards?'
            }
        }
    })

    dataset = prep.rank_classification_formatter(
        input_ds,
        inputs_formats='{fact1}. {question/sub_question/stem} X 0',
        targets_formats=[
            'Correct Answer: {question/choice_A} X 1 Incorrect Answer: '
            '{question/choice_B} X 1',
            'Correct Answer: {question/choice_B} X 1 Incorrect Answer: '
            '{question/choice_A} X 1',
        ],
        mode='eval',
        label_key='answerKey')

    test_utils.assert_dataset(
        dataset,
        [
            {
                'idx': [0, 0],
                'inputs':
                    'creating paper requires cutting down trees. What is the '
                    'ultimate source of greeting cards? X 0',
                'targets':
                    'Correct Answer: forests X 1 Incorrect Answer: canyons X 1',
                'is_correct': True,
            },
            {
                'idx': [0, 1],
                'inputs':
                    'creating paper requires cutting down trees. What is the '
                    'ultimate source of greeting cards? X 0',
                'targets':
                    'Correct Answer: canyons X 1 Incorrect Answer: forests X 1',
                'is_correct': False,
            },
        ])

    with self.assertRaisesRegex(
        ValueError,
        'Final value of key \'question/sub_question\' must be a tf.string. '
        'Got: dict'):
      prep.rank_classification_formatter(
          input_ds,
          inputs_formats='{fact1}. {question/sub_question} X 0',
          targets_formats=['test1', 'test2'],
          mode='eval',
          label_key='answerKey')

    with self.assertRaises(TypeError):
      prep.rank_classification_formatter(
          input_ds,
          inputs_formats='{fact1}. {answerKey} X 0',
          targets_formats=['test1', 'test2'],
          mode='eval',
          label_key='answerKey')

  def test_rank_classification_formatter_with_weight(self):
    input_examples = [
        {
            'premise': 'The farmland needed irrigation.',
            'question': 'effect',
            'choice1': 'a canal was constructed',
            'choice2': 'the crops grew tall',
            'label': 0,
            'weight': 1,
        },
        {
            'premise': 'I decided to stay home last night.',
            'question': 'cause',
            'choice1': 'I wanted to see people',
            'choice2': 'I was too tired',
            'label': 1,
            'weight': 0.5,
        },
    ]

    input_ds = tf.data.Dataset.from_generator(
        lambda: (x for x in input_examples),
        output_types={
            'premise': tf.string,
            'question': tf.string,
            'choice1': tf.string,
            'choice2': tf.string,
            'label': tf.int32,
            'weight': tf.float32,
        },
        output_shapes={
            'premise': [],
            'question': [],
            'choice1': [],
            'choice2': [],
            'label': [],
            'weight': [],
        })

    # all options
    dataset = prep.rank_classification_formatter(
        input_ds,
        inputs_formats='{premise} What is the {question}? X',
        targets_formats=['I think {choice1}.', 'I think {choice2}.'],
        weight_key='weight',
        mode='eval')

    test_utils.assert_dataset(dataset, [
        {
            'idx': [0, 0],
            'inputs': 'The farmland needed irrigation. What is the effect? X',
            'targets': 'I think a canal was constructed.',
            'is_correct': True,
            'weight': 1.0,
        },
        {
            'idx': [0, 1],
            'inputs': 'The farmland needed irrigation. What is the effect? X',
            'targets': 'I think the crops grew tall.',
            'is_correct': False,
            'weight': 1.0,
        },
        {
            'idx': [1, 0],
            'inputs': 'I decided to stay home last night. What is the cause? X',
            'targets': 'I think I wanted to see people.',
            'is_correct': False,
            'weight': 0.5,
        },
        {
            'idx': [1, 1],
            'inputs': 'I decided to stay home last night. What is the cause? X',
            'targets': 'I think I was too tired.',
            'is_correct': True,
            'weight': 0.5,
        },
    ])

    # train mode
    dataset = prep.rank_classification_formatter(
        input_ds,
        inputs_formats='{premise} What is the {question}? X',
        targets_formats=['I think {choice1}.', 'I think {choice2}.'],
        weight_key='weight',
        mode='train')

    test_utils.assert_dataset(dataset, [
        {
            'idx': [0, 0],
            'inputs': 'The farmland needed irrigation. What is the effect? X',
            'targets': 'I think a canal was constructed.',
            'is_correct': True,
            'weight': 1.0,
        },
        {
            'idx': [1, 1],
            'inputs': 'I decided to stay home last night. What is the cause? X',
            'targets': 'I think I was too tired.',
            'is_correct': True,
            'weight': 0.5,
        },
    ])

    # fewshot_eval mode
    dataset = prep.rank_classification_formatter(
        input_ds,
        inputs_formats='{premise} What is the {question}? X',
        targets_formats=['I think {choice1}.', 'I think {choice2}.'],
        weight_key='weight',
        mode='fewshot_eval')

    test_utils.assert_dataset(dataset, [
        {
            'idx': [[0, 0], [0, 1]],
            'inputs': [
                'The farmland needed irrigation. What is the effect? X',
                'The farmland needed irrigation. What is the effect? X',
            ],
            'targets': [
                'I think a canal was constructed.',
                'I think the crops grew tall.',
            ],
            'is_correct': [True, False],
            'weight': 1.0,
        },
        {
            'idx': [[1, 0], [1, 1]],
            'inputs': [
                'I decided to stay home last night. What is the cause? X',
                'I decided to stay home last night. What is the cause? X',
            ],
            'targets': [
                'I think I wanted to see people.',
                'I think I was too tired.',
            ],
            'is_correct': [False, True],
            'weight': 0.5,
        },
    ])

  def test_select_random_chunk(self):
    dataset = tf.data.Dataset.from_tensors({
        'targets': [0, 1, 2, 3],
        'inputs': [4, 5, 6, 7]
    })
    dataset = prep.select_random_chunk(
        dataset, output_features=None, feature_key='targets', max_length=4)
    output = list(dataset.as_numpy_iterator())
    self.assertLen(output, 1)
    output = output[0]
    self.assertSequenceEqual(['targets'], list(output.keys()))
    self.assertNotEmpty(output['targets'])

  def test_select_random_chunk_rank2(self):
    dataset = tf.data.Dataset.from_tensors({
        'targets': [[0, 9], [1, 8], [2, 7], [3, 6]],
        'inputs': [4, 5, 6, 7]
    })
    dataset = prep.select_random_chunk(
        dataset, output_features=None, feature_key='targets', max_length=4)
    output = list(dataset.as_numpy_iterator())
    self.assertLen(output, 1)
    output = output[0]
    self.assertSequenceEqual(['targets'], list(output.keys()))
    self.assertNotEmpty(output['targets'])

  def test_select_random_chunk_passthrough(self):
    dataset = tf.data.Dataset.from_tensors({
        'targets': [0, 1, 2, 3],
        'inputs': [4, 5, 6, 7],
        'notes': 'hi',
    })
    dataset = prep.select_random_chunk(
        dataset, output_features=None, feature_key='targets', max_length=4,
        passthrough_feature_keys=['notes'])
    output = list(dataset.as_numpy_iterator())
    output = output[0]
    self.assertSequenceEqual(['targets', 'notes'], list(output.keys()))
    self.assertStringEqual('hi', output['notes'])

  def test_select_random_chunk_uniform_start(self):
    dataset = tf.data.Dataset.from_tensors({
        'targets': [0, 1, 2, 3],
        'inputs': [4, 5, 6, 7]
    })
    dataset = prep.select_random_chunk(
        dataset, output_features=None, feature_key='targets', max_length=4,
        uniform_random_start=True)
    output = list(dataset.as_numpy_iterator())
    self.assertLen(output, 1)
    output = output[0]
    self.assertSequenceEqual(['targets'], list(output.keys()))
    self.assertNotEmpty(output['targets'])

  def test_select_random_chunk_additional_features(self):
    dataset = tf.data.Dataset.from_tensors({
        'targets': [0, 1, 2, 3],
        'inputs': [[4, 8], [5, 10], [6, 12], [7, 14]]
    })
    dataset = prep.select_random_chunk(
        dataset, output_features=None, feature_key='targets',
        additional_feature_keys=['inputs'], max_length=3)
    output = list(dataset.as_numpy_iterator())
    self.assertLen(output, 1)
    output = output[0]
    self.assertSequenceEqual(['inputs', 'targets'], sorted(list(output.keys())))
    self.assertAllEqual(output['inputs'][:, 0] - 4, output['targets'])

  def test_select_random_chunk_min_length(self):
    dataset = tf.data.Dataset.from_tensors({
        'targets': [0, 1, 2, 3],
        'inputs': [4, 5, 6, 7]
    })
    dataset = prep.select_random_chunk(
        dataset, output_features=None, feature_key='targets', max_length=4,
        uniform_random_start=True, min_length=1)
    output = list(dataset.as_numpy_iterator())
    self.assertLen(output, 1)
    output = output[0]
    self.assertSequenceEqual(['targets'], list(output.keys()))
    self.assertNotEmpty(output['targets'])

  def test_select_random_chunk_different_sizes(self):
    dataset = tf.data.Dataset.from_tensors({
        'targets': [0, 1, 2, 3],
        'inputs': [4, 5]
    })
    with self.assertRaises(tf.errors.InvalidArgumentError):
      dataset = prep.select_random_chunk(
          dataset, output_features=None, feature_key='targets',
          additional_feature_keys=['inputs'], max_length=4)
      _ = list(dataset.as_numpy_iterator())

  def test_pack_prefix_lm_encoder_decoder(self):

    x = [{'targets': [0, 1, 2, 3, 4, 5, 6, 7]},
         {'targets': [8, 9, 10, 11, 12, 13, 14, 15]},
         {'targets': [16, 17, 18, 19, 20, 21, 22, 23]},
         {'targets': [24, 25, 26, 27, 28, 29, 30, 31]}]
    ds = test_utils.create_default_dataset(
        x, feature_names=['targets'], output_shapes={'targets': [8]})

    # With this seed, split points are 3 and 5
    with seqio.utils.map_seed_manager(2):
      packed_ds = prep.pack_prefix_lm_encoder_decoder(
          ds, {'inputs': 8, 'targets': 8})

    expected = [
        {
            'encoder_input_tokens': [0, 1, 2, 8, 9, 10, 11, 12],
            'decoder_target_tokens': [3, 4, 5, 6, 7, 13, 14, 15],
            # The first token of the second sequence (in this case index 5)
            # should be 0 instead of the last token of the first sequence.
            'decoder_input_tokens': [0, 3, 4, 5, 6, 0, 13, 14],
            'encoder_segment_ids': [1, 1, 1, 2, 2, 2, 2, 2],
            'encoder_positions': [0, 1, 2, 0, 1, 2, 3, 4],
            'decoder_loss_weights': [1, 1, 1, 1, 1, 1, 1, 1],
            'decoder_segment_ids': [1, 1, 1, 1, 1, 2, 2, 2],
            'decoder_positions': [0, 1, 2, 3, 4, 0, 1, 2],
        },
        {
            'encoder_input_tokens': [16, 17, 18, 19, 20, 24, 25, 26],
            'decoder_target_tokens': [21, 22, 23, 27, 28, 29, 30, 31],
            'decoder_input_tokens': [0, 21, 22, 0, 27, 28, 29, 30],
            'encoder_segment_ids': [1, 1, 1, 1, 1, 2, 2, 2],
            'encoder_positions': [0, 1, 2, 3, 4, 0, 1, 2],
            'decoder_loss_weights': [1, 1, 1, 1, 1, 1, 1, 1],
            'decoder_segment_ids': [1, 1, 1, 2, 2, 2, 2, 2],
            'decoder_positions': [0, 1, 2, 0, 1, 2, 3, 4],
        }
    ]
    assert_dataset(packed_ds, expected)

  def test_pack_prefix_lm_encoder_decoder_with_padding(self):
    x = [{'targets': [9, 1, 2, 3, 4, 5, 6, 0]},
         {'targets': [8, 9, 10, 11, 12, 13, 0, 0]}]
    ds = test_utils.create_default_dataset(
        x, feature_names=['targets'], output_shapes={'targets': [8]})

    # With this seed, split point is 3.
    with seqio.utils.map_seed_manager(2):
      packed_ds = prep.pack_prefix_lm_encoder_decoder(
          ds, {'inputs': 8, 'targets': 8})

    expected = [
        {
            'encoder_input_tokens': [9, 1, 2, 8, 9, 10, 11, 12],
            'decoder_target_tokens': [3, 4, 5, 6, 0, 13, 0, 0],
            'decoder_input_tokens': [0, 3, 4, 5, 6, 0, 13, 0],
            'encoder_segment_ids': [1, 1, 1, 2, 2, 2, 2, 2],
            'encoder_positions': [0, 1, 2, 0, 1, 2, 3, 4],
            'decoder_loss_weights': [1, 1, 1, 1, 0, 1, 0, 0],
            'decoder_segment_ids': [1, 1, 1, 1, 1, 2, 2, 2],
            'decoder_positions': [0, 1, 2, 3, 4, 0, 1, 2],
        },
    ]
    assert_dataset(packed_ds, expected)

  def test_pack_prefix_lm_decoder_only(self):
    x = [{'targets': [9, 1, 2, 3, 4, 5, 6, 7]},
         {'targets': [8, 9, 10, 11, 12, 13, 14, 15]}]
    ds = test_utils.create_default_dataset(x, feature_names=['targets'])

    # With this seed, split points are 3 and 5.
    with seqio.utils.map_seed_manager(2):
      packed_ds = prep.pack_prefix_lm_decoder_only(ds, {'length': 8})

    expected = [{
        'decoder_target_tokens': [9, 1, 2, 3, 4, 5, 6, 7],
        'decoder_input_tokens': [0, 9, 1, 2, 3, 4, 5, 6],
        'decoder_loss_weights': [0, 0, 0, 1, 1, 1, 1, 1],
        'decoder_causal_attention': [1, 1, 1, 1, 0, 0, 0, 0],
    }, {
        'decoder_target_tokens': [8, 9, 10, 11, 12, 13, 14, 15],
        'decoder_input_tokens': [0, 8, 9, 10, 11, 12, 13, 14],
        'decoder_loss_weights': [0, 0, 0, 0, 0, 1, 1, 1],
        'decoder_causal_attention': [1, 1, 1, 1, 1, 1, 0, 0],
    }]
    assert_dataset(packed_ds, expected)

  def test_pack_prefix_lm_decoder_only_with_padding(self):
    x = [{'targets': [8, 9, 10, 11, 12, 13, 0, 0]}]
    ds = test_utils.create_default_dataset(x, feature_names=['targets'])

    # With this seed, split point is 3.
    with seqio.utils.map_seed_manager(2):
      packed_ds = prep.pack_prefix_lm_decoder_only(ds, {'length': 8})

    expected = [{
        'decoder_target_tokens': [8, 9, 10, 11, 12, 13, 0, 0],
        'decoder_input_tokens': [0, 8, 9, 10, 11, 12, 13, 0],
        'decoder_loss_weights': [0, 0, 0, 1, 1, 1, 0, 0],
        'decoder_causal_attention': [1, 1, 1, 1, 0, 0, 0, 0],
    }]
    assert_dataset(packed_ds, expected)

  def test_pack_prefix_lm_decoder_only_with_padding_loss_on_targets_false(self):
    x = [{'targets': [8, 9, 10, 11, 12, 13, 0, 0]}]
    ds = test_utils.create_default_dataset(x, feature_names=['targets'])

    # With this seed, split point is 3.
    with seqio.utils.map_seed_manager(2):
      packed_ds = prep.pack_prefix_lm_decoder_only(
          ds, {'length': 8}, loss_on_targets_only=False)

    expected = [{
        'decoder_target_tokens': [8, 9, 10, 11, 12, 13, 0, 0],
        'decoder_input_tokens': [0, 8, 9, 10, 11, 12, 13, 0],
        'decoder_loss_weights': [1, 1, 1, 1, 1, 1, 0, 0],
        'decoder_causal_attention': [1, 1, 1, 1, 0, 0, 0, 0],
    }]
    assert_dataset(packed_ds, expected)

  def test_preprocess_tsv_with_field_names(self):
    x = tf.data.Dataset.from_tensor_slices(['6,7,42'])
    dataset = prep.preprocess_tsv(
        x,
        field_delim=',',
        field_names=['quot', 'denom', 'numer'],
        inputs_format='numerator: {numer} denominator: {denom}',
        targets_format='quotient: {quot}')
    expected = {
        'inputs': 'numerator: 42 denominator: 7',
        'targets': 'quotient: 6'
    }
    assert_dataset(dataset, expected)

  def test_preprocess_tsv_with_positions(self):
    x = tf.data.Dataset.from_tensor_slices(
        ['6,7,42,43,44,45,46,47,48,49,fifty,51,52'])
    dataset = prep.preprocess_tsv(
        x,
        num_fields=13,
        field_delim=',',
        inputs_format='numerator: {2} denominator: {1} fact: {12} - 50 != {0}',
        targets_format='quotient: {0} and yes: 52 - {10} != 6')
    expected = {
        'inputs': 'numerator: 42 denominator: 7 fact: 52 - 50 != 6',
        'targets': 'quotient: 6 and yes: 52 - fifty != 6'
    }
    assert_dataset(dataset, expected)

  # TODO(adarob): Add more than a smoke test.
  def test_span_corruption(self):
    vocab = test_utils.sentencepiece_vocab()
    inp = list(range(1, 100))
    og_dataset = tf.data.Dataset.from_tensor_slices({'targets': [inp]})
    og_dataset = og_dataset.repeat(100)
    output_features = {
        'targets': seqio.Feature(vocab),
        'inputs': seqio.Feature(vocab),
    }
    output_dataset = prep.span_corruption(
        og_dataset,
        sequence_length={'targets': 100, 'inputs': 100},
        output_features=output_features,
        merge_examples_to_reduce_padding=True)
    output_keys = list(output_dataset.as_numpy_iterator())[0].keys()
    self.assertSequenceEqual(['inputs', 'targets'], list(output_keys))

  def test_span_corruption_passthrough(self):
    # No merging of examples, passthrough keys
    vocab = test_utils.sentencepiece_vocab()
    inp = list(range(1, 100))
    pt = list(range(1, 20))
    og_dataset = tf.data.Dataset.from_tensor_slices({
        'targets': [inp],
        'passthrough': [pt],
    })
    og_dataset = og_dataset.repeat(100)
    output_features = {
        'targets': seqio.Feature(vocab),
        'inputs': seqio.Feature(vocab),
        'passthrough': seqio.Feature(vocab),
    }

    output_dataset = prep.span_corruption(
        og_dataset,
        sequence_length={'targets': 100, 'inputs': 100},
        output_features=output_features,
        merge_examples_to_reduce_padding=False,
        passthrough_feature_keys=['passthrough'])

    for ex in output_dataset.as_numpy_iterator():
      self.assertLessEqual(len(ex['inputs']), len(inp))
      self.assertAllEqual(pt, ex['passthrough'])

  def test_span_corruption_passthrough_fail(self):
    og_dataset = tf.data.Dataset.from_tensor_slices({
        'targets': [list(range(1, 100))],
        'passthrough': [list(range(1, 20))],
    })
    with self.assertRaises(ValueError):
      _ = prep.span_corruption(
          og_dataset,
          sequence_length={'targets': 100, 'inputs': 100},
          output_features=None,
          merge_examples_to_reduce_padding=True,
          passthrough_feature_keys=['passthrough'])


if __name__ == '__main__':
  absltest.main()
