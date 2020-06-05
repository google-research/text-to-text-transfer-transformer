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

"""Tests for from t5.preprocessors."""

from absl.testing import absltest
from t5.data import preprocessors as prep
from t5.data import test_utils
from t5.data import utils
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
tf.enable_eager_execution()

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
          noise_density=noise_density)
      first = noise_mask[0]
      last = noise_mask[-1]
      self.assertTrue(self.evaluate(first))
      self.assertFalse(self.evaluate(last))

  def test_random_spans_noise_mask(self):
    tf.set_random_seed(55)
    length = 32
    noise_density = 0.25
    mean_noise_span_length = 2.0
    # there should be 4 noise spans with a total length of 8.
    noise_mask = prep.random_spans_noise_mask(
        length, noise_density, mean_noise_span_length)
    output = self.evaluate(tf.cast(noise_mask, tf.int32))
    expected_output = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1]
    self.assertAllEqual(output, expected_output)

  def test_noise_token_to_sentinel(self):
    vocabulary = test_utils.mock_vocabulary({'foo': 10}, vocab_size=1000)
    tokens = tf.constant([10, 11, 12, 13, 14, 15])
    noise_mask = tf.constant([True, True, False, False, True, False])
    expected_output = [999, 999, 12, 13, 999, 15]
    output = self.evaluate(prep.noise_token_to_sentinel(
        tokens, noise_mask, vocabulary))
    self.assertAllEqual(output, expected_output)

  def test_noise_span_to_sentinel(self):
    vocabulary = test_utils.mock_vocabulary({'foo': 10}, vocab_size=1000)
    tokens = tf.constant([10, 11, 12, 13, 14, 15])
    noise_mask = tf.constant([True, True, False, False, True, False])
    expected_output = [999, 12, 13, 999, 15]
    output = self.evaluate(prep.noise_span_to_sentinel(
        tokens, noise_mask, vocabulary))
    self.assertAllEqual(output, expected_output)

  def test_nonnoise_span_to_sentinel(self):
    vocabulary = test_utils.mock_vocabulary({'foo': 10}, vocab_size=1000)
    tokens = tf.constant([10, 11, 12, 13, 14, 15])
    noise_mask = tf.constant([True, True, False, False, True, False])
    expected_output = [10, 11, 999, 14, 999]
    output = self.evaluate(prep.nonnoise_span_to_sentinel(
        tokens, noise_mask, vocabulary))
    self.assertAllEqual(output, expected_output)

  def test_noise_span_to_unique_sentinel(self):
    vocabulary = test_utils.mock_vocabulary({'foo': 10}, vocab_size=1000)
    tokens = tf.constant([10, 11, 12, 13, 14, 15])
    noise_mask = tf.constant([True, True, False, False, True, False])
    expected_output = [999, 12, 13, 998, 15]
    output = self.evaluate(prep.noise_span_to_unique_sentinel(
        tokens, noise_mask, vocabulary))
    self.assertAllEqual(output, expected_output)

  def test_drop_noise_tokens(self):
    vocabulary = test_utils.mock_vocabulary({'foo': 10}, vocab_size=1000)
    tokens = tf.constant([10, 11, 12, 13, 14, 15])
    noise_mask = tf.constant([True, True, False, False, True, False])
    expected_output = [12, 13, 15]
    output = self.evaluate(prep.drop_noise_tokens(
        tokens, noise_mask, vocabulary))
    self.assertAllEqual(output, expected_output)

  def test_drop_nonnoise_tokens(self):
    vocabulary = test_utils.mock_vocabulary({'foo': 10}, vocab_size=1000)
    tokens = tf.constant([10, 11, 12, 13, 14, 15])
    noise_mask = tf.constant([True, True, False, False, True, False])
    expected_output = [10, 11, 14]
    output = self.evaluate(prep.drop_nonnoise_tokens(
        tokens, noise_mask, vocabulary))
    self.assertAllEqual(output, expected_output)

  def test_permute_noise_tokens(self):
    tf.set_random_seed(55)
    vocabulary = test_utils.mock_vocabulary({'foo': 10}, vocab_size=1000)
    tokens = tf.constant([10, 11, 12, 13, 14, 15])
    noise_mask = tf.constant([True, True, False, False, True, False])
    expected_output = [11, 14, 12, 13, 10, 15]
    output = self.evaluate(prep.permute_noise_tokens(
        tokens, noise_mask, vocabulary))
    self.assertAllEqual(output, expected_output)

  def test_noise_token_to_gathered_token(self):
    tf.set_random_seed(55)
    vocabulary = test_utils.mock_vocabulary({'foo': 10}, vocab_size=1000)
    tokens = tf.constant([10, 11, 12, 13, 14, 15])
    noise_mask = tf.constant([True, True, False, False, True, False])
    expected_output = [11, 11, 12, 13, 15, 15]
    output = self.evaluate(prep.noise_token_to_gathered_token(
        tokens, noise_mask, vocabulary))
    self.assertAllEqual(output, expected_output)

  def test_noise_token_to_random_token(self):
    tf.set_random_seed(55)
    vocabulary = test_utils.mock_vocabulary({'foo': 10}, vocab_size=1000)
    tokens = tf.constant([10, 11, 12, 13, 14, 15])
    noise_mask = tf.constant([True, True, False, False, True, False])
    expected_output = [811, 309, 12, 13, 451, 15]

    output = self.evaluate(prep.noise_token_to_random_token(
        tokens, noise_mask, vocabulary))
    self.assertAllEqual(output, expected_output)

  def test_noise_token_to_random_token_or_sentinel(self):
    tf.set_random_seed(55)
    vocabulary = test_utils.mock_vocabulary({'foo': 10}, vocab_size=1000)
    tokens = tf.constant(list(range(10)))
    noise_mask = tf.constant(
        [True, True, False, False, True, False, True, True, True, True])
    expected_output = [436, 999, 2, 3, 999, 5, 999, 999, 999, 999]
    output = self.evaluate(prep.noise_token_to_random_token_or_sentinel(
        tokens, noise_mask, vocabulary, random_prob=0.2))
    self.assertAllEqual(output, expected_output)

  def test_rekey(self):
    og_dataset = tf.data.Dataset.from_tensors({
        'text': 'That is good.', 'other': 'That is bad.'})
    dataset = prep.rekey(og_dataset, {'inputs': 'other', 'targets': 'text'})
    assert_dataset(
        dataset,
        {'inputs': 'That is bad.', 'targets': 'That is good.'})

    dataset = prep.rekey(og_dataset, {'targets': 'text'})
    assert_dataset(dataset, {'targets': 'That is good.'})

    dataset = prep.rekey(og_dataset, {'inputs': 'text'})
    assert_dataset(dataset, {'inputs': 'That is good.'})

    dataset = prep.rekey(og_dataset)
    assert_dataset(dataset, {'text': 'That is good.', 'other': 'That is bad.'})

    dataset = prep.rekey(og_dataset, {'inputs': 'text', 'targets': None})
    assert_dataset(dataset, {'inputs': 'That is good.', 'targets': ''})

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

  def test_prefix_lm(self):
    num_tries = 100
    original = 'This is a long test with lots of words to see if it works ok.'
    dataset = tf.data.Dataset.from_tensor_slices(
        {'text': [original] * num_tries})
    dataset = prep.prefix_lm(dataset)
    for data in test_utils.dataset_as_text(dataset):
      inputs = data['inputs'].replace('prefix: ', '')
      targets = data['targets']

      reconstructed = ''.join(inputs)
      if inputs:
        reconstructed += ' '
      reconstructed += ''.join(targets)

      self.assertEqual(reconstructed, original)

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

  def test_split_text_to_words(self):
    dataset = tf.data.Dataset.from_tensor_slices(
        {'text': ['That good.', 'That.']})
    dataset = prep._split_text_to_words(dataset)
    assert_dataset(
        dataset,
        {
            'text': 'That good.',
            'words': ['That', 'good.']
        })

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

  def test_take(self):
    og_dataset = tf.data.Dataset.from_tensor_slices({'inputs': [1]*100})
    dataset = prep.take(og_dataset, 5)
    assert_dataset(dataset, [{'inputs': 1} for _ in range(5)])
    dataset = prep.take(og_dataset, -1)
    assert_dataset(dataset, [{'inputs': 1} for _ in range(100)])

  def parse_tsv(self):
    og_dataset = tf.data.Dataset.from_tensor_slices(['a\tb', 'c\td'])
    dataset = prep.parse_tsv(og_dataset, field_names=['f1', 'f2'])
    assert_dataset(dataset, [{'f1': 'a', 'f2': 'b'}, {'f1': 'c', 'f2': 'd'}])

  def test_denoise(self):
    tf.set_random_seed(55)

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
        'targets': utils.Feature(vocab),
    }

    # These are the parameters of denoise in the operative config of 'base'.
    # Except noise_density, bumped up from 0.15 to 0.3 in order to demonstrate
    # multiple corrupted spans.
    denoised_dataset = prep.denoise(
        og_dataset,
        output_features,
        noise_density=0.3,
        noise_mask_fn=prep.random_spans_noise_mask,
        inputs_fn=prep.noise_span_to_unique_sentinel,
        targets_fn=prep.nonnoise_span_to_unique_sentinel)

    # Two spans corrupted, [2] and [22, 3, 2, 7, 2], replaced by unique
    # sentinels 25 and 24 respectively.
    assert_dataset(denoised_dataset, [
        {
            'inputs': [
                3, 25, 20, 4, 3, 2, 8, 13, 2, 3, 2, 23, 7, 19, 24
            ],
            'targets': [
                25, 2, 24, 22, 3, 2, 7, 2
            ],
        },
    ])


if __name__ == '__main__':
  absltest.main()
