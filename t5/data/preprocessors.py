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

"""Preprocessors for T5 Tasks."""
# TODO(adarob): Move some of the more general preprocessors to seqio.

import collections
import functools
import math
import re
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence, Union
import uuid

from absl import logging
import babel
import gin
import seqio
import tensorflow.compat.v2 as tf

# We disable no-value-for-parameter since the seqio.map_over_dataset leads to
# a false positive when seeds are provided.
# pylint:disable=no-value-for-parameter
AUTOTUNE = tf.data.experimental.AUTOTUNE

FeatureType = Mapping[str, tf.Tensor]

rekey = seqio.preprocessors.rekey
tokenize = seqio.preprocessors.tokenize


@seqio.map_over_dataset
def translate(x, source_language, target_language):
  """Convert a translation dataset to a text2text pair.

  For example, say the dataset returns examples of this format:
    {'de': 'Das ist gut.', 'en': 'That is good.'}
  If source_language = 'de', target_language = 'en', then the outputs will have
  the format:
    {'inputs': 'translate German to English: Das ist gut.',
     'targets': 'That is good.'}

  Args:
    x: an example to process.
    source_language: source language code (e.g. 'en') to translate from.
    target_language: target language code (e.g. 'de') to translate to.

  Returns:
    A preprocessed example with the format listed above.
  """
  # Language codes like zh-cn are not supported; use only the first 2 chars
  for language in (source_language, target_language):
    if language != language[:2]:
      logging.warning(
          'Extended language code %s not supported. Falling back on %s.',
          language, language[:2]
      )
  lang_id_to_string = {
      source_language: babel.Locale(source_language[:2]).english_name,
      target_language: babel.Locale(target_language[:2]).english_name,
  }
  src_str = 'translate {}'.format(lang_id_to_string[source_language])
  tgt_str = ' to {}: '.format(lang_id_to_string[target_language])
  return {
      'inputs': tf.strings.join([src_str, tgt_str, x[source_language]]),
      'targets': x[target_language],
  }


@seqio.map_over_dataset
def summarize(x, article_key, summary_key):
  """Convert a summarization dataset to a text2text pair.

  For example, say the dataset returns examples of this format:
    {'article': <article>, 'highlights': <summary>}
  If article_key = 'article', summary_key = 'highlights', then the outputs will
  have the format:
    {'inputs': 'summarize': <article>, 'targets': <summary>}

  Args:
    x: an example to process.
    article_key: the feature key for the article to summarize.
    summary_key: the feature key for the target summary.
  Returns:
    A preprocessed example with the format listed above.
  """
  strs_to_join = ['summarize:', x[article_key]]
  return {
      'inputs': tf.strings.join(strs_to_join, separator=' '),
      'targets': x[summary_key],
  }


# Unicode ranges for characters in non-spaced languages.
# https://en.wikipedia.org/wiki/Category:Writing_systems_without_word_boundaries
# https://en.wikipedia.org/wiki/Han_unification#Unicode_ranges
# https://linguistics.stackexchange.com/questions/6131
NON_SPACED_LANGUAGE_RANGES = (
    '\u1000-\u104f',  # Burmese
    '\u4e00-\u9fff',  # CJK Unified Ideographs
    '\u3400-\u4dbf',  # CJK Unified Ideographs Extension A
    '\uf900-\ufaff',  # CJK Compatibility Ideographs
    '\u2e80-\u2eff',  # CJK Radicals Supplement
    '\u31c0-\u31ef',  # CJK Strokes
    '\u3000-\u303f',  # CJK Symbols and Punctuation
    '\u3040-\u309f',  # Japanese Hiragana
    '\u30a0-\u30ff',  # Japanese Katakana
    '\ua980-\ua9df',  # Javanese
    '\u1780-\u17ff',  # Khmer
    '\u19e0-\u19ff',  # Khmer Symbols
    '\u0e80-\u0eff',  # Lao
    '\u1980-\u19df',  # Tai Lue
    '\u1a20-\u1aaf',  # Tai Tham
    '\u0e00-\u0e7f',  # Thai
    '\u0f00-\u0fff',  # Tibetan
)


@seqio.map_over_dataset
def pad_nonspaced_languages(x, text_key='text'):
  """Pad non-spaced languages with spaces around each character.

  Args:
    x: an example to process.
    text_key: a string, the key for the text feature to preprocess in the
      dataset examples.

  Returns:
    A preprocessed example.
  """
  res = dict(x)
  text = res[text_key]
  # Add spaces around any character from a non-spaced language.
  pattern = ''.join(NON_SPACED_LANGUAGE_RANGES)
  text = tf.strings.regex_replace(text, u'([{}])'.format(pattern), r' \1 ')
  # Collapse consecutive whitespace into one space.
  text = tf.strings.regex_replace(text, r'\s+', ' ')
  res[text_key] = text
  return res


def _pad_punctuation(text):
  """Adds spaces around punctuation."""
  # Add space around punctuation.
  text = tf.strings.regex_replace(text, r'([[:punct:]])', r' \1 ')
  # Collapse consecutive whitespace into one space.
  text = tf.strings.regex_replace(text, r'\s+', ' ')
  return text


def _string_join(lst):
  # Join on space, but collapse consecutive spaces.
  out = tf.strings.join(lst, separator=' ')
  return tf.strings.regex_replace(out, r'\s+', ' ')


def trivia_qa(dataset):
  """Convert a TriviaQA example to multiple flattened examples.

  TriviaQA produces examples with this form:
    {'entity_pages': {dict of wiki entities},
     'search_results': <dict of web search results>,
     'answer': {dict of all answers}, 'question': <question>,
     'question_id': <question_id>, 'question_source': <question_source>}
  This function will return flattend examples of the format:
    {'inputs': 'question: <question> context: <article>'
     'targets': 'answer: <sampled answer>'}

  Args:
    dataset: a tf.data.Dataset to process.
  Returns:
    A preprocessed tf.data.Dataset with the format listed above.
  """
  def triviaqa_question_answer_context(x):
    """Extracts matched contexts and answers.

    Returns all matched (question-context, answer) pairs.

    Args:
      x: A tfds sample.

    Returns:
      Flattened samples: (question-context, answer).
    """
    contexts = []
    if 'entity_pages' in x:
      contexts.append(x['entity_pages']['wiki_context'])
    if 'search_results' in x:
      contexts.append(x['search_results']['search_context'])
    contexts = tf.concat(contexts, 0)

    q = _pad_punctuation(x['question'])
    answers = x['answer']['normalized_aliases']

    combination_size = tf.size(answers)*tf.size(contexts)
    find_answers = tf.TensorArray(
        tf.bool, size=combination_size, dynamic_size=True)
    selected_answers = tf.TensorArray(
        tf.string, size=combination_size, dynamic_size=True)
    join_q_c = tf.TensorArray(
        tf.string, size=combination_size, dynamic_size=True)

    def cond_fn(i, find_answers, selected_answers, join_q_c):
      del find_answers, selected_answers, join_q_c  # Unused
      return tf.less(i, combination_size)

    def body_fn(i, find_answers, selected_answers, join_q_c):
      """Find answers from contexts and join."""
      context_idx = tf.math.floordiv(i, tf.size(answers))
      answer_idx = tf.math.mod(i, tf.size(answers))

      a = _pad_punctuation(answers[answer_idx])
      a_ = tf.strings.join(['.*', a, '.*'])
      c = _pad_punctuation(contexts[context_idx])
      find_a = tf.strings.regex_full_match(
          tf.strings.lower(c),
          tf.strings.lower(a_))
      find_answers = find_answers.write(i, find_a)
      selected_answers = selected_answers.write(i, a)

      join_q_c_str = _string_join(['question:', q, 'context:', c])
      join_q_c = join_q_c.write(i, join_q_c_str)
      return (i + 1, find_answers, selected_answers, join_q_c)

    _, find_answers, selected_answers, join_q_c = tf.while_loop(
        cond_fn,
        body_fn,
        loop_vars=[
            tf.constant(0), find_answers, selected_answers,
            join_q_c
        ])
    find_answers = find_answers.stack()
    selected_answers = selected_answers.stack()
    join_q_c = join_q_c.stack()

    selected_answers = tf.boolean_mask(selected_answers, find_answers)
    selected_join_q_c = tf.boolean_mask(join_q_c, find_answers)

    return selected_join_q_c, selected_answers

  def my_fn(x):
    """Create TriviaQA example."""
    join_q_c, a = triviaqa_question_answer_context(x)
    return {
        'inputs': join_q_c,
        'targets': a
    }

  dataset = dataset.map(my_fn, num_parallel_calls=AUTOTUNE)
  return dataset.unbatch()


@seqio.map_over_dataset
def squad(x, include_context=True):
  """Convert SQuAD examples to a text2text pair.

  SQuAD produces examples with this form:
    {'id': <id>, context': <article>, 'question': <question>,
     'answers': { 'text': [<n answers>] }}
  This function will return examples of the format:
    {'inputs': 'question: <question> context: <article>',
     'targets': '<answer_0>',
     'id': <id>, 'question': <question>, 'context': <context>,
     'answers': [<n answers>]},

  Args:
    x: an example to process.
    include_context: a boolean
  Returns:
    A preprocessed example with the format listed above.
  """
  a = _pad_punctuation(x['answers']['text'])
  q = _pad_punctuation(x['question'])
  c = _pad_punctuation(x['context'])
  if include_context:
    inputs = _string_join(['question:', q, 'context:', c])
  else:
    inputs = _string_join(['squad trivia question:', q])
  return {
      'inputs': inputs,
      'targets': a[0],
      'id': x['id'],
      'context': c,
      'question': q,
      'answers': a
  }


def _span_answer(context, answer_text):
  """Finds start/end indices of answer_text in context after space tokenization.

  If answer_tokens is not a sublist of context_tokens, returns empty string.

  Args:
    context: 0-d string tensor
    answer_text: 0-d string

  Returns:
    A string tensor.
  """
  def space_tok(s):
    """Replace non-word chars with space then split on space."""
    s = tf.strings.regex_replace(s, r'\W', ' ')
    return tf.strings.split(input=[s], sep=' ').values

  def find_subseq(n, h):
    """Finds index of needle subsequence inside haystack.

    Args:
      n: 1-d tensor
      h: 1-d tensor same type as n

    Returns:
      Index of start of n if found found; otherwise -1.
    """
    l_n = tf.size(n)
    l_h = tf.size(h)
    found = -1
    for i in tf.range(0, l_h - l_n):
      if tf.reduce_all(tf.equal(h[i:i+l_n], n)):
        found = i
        break
    return found

  answer_tokens = space_tok(answer_text)
  context_tokens = space_tok(context)
  start = find_subseq(answer_tokens, context_tokens)
  end = start + tf.size(answer_tokens) - 1
  # Just take the first candidate that matches exactly.
  if tf.equal(start, -1):
    return ''
  return tf.strings.format('start: {} end: {}', [start, end])


def squad_span_space_tokenized(dataset):
  """Convert SQuAD examples to a text2text pair with span output.

  SQuAD produces examples with this form:
    {'context': <article>, 'question': <question>,
     'answers': { 'text': [<all answers>] }}

  This function returns examples with the format
    {'inputs': 'context: <article> question: <question>',
     'targets': 'start: <start_index> end: <end_index>'}
  where <start_index> and <end_index> specify the space-tokenized span
  start/end indices. Both <start_index> and <end_index> are included in
  the answer. In the case where the tokenized answer is
  not found in the tokenized context, the example is skipped.

  Args:
    dataset: a tf.data.Dataset to process.
  Returns:
    A preprocessed tf.data.Dataset with the format listed above.
  """
  def my_fn(x):
    """Create squad example as in squad_span_char, but tokenized on spaces."""
    res = dict(x)
    res['targets'] = _span_answer(x['context'], x['targets'])
    return res

  dataset = squad(dataset)
  dataset = dataset.map(my_fn, num_parallel_calls=AUTOTUNE)
  return dataset.filter(lambda x: tf.strings.length(x['targets']) > 0)


def random_split_text(dataset,
                      text_key='text',
                      min_words_per_segment=16,
                      max_words_per_segment=512,
                      max_words_total=8192):
  """Randomly split single-string examples into multiple examples each.

  Segment lengths are chosen according to a log-uniform distribution.
  Each incoming string is chopped into multiple equal-length examples
  with the last one possibly being shorter.

  If the input string is longer than max_words_total, then we use one random
  chunk and discard the rest.  This may help with model stability.

  The intended use case is to break up long text examples for use in
  unsupervised transfer-learning.

  We don't really want to use this preprocessor for any dataset which has a
  well-defined evaluation procedure. If apply this preprocessor e.g. in an MT
  component, then the evaluation job will randomly split text when evaluating
  and the BLEU will get funky.

  Args:
    dataset: a tf.data.Dataset with dictionaries containing the key text_key
    text_key: a string
    min_words_per_segment: an integer
    max_words_per_segment: an integer
    max_words_total: an integer

  Returns:
    a dataset
  """
  def random_chunk(x, chunk_size, seed):
    """Pick a random chunk of a 1d Tensor.

    The tensor is divided into chunks of length chunk_size, with the last
    chunk being potentially smaller.  A random chunk is returned.

    Args:
      x: a 1d tf.Tensor.
      chunk_size: an integer.
      seed: int32 [2]-Tensor, the random seed.
    Returns:
      a 1d tf.Tensor with length <= chunk_size.
    """
    size = tf.size(x)
    num_chunks = tf.maximum(1, (size - 1) // chunk_size + 1)
    chunk_num = tf.random.stateless_uniform(
        [],
        seed=seed,
        minval=0,
        maxval=num_chunks,
        dtype=tf.int32)
    return x[chunk_size * chunk_num:chunk_size * (chunk_num + 1)]

  @seqio.map_over_dataset(num_seeds=2)
  def my_fn(x, seeds):
    """Split one string into multiple strings.

    Args:
      x: a feature dictionary
      seeds: an int32 Tensor, shaped (2, 2), the random seeds.
    Returns:
      a feature dictionary
    """
    text = x[text_key]
    words = tf.strings.split([text]).values
    if max_words_total:
      words = random_chunk(words, max_words_total, seed=seeds[0])
    n_words = tf.size(words)
    # first pick a length (number of words per segment)
    length = tf.cast(
        tf.exp(
            tf.random.stateless_uniform(
                [],
                minval=math.log(min_words_per_segment),
                maxval=math.log(max_words_per_segment),
                seed=seeds[1],
            )
        ),
        tf.int32)
    # Pad to a multiple of length, then use tf.reshape to split up the words
    # into num_segments segments each of the given length.
    num_segments = tf.cast(
        tf.math.ceil(
            tf.cast(n_words, tf.float32) / tf.cast(length, tf.float32)
        ),
        tf.int32)
    padding = num_segments * length - n_words
    words = tf.pad(words, [[0, padding]])
    words = tf.reshape(words, [-1, length])
    # Finally, join with spaces and strip.  The padding turns into a bunch of
    # spaces that get stripped out.
    words = tf.strings.reduce_join(words, axis=1, separator=' ')
    return {text_key: tf.strings.strip(words)}

  return my_fn(dataset).unbatch()


def split_text_to_words(dataset, text_key='text', min_num_words=2):
  """Split text to words and filter out examples with too few words."""
  def split(x):
    res = dict(x)
    res['words'] = tf.strings.split([x[text_key]]).values
    return res

  dataset = dataset.map(split, num_parallel_calls=AUTOTUNE)
  return dataset.filter(lambda x: tf.size(x['words']) >= min_num_words)


def fill_in_the_blank(dataset,
                      text_key='text',
                      label='fill: '):
  """Create a dataset consisting of fill-in-the-blank text examples.

  The input examples should have a key text_key associated with a tf.string
  value.

  The output examples have keys 'inputs' and 'targets'.

  The input string is split on whitespace to form a sequence of words.
  This sequence is chopped randomly into segments of one or more words.
  Alternate segments are included in the inputs and targets, with a special
  word 'X' marking a missing segment.

  The given label is prepended to the inputs. Each input string produces two
  examples - one the inverse of the other. Inputs with less than two words
  are dropped.

  EXAMPLE:

  input:
  {
    'text': 'The fat cat sat on the mat.'
  }
  outputs:
  {
    'inputs': 'fill: The fat X the X'
    'targets': 'X cat sat on X mat.'
  }
  {
    'inputs': 'fill: X cat sat on X mat.'
    'targets': 'The fat X the X'
  }

  Args:
    dataset: a tf.data.Dataset
    text_key: a string, the key for the text feature to preprocess in the
      dataset examples.
    label: a string, the label to prepend to the inputs.
  Returns:
    a tf.data.Dataset
  """
  @seqio.map_over_dataset(num_seeds=3)
  def my_fn(x, seeds):
    """Generates two preprocessed examples that are roughly inverses.

    Args:
      x: an example dict with text pre-split in `words` feature.
      seeds: an int32 Tensor, shaped (3, 2), the random seeds.
    Returns:
      an example dict with two inputs and two targets, one for each resulting
      preprocessed example.
    """
    words = x['words']
    n_words = tf.size(words)

    # First select the break probability.  We pick this on a log-uniform
    # distribution between 1/(n_words + 1) and  1/2.  This means that some
    # sequences will be chopped roughly and others finely.
    min_log_p_break = -tf.math.log(tf.cast(n_words, tf.float32) + 2.0)
    max_log_p_break = -tf.math.log(2.0)
    p_break = tf.exp(
        tf.random.stateless_uniform(
            [],
            minval=min_log_p_break,
            maxval=max_log_p_break,
            seed=seeds[0])
    )
    # craffel@ says that there may be bugs in random.uniform making it not
    # really uniform.  This doesn't seem horribly important here, but may
    # need another look.
    breaks = tf.less(
        tf.random.stateless_uniform([n_words - 1], seed=seeds[1]),
        p_break)
    def one_random_break():
      pos = tf.random.stateless_uniform(
          [],
          minval=0,
          maxval=n_words - 1,
          dtype=tf.int32,
          seed=seeds[2])
      return tf.one_hot(pos, n_words - 1,
                        dtype=tf.bool, on_value=True, off_value=False)
    breaks = tf.cond(
        tf.math.reduce_any(breaks), lambda: breaks, one_random_break)
    breaks = tf.concat([[True], breaks], axis=0)
    word_to_seq_id = tf.math.mod(tf.math.cumsum(tf.cast(breaks, tf.int32)), 2)
    # separators:
    #   if in your segment: ' '
    #   if break to other segment: ' X'
    #   else: ''
    results = []
    for seq_id in [0, 1]:
      in_my_seq = tf.equal(word_to_seq_id, seq_id)
      separator_strings = tf.where(
          in_my_seq,
          ' ',
          tf.where(breaks, ' X', '')
      )
      word_strings = tf.where(in_my_seq, words, '')
      all_strings = tf.stack([separator_strings, word_strings], axis=1)
      results.append(tf.strings.substr(
          tf.strings.reduce_join(all_strings), 1, tf.int32.max))
    inputs = tf.stack([tf.strings.join([label, results[0]]),
                       tf.strings.join([label, results[1]])])
    targets = tf.stack([results[1], results[0]])
    return {'inputs': inputs, 'targets': targets}
  dataset = split_text_to_words(dataset, text_key, min_num_words=2)
  return my_fn(dataset).unbatch()


def fill_in_the_blank_sized(
    dataset,
    size_bins=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512),
    text_key='text',
    label='fill: '):
  """Fill in the blank preprocessor that labels blank with a binned size.

  The actual blank size is sampled uniformly from the inclusive range of the min
  and max bin. The blank is then filled in with the closest bin size to the
  actual blank size.

  Args:
    dataset: a tf.data.Dataset, the dataset to preprocess.
    size_bins: a list, a list of blank sizes to select from when labelling the
      blank.
    text_key: a string, the key for the text feature to preprocess in the
      dataset examples.
    label: a string, the label to prepend to the inputs.

  Returns:
    a tf.data.Dataset
  """
  bins = sorted(size_bins)

  @seqio.map_over_dataset(num_seeds=2)
  def my_fn(x, seeds):
    """Apply transformation."""
    words = x['words']
    n_words = tf.size(words)

    blank_size = tf.random.stateless_uniform(
        [],
        minval=bins[0],
        maxval=tf.math.minimum(n_words, bins[-1]),
        dtype=tf.dtypes.int32,
        seed=seeds[0])
    bin_delta = tf.math.abs(bins - blank_size)
    bin_ = tf.gather(bins, tf.argmin(bin_delta))
    blank_start = tf.random.stateless_uniform(
        [],
        minval=0,
        maxval=tf.math.maximum(0, n_words-blank_size) + 1,
        dtype=tf.dtypes.int32,
        seed=seeds[1])

    pre_blank = tf.strings.reduce_join(words[0:blank_start], separator=' ')
    post_blank = tf.strings.reduce_join(
        words[blank_start+blank_size:], separator=' ')
    blank = tf.strings.format('_{}_', bin_)
    # We strip to handle cases where blank is at beginning or end.
    input_ = tf.strings.strip(
        tf.strings.join([pre_blank, blank, post_blank], ' '))
    input_ = tf.strings.join([label, input_])
    target = tf.strings.reduce_join(
        words[blank_start:blank_start+blank_size], separator=' ')
    return {
        'inputs': tf.strings.strip(input_),
        'targets': tf.strings.strip(target)}
  dataset = split_text_to_words(dataset, text_key, min_num_words=2)
  # Filter out examples with fewer words than the minimum.
  dataset = dataset.filter(lambda x: tf.size(x['words']) >= bins[0])
  return my_fn(dataset)


def neighboring_pairs(dataset, text_key='text', reuse_sentences=True):
  """Create a dataset consisting of neighboring sentence pairs.

  The input examples should have a key text_key associated with a tf.string
  value.

  The output examples have keys 'first' and 'second'.

  We only take sentence pairs from within the same line since lines seem to
  represent paragraph-like structures in our text datasets. Empty lines and
  1-sentence lines will thus be ignored.

  The argument reuse_sentences determines whether a sentence can be used as both
  the first and last element in the pair. For example, the input with sentences
  A,B,C,D will return (A,B),(B,C),(C,D) if reuse_sentences is True and
  (A,B),(C,D) if reuse_sentences is False.

  Args:
    dataset: a tf.data.Dataset
    text_key: a string, the key for the text feature to preprocess in the
      dataset examples.
    reuse_sentences: a boolean

  Returns:
    a tf.data.Dataset
  """

  def split_by_lines(dataset):
    """Splits text in dataset by line, removing empty lines."""
    def my_fn(text):
      lines = tf.strings.split([text], sep='\n').values
      return tf.strings.strip(lines)

    dataset = dataset.map(my_fn, num_parallel_calls=AUTOTUNE)
    dataset = dataset.unbatch()
    return dataset.filter(lambda x: tf.strings.length(x) > 0)

  def split_into_pairs(line):
    """Split a given text example into pairs of neighboring sentences."""
    # TODO(mmatena): Use better sentence segmentation.
    sep = str(uuid.uuid4())
    sentences = tf.strings.regex_replace(line, r'((?:\.|\!|\?)+)', r'\1' + sep)
    sentences = tf.strings.strip(tf.strings.split([sentences], sep).values)
    if reuse_sentences:
      firsts = sentences[:-1]
      seconds = sentences[1:]
    else:
      firsts = sentences[:-1:2]
      seconds = sentences[1::2]
    return {
        'first': firsts,
        'second': seconds,
    }

  def example_len(x):
    return tf.math.minimum(
        tf.strings.length(x['first']), tf.strings.length(x['second']))

  # Split by lines.
  dataset = dataset.map(lambda x: x[text_key], num_parallel_calls=AUTOTUNE)
  dataset = split_by_lines(dataset)

  # Get pairs of neighboring sentences.
  dataset = dataset.map(split_into_pairs, num_parallel_calls=AUTOTUNE)
  dataset = dataset.unbatch()

  # Remove examples with empty strings.
  dataset = dataset.filter(lambda x: example_len(x) > 0)
  return dataset


@seqio.map_over_dataset
def glue(x, benchmark_name, label_names, feature_names=None, id_key='idx'):
  """Convert a dataset from glue to text2text examples.

  This function uses the feature names from the dataset to unpack examples into
  a format amenable for a text2text problem. For example, consider the Quora
  Question Pairs (QQP) benchmark, which would suggest
  benchmark_name="qqp"
  label_names=['not_duplicate', 'duplicate']
  For QQP, a typical example might look like
  {
      "question1": "Why do I easily get bored of my friends?",
      "question2": "Why do I get bored of friends so quickly?",
      "label": 1,
      "idx": 10,
  }

  This example would be transformed to
  {
       "inputs": (
           "qqp question1: Why do I easily get bored of my friends? question2: "
           "Why do I get bored of my friends so quickly?"
       ),
       "targets": "duplicate",
      "idx": 10,
  }

  Args:
    x: an example to process.
    benchmark_name: the name of the GLUE benchmark for this dataset.
    label_names: a list of label names corresponding to class index.
    feature_names: an optional ordered list of feature names. If provided,
      features will be ordered in this way in the output. If not provided, all
      features (except 'idx' and 'label') will be used, sorted by name.
    id_key: str, key for id in the dataset. If not provided, 'idx' will be used.
      if None, no id will be added to the dataset.

  Returns:
    A preprocessed example.
  """
  # If an ordering is not provided, sort feature keys to ensure a consistent
  # order.
  feature_keys = (
      feature_names or sorted(set(x.keys()).difference(['label', 'idx'])))
  # Pack keys (formatted as " key: ") and corresponding text feature
  strs_to_join = []
  for key in feature_keys:
    strs_to_join.append('{}:'.format(key))
    strs_to_join.append(x[key])
  # Add benchmark name at the start
  strs_to_join.insert(0, benchmark_name)
  label_name = tf.cond(
      # When no label is provided (label == -1), use "<unk>"
      tf.equal(x['label'], -1),
      lambda: tf.constant('<unk>'),
      # Otherwise grab the label text from label_names
      lambda: tf.gather(label_names, x['label']),
  )
  joined = tf.strings.join(strs_to_join, separator=' ')

  ex = {}

  if benchmark_name == 'multirc':
    # Remove HTML markup.
    joined = tf.strings.regex_replace(joined, '<br>', ' ')
    joined = tf.strings.regex_replace(joined, '<(/)?b>', '')

    # Store the data index in the returned example (used by eval)
    ex['idx/paragraph'] = x['idx']['paragraph']
    ex['idx/question'] = x['idx']['question']
    ex['idx/answer'] = x['idx']['answer']
  else:
    # Store the data index in the returned example (used by eval)
    if id_key:
      ex['idx'] = x[id_key]

  ex['inputs'] = joined
  ex['targets'] = label_name

  return ex


@seqio.map_over_dataset
def stsb(x):
  """Convert STSB examples to text2text format.

  STSB maps two sentences to a floating point number between 1 and 5
  representing their semantic similarity. Since we are treating all tasks as
  text-to-text tasks we need to convert this floating point number to a string.
  The vast majority of the similarity score labels in STSB are in the set
  [0, 0.2, 0.4, ..., 4.8, 5.0]. So, we first round the number to the closest
  entry in this set, and then we convert the result to a string (literally e.g.
  "3.4"). This converts STSB roughly into a 26-class classification dataset.
  This function uses the feature names from the dataset to unpack examples into
  a format amenable for a text2text problem.

  For example, a typical example from STSB might look like
  {
      "sentence1": "Three more US soldiers killed in Afghanistan",
      "sentence2": "NATO Soldier Killed in Afghanistan",
      "label": 1.8,
  }

  This example would be transformed to
  {
       "inputs": (
           "stsb sentence1: Three more US soldiers killed in Afghanistan "
           "sentence2: NATO Soldier Killed in Afghanistan"
       ),
       "targets": "1.8",
  }

  Args:
    x: an example to process.
  Returns:
    A preprocessed example.
  """
  strs_to_join = [
      'stsb sentence1:', x['sentence1'], 'sentence2:', x['sentence2']
  ]
  label_string = tf.as_string(tf.round(x['label'] * 5) / 5, precision=1)
  joined = tf.strings.join(strs_to_join, separator=' ')
  return {'inputs': joined, 'targets': label_string, 'idx': x['idx']}


@seqio.map_over_dataset
def wsc(x):
  """Convert WSC examples to text2text format.

  WSC includes a sentence along with 2 'spans': the first denoting a noun and
  the other a pronoun. The 'label' specifies whether or not the pronoun is
  referencing the noun. This preprocessor puts ' * ' around the noun and ' # '
  around the pronoun.

  For example, a typical example from WSC might look like
  {
      'text': 'This is a test sentence .',
      'span1_text': 'test',
      'span1_index': 3,
      'span2_text': 'This',
      'span2_index': 0,
      'label': 0
  }

  This example would be transformed to
  {
      'inputs': 'wsc text: # This # is a * test * sentence .',
      'targets': 'False'
  }

  Args:
    x: an example to process.
  Returns:
    A preprocessed example.
  """

  def _mark_span(text, span_str, span_idx, mark):
    pattern_tmpl = r'^((?:\S+\s){N})(W)'
    pattern = tf.strings.regex_replace(pattern_tmpl, 'N',
                                       tf.as_string(span_idx))
    pattern = tf.strings.regex_replace(pattern, 'W', span_str)
    return tf.strings.regex_replace(text, pattern, r'\1{0} \2 {0}'.format(mark))

  text = x['text']
  text = _mark_span(text, x['span1_text'], x['span1_index'], '*')
  # Compensate for 2 added "words" added in previous step.
  span2_index = x['span2_index'] + 2 * tf.cast(
      x['span1_index'] < x['span2_index'], tf.int32)
  text = _mark_span(text, x['span2_text'], span2_index, '#')

  # Add benchmark name at the start
  strs_to_join = ['wsc', 'text:', text]
  label_name = tf.cond(
      # When no label is provided (label == -1), use "<unk>"
      tf.equal(x['label'], -1),
      lambda: tf.constant('<unk>'),
      # Otherwise use False/True.
      lambda: tf.gather(['False', 'True'], x['label']))

  joined = tf.strings.join(strs_to_join, separator=' ')
  return {'inputs': joined, 'targets': label_name, 'idx': x['idx']}


@gin.configurable
def record(dataset):
  """Convert ReCoRD examples to text2text examples.

  ReCoRD contains a passage, query containing a '@placeholder' string, and a set
  of entities that are the possible values of the placeholder. Each train and
  validation example will have a list of answers, any of which would be
  considered correct.

  For example, a typical example from ReCoRD might look like
  {
      'passsage': 'This is the passage.',
      'query': 'A @placeholder is a bird.',
      'entities': ['penguin', 'potato', 'pigeon'],
      'answers': ['penguin', 'pigeon'],
  }
  which this preprocessor would turn into the following two examples:
  {
      'inputs': 'record query: A @placeholder is a bird. entities: penguin, '
                'potato, pigeon passage: This is the passage.',
      'targets': 'penguin',
  }
  and
  {
      'inputs': 'record query: A @placeholder is a bird. entities: penguin, '
                'potato, pigeon passage: This is the passage.',
      'targets': 'potato',
  }

  Args:
    dataset: a tf.data.Dataset to process.

  Returns:
    a tf.data.Dataset
  """

  def process_answers(x):
    """Helper fn to get one example per answer."""
    ex = x.copy()
    num_answers = tf.size(ex['answers'])

    def duplicate_along_first_dim(t):
      n_duplicates = tf.math.maximum(num_answers, 1)
      return tf.broadcast_to(
          t, shape=tf.concat([[n_duplicates], tf.shape(t)], axis=0))

    for k, v in x.items():
      if k != 'idx':
        ex[k] = duplicate_along_first_dim(v)
    ex['targets'] = tf.cond(
        tf.greater(num_answers, 0), lambda: x['answers'],
        lambda: tf.constant(['<unk>']))
    ex['idx'] = {
        'passage': duplicate_along_first_dim(x['idx']['passage']),
        'query': duplicate_along_first_dim(x['idx']['query']),
    }

    return ex

  def my_fn(x):
    """Converts the processed example to text2text strings."""
    passage = x['passage']
    passage = tf.strings.regex_replace(passage,
                                       r'(\.|\?|\!|\"|\')\n@highlight\n',
                                       r'\1 ')
    passage = tf.strings.regex_replace(passage, r'\n@highlight\n', '. ')

    strs_to_join = [
        'record query:', x['query'], 'entities:',
        tf.strings.reduce_join(x['entities'], separator=', '), 'passage:',
        passage
    ]
    joined = tf.strings.join(strs_to_join, separator=' ')

    ex = {}

    # Store the data index in the returned example (used by eval)
    ex['idx/passage'] = x['idx']['passage']
    ex['idx/query'] = x['idx']['query']

    ex['inputs'] = joined
    # Note that "answers" has been converted to a single string by the
    # process_answers function.
    ex['targets'] = x['targets']
    # Pass-through full list of answers for eval
    ex['answers'] = x['answers']
    return ex

  dataset = dataset.map(process_answers, num_parallel_calls=AUTOTUNE)
  dataset = dataset.unbatch()
  return dataset.map(my_fn, num_parallel_calls=AUTOTUNE)


def multi_translate(dataset, source_language, target_language):
  """Convert a multi-translate dataset to a text2text pair.

  For example, say the dataset returns examples which have a 'translations'
  feature key so that examples have the following format:
    {
     ...
     'translations': {
         'language': ['de', 'fr', 'en'],
         'translation': ['Das ist gut.', 'Ca c'est bon', 'That is good.']
     },
     ...
    }
  If source_language = 'de', target_language = 'en', then this function will
  return examples of the format:
    {'inputs': 'translate German to English: Das is gut.',
     'targets': 'That is good.'}
  Any other languages present in the dataset will be filtered out.

  Args:
    dataset: a tf.data.Dataset to process.
    source_language: source language code (e.g. 'en') to translate from.
    target_language: target language code (e.g. 'de') to translate to.
  Returns:
    A preprocessed tf.data.Dataset with the format listed above.
  """
  def filter_fn(x):
    langs = x['translations']['language']
    # Test whether both source/target_language appear in the language list
    source_in_langs = tf.reduce_any(tf.equal(source_language, langs))
    target_in_langs = tf.reduce_any(tf.equal(target_language, langs))
    return tf.logical_and(source_in_langs, target_in_langs)
  def map_fn(x):
    langs = x['translations']['language']
    # Retrieve the index in langs where source/target_language appears
    src_idx = tf.squeeze(tf.where(tf.equal(langs, source_language)))
    tgt_idx = tf.squeeze(tf.where(tf.equal(langs, target_language)))
    return {
        source_language: x['translations']['translation'][src_idx],
        target_language: x['translations']['translation'][tgt_idx],
    }
  dataset = dataset.filter(filter_fn)
  dataset = dataset.map(map_fn, num_parallel_calls=AUTOTUNE)
  return translate(dataset, source_language, target_language)


@seqio.map_over_dataset
def definite_pronoun_resolution_simple(x, label='wsc:'):
  """Converts DPR examples to a simple text to text format.

  A typical example from the definite pronoun resolution dataset might look like
  {
     'sentence': 'Bob asked Tom if he can lend some money.',
     'pronoun': 'he',
     'candidates': ['Bob', 'Tom'],
     'label': 1,
  }

  This will be transformed to
  {
     'inputs': 'wsc: Bob asked Tom if *he* can lend some money.'
     'targets': 'Tom',
  }

  Args:
    x: an example to process.
    label: a string, the label to prepend to the inputs.

  Returns:
    A preprocessed example.
  """
  # If there are multiple instances of the pronoun in the sentence, the first
  # one is the one that needs to be resolved.
  inputs = [
      label,
      tf.strings.regex_replace(
          x['sentence'],
          tf.strings.join([r' (', x['pronoun'], r')( |\.|,)']),
          r' *\1*\2',
          replace_global=False,
      ),
  ]
  return {
      'inputs': tf.strings.join(inputs, separator=' '),
      'targets': x['candidates'][x['label']],
  }


def next_sentence_prediction(dataset,
                             text_key='text',
                             reuse_sentences=True,
                             label_sentences=False,
                             p_neighbors=0.5,
                             label='nsp: ',
                             buffer_size=50000):
  """Create a dataset containing a next sentence prediction objective.

  The input examples should have a key text_key associated with a tf.string
  value.

  The output examples have keys 'inputs' and 'targets'.

  EXAMPLE OUTPUTS:

  {
    input: "nsp: sentence1: The man went to the store. sentence2: Penguins are "
           "flightless birds.",
    target: "not_next"
  }

  The "sentence1:" and "sentence2:" labels will be omitted if label_sentences is
  False.

  Args:
    dataset: a tf.data.Dataset
    text_key: a string, the key for the text feature to preprocess in the
      dataset examples.
    reuse_sentences: a boolean, see docs for `neighboring_pairs` for more info.
    label_sentences: a boolean
    p_neighbors: a float between 0 and 1, the probability that a sentence pair
      will be neighbors.
    label: a string, the label to prepend to the inputs.
    buffer_size: an int, the size of the shuffle buffer used to get
      non-neighboring sentences.

  Returns:
    a tf.data.Dataset
  """
  sentence1_label, sentence2_label = '', ''
  if label_sentences:
    sentence1_label, sentence2_label = 'sentence1: ', 'sentence2: '

  empty = tf.constant('', dtype=tf.string, shape=[1])

  dataset = neighboring_pairs(
      dataset, text_key=text_key, reuse_sentences=reuse_sentences)
  dataset = dataset.shuffle(buffer_size).batch(2, drop_remainder=True)

  def some_are_empty(*tensors):
    """See if at least one tensor has shape [0]."""
    empty = [tf.equal(tf.size(t), 0) for t in tensors]
    return tf.reduce_any(empty)

  @seqio.map_over_dataset(num_seeds=1)
  def my_fn(x, seed):
    """Function to be applied to each example in dataset."""
    use_neighbors = (
        tf.random.stateless_uniform(shape=[], seed=seed) < p_neighbors
    )
    firsts, seconds = tf.cond(
        use_neighbors,
        lambda: (x['first'], x['second']),
        lambda: (x['first'], tf.stack([x['second'][1], x['second'][0]])),
    )
    relation_label = tf.cond(
        use_neighbors,
        lambda: 'next',
        lambda: 'not_next',
    )

    inputs = []
    for i in range(2):
      first_inputs = firsts[i]
      second_inputs = seconds[i]

      def create_examples(first_i=first_inputs, second_i=second_inputs):
        return tf.strings.join([
            label,
            sentence1_label,
            first_i,
            ' ',
            sentence2_label,
            second_i,
        ])

      inpt = tf.cond(
          some_are_empty(first_inputs, second_inputs),
          lambda: empty,
          create_examples,
      )
      inputs.append(tf.strings.strip(inpt))
    inputs = tf.reshape(inputs, [-1])
    targets = tf.reshape(2 * [relation_label], [-1])

    return {'inputs': inputs, 'targets': targets}

  dataset = my_fn(dataset).unbatch()

  def example_len(x):
    return tf.math.minimum(
        tf.strings.length(x['inputs']), tf.strings.length(x['targets']))

  # Remove examples with empty strings.
  return dataset.filter(lambda x: example_len(x) > 0)


@seqio.map_over_dataset
def lm(x):
  """Basic language modeling objective for text - empty inputs.

  Given inputs with the format:
  {"text": "Here is some text."}
  This preprocess produces examples with the format
  {"inputs": "", "targets": "Here is some text."}

  Args:
    x: an example to process.

  Returns:
    A preprocessed example.
  """
  return {'inputs': '', 'targets': x['text']}


def _wsc_inputs(x):
  """Given an example from SuperGLUE WSC, compute the 'inputs' value.

  The output will look like a fill in the blank with the pronoun blanked out.
  For example, the text
    'Mitchell asked Tom if he could lend some money.'
  would be transformed to
    'Mitchell asked Tom if X could lend some money.'

  Args:
    x: A dict that is an example from the WSC task of SuperGLUE.

  Returns:
    A scalar string tensor.
  """
  words = tf.strings.split([x['text']], sep=' ').values

  # We would need some special logic to handle the case where the pronoun is the
  # first or last word in the text. None of the examples in WSC seem to have
  # this, so we are ignoring these cases.
  with tf.control_dependencies([
      tf.assert_greater(x['span2_index'], 0),
      tf.assert_less(x['span2_index'], tf.size(words)),
  ]):
    pronoun_index = tf.identity(x['span2_index'])

  def create_input():
    with tf.control_dependencies(
        [tf.assert_equal(words[pronoun_index], x['span2_text'])]):
      return tf.strings.join(
          [
              tf.strings.reduce_join(words[:pronoun_index], separator=' '),
              'X',
              tf.strings.reduce_join(
                  words[pronoun_index + 1:], separator=' '),
          ],
          separator=' ',
      )

  # Handle some special cases.
  if tf.equal(
      x['text'],
      'The boy continued to whip the pony , and eventually the pony threw him over. John laughed out quite loud. \"Good for him,\" he said. '
      ):
    return (
        'The boy continued to whip the pony , and eventually the pony threw '
        'him over. John laughed out quite loud. "Good for X ," he said.'
    )

  # Using the span2_index, we get 'use' instead of 'it'.
  if tf.equal(
      x['text'],
      'When they had eventually calmed down a bit , and had gotten home, Mr. Farley put the magic pebble in an iron safe . Some day they might want to use it , but really for now, what more could they wish for?'
      ):
    return (
        'When they had eventually calmed down a bit , and had gotten home, '
        'Mr. Farley put the magic pebble in an iron safe . Some day they might '
        'want to use X , but really for now, what more could they wish for?'
    )

  return create_input()


def wsc_simple(dataset,
               label='wsc:',
               correct_referent_only=False):
  """Converts SuperGLUE WSC examples to a simple text to text format.

  A typical example from SuperGLUE WSC might look like
  {
    'text': 'Mitchell asked Tom if he could lend some money.',
    'span1_text': 'Tom',
    'span2_text': 'he',
    'span2_index': 4,
  }

  This will be transformed to
  {
     'inputs': 'wsc: Bob asked Tom if *he* can lend some money.'
     'targets': 'Tom',
  }

  The targets will always be the text of the referent regardless of whether it
  is the correct referrent of the pronoun. Thus for training purposes, please
  set `correct_referent_only` to be True.

  Args:
    dataset: a tf.data.Dataset
    label: a string, the label to prepend to the inputs.
    correct_referent_only: a bool, whether to filter out examples for which the
      targets is not the correct referent of the pronoun.

  Returns:
    a tf.data.Dataset
  """

  def map_fn(x):
    """Function to be called for every example in dataset."""
    inputs = [
        label,
        tf.strings.regex_replace(
            _wsc_inputs(x), r' X ', ' *' + x['span2_text'] + '* '),
    ]
    referent = x['span1_text']
    return {
        'inputs': tf.strings.join(inputs, separator=' '),
        # The reshape is necessary as otherwise the tensor has unknown rank.
        'targets': tf.reshape(referent, shape=[]),
        'label': x.get('label', 0),
        'idx': x['idx'],
    }

  if correct_referent_only:
    dataset = dataset.filter(lambda x: tf.cast(x.get('label', False), tf.bool))

  return dataset.map(map_fn, num_parallel_calls=AUTOTUNE)


@seqio.map_over_dataset
def wnli_simple(x, label='wsc:'):
  """Converts GLUE WNLI examples to a simple text to text format.

  A typical example from WNLI might look like:
  {
    'sentence1': 'The fish ate the worm. It was tasty.',
    'sentence2': 'The worm was tasty.',
    'label': 1,
  }

  This will be transformed to:
  {
    'inputs': 'wsc: The fish ate the worm. *It* was tasty.',
    'targets': 'The worm',
    'premise': 'The fish ate the worm. It was tasty.,
    'hypothesis': 'The worm was tasty.',
    'label': 1,
  }

  This preprocessor has been manually verified to produce reasonable WSC
  examples for the dev and test sets. Tasks using this preprocessor should only
  be used eval and not train.

  Args:
    x: an example to process.
    label: a string, the label to prepend to the inputs.

  Returns:
    A preprocessed example.
  """
  pronouns = ['he', 'she', 'they', 'it', 'her', 'his', 'their', 'them', 'him']
  PronounMatch = collections.namedtuple(  # pylint: disable=invalid-name
      'PronounMatch', ['score', 'index_in_premise', 'candidate'])

  def split_clean(s):
    """Returns array of words with punctuation and capitalization removed."""
    words = [
        re.sub(r'(\.|,|\?|\!)$', '', w) for w in s.strip().lower().split(' ')
    ]
    return [w for w in words if w]

  def get_all_pronoun_indices(s):
    return [i for i, w in enumerate(s) if w in pronouns]

  def get_post_match_size(hypothesis, words):
    """Returns len of largest prefix of words that is substr of hypothesis."""
    hypothesis = ' '.join(hypothesis)
    for i in range(len(words)):
      if ' '.join(words[:i + 1]) not in hypothesis:
        return i
    return len(words)

  def get_pre_match_size(hypothesis, words):
    """Returns len of largest suffix of words that is substr of hypothesis."""
    return get_post_match_size(hypothesis[::-1], words[::-1])

  def get_pronoun_match(premise, hypothesis, index):
    """Return the PronounMatch for the pronoun at `index` in premise."""
    pre, post = premise[:index], premise[index + 1:]

    pre_match_size = get_pre_match_size(hypothesis, pre)
    post_match_size = get_post_match_size(hypothesis, post)
    score = pre_match_size + post_match_size

    candidate = ''
    if score:
      pre_match = pre[-pre_match_size or len(pre):]
      post_match = post[:post_match_size]
      m = re.search(' '.join(pre_match + [r'(.+)'] + post_match),
                    ' '.join(hypothesis))
      if not m:
        # Handle cases where the candidate is at the start of the hypthesis.
        m = re.search(' '.join([r'^(.+)'] + post_match), ' '.join(hypothesis))
      if not m:
        # Handle cases where the candidate is at the end of the hypthesis.
        m = re.search(' '.join(pre_match + [r'(.+)$']), ' '.join(hypothesis))

      if m:
        candidate = m.group(1)

    return PronounMatch(
        score=score, index_in_premise=index, candidate=candidate)

  def get_best_pronoun_match(premise, hypothesis):
    """Returns the match for the pronoun in the premise to disambiguate."""
    pronoun_indices = get_all_pronoun_indices(premise)
    scoredpronouns = [
        get_pronoun_match(premise, hypothesis, index)
        for index in pronoun_indices
    ]
    return max(scoredpronouns, key=lambda x: x.score)

  def highlight(sentence, index):
    words = sentence.split(' ')
    word = words[index]
    if word[-1] in ['.', ',', '!', '?']:
      highlighted = '*{}* {}'.format(word[:-1], word[-1])
    else:
      highlighted = '*{}*'.format(word)
    return ' '.join(words[:index] + [highlighted] + words[index + 1:])

  def make_nonpossessive(word):
    # WSC simple targets will never be possessive, even when the pronoun is
    # possesive.
    if word.endswith("'"):
      return word[:-1]
    elif word.endswith("'s"):
      return word[:-2]
    else:
      return word

  def clean_up(candidate):
    words = candidate.split(' ')
    # Sometimes the candidate extraction messes up, and the candidate will start
    # with the start of the hypothesis and extend to the correct candidate. We
    # can try to clean up the candidate in some cases by removing everything up
    # to the last article in the sentence.
    article_index = max(
        [words.index(art) for art in {'a', 'an', 'the'} if art in words] or [0])
    return ' '.join(words[article_index:])

  def process_candidate(candidate, hypothesis):
    """Handles special cases and adds proper punctuation/capitalization."""
    candidate = clean_up(candidate)

    pattern = '({})'.format(' '.join([
        r'{}(?:\.|,|\?|\!)?'.format(re.escape(c)) for c in candidate.split(' ')
    ]))
    m = re.search(pattern, hypothesis, re.IGNORECASE)
    if not m:
      raise ValueError(
          'Unable to find candidate "{}" in hypothesis "{}".'.format(
              candidate, hypothesis))

    candidate = m.group(1)
    if candidate and candidate[-1] in ['.', ',', '!', '?']:
      candidate = candidate[:-1]
    return make_nonpossessive(candidate)

  def compute_inputs_and_targets(premise, hypothesis):
    """Compute inputs and targets for WNLI simple."""
    premise = tf.compat.as_text(premise.numpy())
    hypothesis = tf.compat.as_text(hypothesis.numpy())

    match = get_best_pronoun_match(
        split_clean(premise), split_clean(hypothesis))
    targets = process_candidate(match.candidate, hypothesis)
    inputs = '{} {}'.format(label, highlight(premise, match.index_in_premise))
    return inputs, targets

  inputs, targets = tf.py_function(
      compute_inputs_and_targets,
      inp=[x['sentence1'], x['sentence2']],
      Tout=[tf.string, tf.string])
  return {
      # The reshape is necessary as otherwise the tensor has unknown rank.
      'inputs': tf.reshape(inputs, shape=[]),
      'targets': tf.reshape(targets, shape=[]),
      'premise': x['sentence1'],
      'hypothesis': x['sentence2'],
      'label': x.get('label', 0),
      'idx': x['idx'],
  }


def rank_classification(
    ds: tf.data.Dataset,
    inputs_fn: Callable[[FeatureType], tf.Tensor],
    targets_fn: Callable[[FeatureType], tf.Tensor],
    is_correct_fn: Callable[[FeatureType], tf.Tensor],
    weight_fn: Optional[Callable[[FeatureType], tf.Tensor]] = None,
    mode: str = 'eval',
    passthrough_feature_keys: Optional[Sequence[str]] = None,
) -> tf.data.Dataset:
  """Prepare dataset for rank classification scoring.

  Intended to be used with `rank_classification` postprocessor and metric.

  `inputs_fn` and `targets_fn` must return the 'inputs' and 'targets' features,
  respectively, for each possible class label given the raw example features.
  'is_correct_fn' must return the 'is_correct' feature, a boolean for whether
  each label is matching with the ground truth target before the examples are
  expanded.

  In 'train' mode, only the inputs / targets marked correct will be produced.
  In 'eval' mode, all inputs / targets will be produced.
  In 'fewshot_eval', all inputs / targets will be produced as a single batch.

  Each output example will also be given a unique 'idx' feature. The first dim
  is a sequential index for the input example and the second is the index of the
  generated output for it. E.g., the second output example from the fourth input
  example would be `[3, 1]`.

  To be clear, consider the following arguments:

  inputs_fn=lambda ex: ex['prefix'],
  targets_fn=lambda ex: ex['suffix'],
  is_correct_fn=lambda ex: tf.one_hot(ex['label'], num_classes)
  weight_fn=lambda ex: ex['weight']

  Given the following example:

  {
    'prefix': ['The farmland needed ', 'The farmland wanted '],
    'suffix': ['water', 'cows'],
    'label': 0,
    'weight': 1.0,
  }
  the preprocessor would return:

  [{
      'idx': [0, 0],
      'inputs': 'The farmland needed ',
      'targets': 'water',
      'is_correct': True,
      'weight': 1.0
   },
   {
     'idx': [0, 1],
     'inputs': 'The farmland wanted ',
     'targets': 'cows',
     'is_correct': False,
     'weight': 1.0
   }]

  With mode set to 'train', it would return only the first example,
  since it uses the correct label. With mode set to 'fewshot_eval', it would
  return both examples in a single batch.

  Args:
    ds: a tf.data.Dataset to preprocess.
    inputs_fn: a callable that returns the 'inputs' features for each label
      given the input example.
    targets_fn: a callable that returns the 'targets' features for each label
      given the input example.
    is_correct_fn: a callable that returns the 'label' feature. May be an int32
      scalar or 1-D Tensor.
    weight_fn: a callable that returns the 'weight' feature (float32 scalar).
    mode: A string, one of 'train' or'eval 'train' produces only the correct
      example(s) based on the label value(s). 'eval' produces an example for
      every possible class value, sequentially. 'fewshot_eval' produces an
      example for every possible class value, batched together for each input
      example.
    passthrough_feature_keys: a sequence of feature names that should be passed
      through to the output of this preprocessor. eg: ["starburst", "tokens"]

  Returns:
    A tf.data.Dataset containing 'idx', inputs', 'targets', and 'is_correct'.
  """
  if mode not in ('train', 'eval', 'fewshot_eval'):
    raise ValueError(
        "Mode must be one of 'train', 'eval', or 'fewshot_eval'. "
        f"Got '{mode}'.")

  def make_examples(idx, ex):
    inputs = inputs_fn(ex)
    targets = targets_fn(ex)
    is_correct = tf.cast(is_correct_fn(ex), tf.bool)

    tf.debugging.assert_equal(
        tf.size(is_correct), [tf.size(inputs), tf.size(targets)],
        '`inputs_fn`, `targets_fn`, and `is_correct_fn` must return the same '
        'size tensors.')

    num_out = tf.size(is_correct)
    in_idx = tf.fill([num_out], tf.cast(idx, tf.int32))
    out_idx = tf.range(num_out)

    output = {
        'idx': tf.stack([in_idx, out_idx], 1),
        'inputs': inputs,
        'targets': targets,
        'is_correct': is_correct,
    }

    if passthrough_feature_keys is not None:
      for feature_name in passthrough_feature_keys:
        tiled_shape = tf.concat(
            [tf.expand_dims(tf.shape(targets)[0], axis=0),
             tf.ones(len(ex[feature_name].shape), dtype=tf.int32)],
            axis=0)
        output[feature_name] = tf.tile(
            tf.expand_dims(ex[feature_name], axis=0),
            tiled_shape)

    if weight_fn is not None:
      output['weight'] = tf.fill(tf.shape(is_correct), weight_fn(ex))
      output['weight'] = tf.cast(output['weight'], tf.float32)

    return output

  ds = ds.enumerate()
  ds = ds.map(make_examples, num_parallel_calls=AUTOTUNE)
  if mode != 'fewshot_eval':
    ds = ds.unbatch()
  if mode == 'train':
    ds = ds.filter(lambda ex: ex['is_correct'])
  return ds


def rank_classification_formatter(
    ds: tf.data.Dataset,
    inputs_formats: Union[str, Sequence[str]],
    targets_formats: Union[str, Sequence[str]],
    mode: str = 'eval',
    label_key: str = 'label',
    weight_key: Optional[str] = None) -> tf.data.Dataset:
  """Create 'inputs' and 'targets' strings for ranking classification.

  Intended to be used with `rank_classification` postprocessor and metric.

  Inputs will be formatted by filling in the feature values in the
  `inputs_formats` and `targets_formats` strings.

  Nested features can be accessed by concatenating the features using forward
  slash. For eg: if sub-sub-key is nested under sub-key, which is nested under
  key, then sub-sub-key can be accessed using key/sub-key/sub-sub-key.

  In 'eval' mode, a separate example will be produced for each targets / inputs
  format string. These can then be scored to find the one with the highest
  likelihood. The `rank_classification` postprocessor and metric allow you to
  evaluate with this technique.

  In 'train' mode, only the targets / inputs format string indexed by the
  label(s) will be produced. In 'eval' mode, all inputs / targets will be
  produced.

  Each input example will also be given a unique, sequential index called 'idx'.

  For example, with arguments:

  ```
  inputs_format='{premise} What is the {question}? X',
  targets_formats=[
    'I think {choice1}.',
    'I think {choice2}.'
  ],
  mode='eval'
  ```

  given the input:

  {
    'premise': 'The farmland needed irrigation.',
    'question': 'effect',
    'choice1' : 'a canal was constructed',
    'choice2': 'the crops grew tall',
    'label': 0,
  }

  the preprocessor would return:
  [{
     'idx': 0,
     'inputs': 'The farmland needed irrigation. What is the effect? X',
     'targets': 'I think a canal was constructed.',
     'is_correct': True
   },
   {
     'idx': 0,
     'inputs': 'The farmland needed irrigation. What is the effect? X',
     'targets': 'I think the crops grew tall.',
     'is_correct': False
   }]

  With `mode='train'`, it would return only the first example,
  since it uses the correct label.

  With `mode='fewshot_eval'`, it would return both examples in a single batch.

  Args:
    ds: a tf.data.Dataset to preprocess.
    inputs_formats: A string or a list of strings to format with feature values
      to produce 'inputs'. Feature keys should be surrounded by curly braces to
      be replaced.
    targets_formats: A string or a list of strings to format with feature values
      to produce 'targets', one for each possible class value. Feature keys
      should be surrounded by curly braces to be replaced.
    mode: A string, one of 'train', 'eval', or 'fewshot_train') 'train' produces
      only the correct example(s) based on the label value(s). 'eval' produces
      an example for every possible class value, sequentially.
      'fewshot_eval': produces an example for every possible class value,
        batched together for each input example.
    label_key: A string, the feature key for the integer label value(s).
    weight_key: A string, the feature key for the float example weight.

  Returns:
    A tf.data.Dataset containing 'idx', inputs', 'targets', and 'is_correct'.
  """
  if (isinstance(inputs_formats, (list, tuple)) and
      isinstance(targets_formats, (list, tuple))):
    if len(inputs_formats) != len(targets_formats):
      raise ValueError(
          f'The inputs_formats ({len(inputs_formats)}) and '
          f'targets_formats ({len(targets_formats)}) are both instances '
          'of list or tuple, but do not have matching lengths.')
  elif isinstance(inputs_formats, (list, tuple)):
    num_classes = len(inputs_formats)
    targets_formats = [targets_formats] * num_classes
  elif isinstance(targets_formats, (list, tuple)):
    num_classes = len(targets_formats)
    inputs_formats = [inputs_formats] * num_classes
  else:
    raise ValueError(
        'One of the inputs_formats and targets_formats has to '
        f'be a list or tuple, inputs_formats: {inputs_formats}, '
        f'target_formats: {targets_formats}.')

  def _format_str(features, fmt):
    keys = set(re.findall(r'{(\S+)}', fmt))
    s = fmt
    for k in keys:
      value = features
      for subkey in k.split('/'):
        value = value[subkey]
      if not isinstance(value, tf.Tensor):
        raise ValueError(
            f'Final value of key \'{k}\' must be a tf.string. '
            f'Got: {type(value).__name__}')
      tf.debugging.assert_type(
          value, tf.string,
          f'Final value of key \'{k}\' must be a tf.string. '
          f'Got: {value.dtype.name}')
      s = tf.strings.regex_replace(s, '{%s}' % k, value)
    return s

  def _apply_formats(features, fmts):
    return [_format_str(features, fmt) for fmt in fmts]

  def _is_correct_fn(ex):
    labels = ex[label_key]
    is_correct = tf.one_hot(labels, num_classes, on_value=True, off_value=False)
    if labels.shape.rank:
      is_correct = tf.math.reduce_any(is_correct, axis=0)
    return is_correct

  def _weight_fn(ex):
    return ex[weight_key]

  return rank_classification(
      ds,
      inputs_fn=functools.partial(_apply_formats, fmts=inputs_formats),
      targets_fn=functools.partial(_apply_formats, fmts=targets_formats),
      is_correct_fn=_is_correct_fn,
      weight_fn=None if weight_key is None else _weight_fn,
      mode=mode)


@seqio.map_over_dataset
def parse_tsv(line, field_names=None, field_delim='\t', field_columns=None):
  """Splits TSV lines into dict examples mapping field name to string value.

  Args:
    line: an example containing a comma/tab-delimited string.
    field_names: a list of strings, the ordered names of the TSV fields.
      Defaults to "inputs" and "targets".
    field_delim: a string, the delimiter to split on e.g. ',' for csv.
    field_columns: a list of column indices for each field.
      Defaults to consecutive numbering of the provided `field_names`.

  Returns:
    A feature dict mapping field name to string value.
  """
  field_names = field_names or ['inputs', 'targets']
  field_columns = field_columns or list(range(len(field_names)))
  return dict(
      zip(field_names,
          tf.io.decode_csv(
              line,
              record_defaults=[''] * len(field_names),
              field_delim=field_delim,
              use_quote_delim=False,
              select_cols=field_columns)))


@seqio.map_over_dataset
def preprocess_tsv(line,
                   field_delim='\t',
                   num_fields=2,
                   inputs_format='{0}',
                   targets_format='{1}',
                   field_names=None,
                   use_quote_delim=False):
  r"""Parse tab-delimited strings into inputs and targets.

  This function takes a tf.data.Dataset of strings, each of which contains
  tab-delimited fields.  The function returns a tf.data.Dataset of feature
  dictionaries of the form {"inputs": string, "targets": string}.

  inputs_format contains a template string and field numbers or names used to
  produce the "inputs" string.
  targets_format contains a template string and field numbers or names used to
  produce the "targets" string.

  Example (field numbers):
    The input dataset contains the lines:
    "6,7,42"
    "2,9,18"
    preprocess_tsv(dataset,
                   field_delim=',',
                   inputs_format='numerator: {2} denominator: {1}',
                   targets_format='quotient: {0}'
    would produce a dataset containing the dictionaries:
    {"inputs": "numerator: 42 denomnator: 7", "targets": "quotient: 6"}
    {"inputs": "numerator: 18 denomnator: 9", "targets": "quotient: 2"}

  Example (field names):
    The input dataset contains the lines:
    "6,7,42"
    "2,9,18"
    preprocess_tsv(dataset,
                   field_delim=',',
                   field_names=['quot', 'denom', 'numer'],
                   inputs_format='numerator: {numer} denominator: {denom}',
                   targets_format='quotient: {quot}'
    would produce a dataset containing the dictionaries:
    {"inputs": "numerator: 42 denominator: 7", "targets": "quotient: 6"}
    {"inputs": "numerator: 18 denominator: 9", "targets": "quotient: 2"}

  Args:
    line: an example containing comma/tab-delimited string.
    field_delim: a string, the delimiter to split on e.g. ',' for csv.
    num_fields: an integer
    inputs_format: a string, the desired output format with placeholders for
      field values.
    targets_format: a string, the desired output format with placeholders for
      field values.
    field_names: a list of strings, the ordered names of the TSV fields.
      defaults to None (i.e. use field number in *_format)
    use_quote_delim: If false, treats double quotation marks as regular
      characters inside of the string fields (ignoring RFC 4180, Section 2,
      Bullet 5).

  Returns:
    A feature dict with 'inputs' and 'targets' features.
  """
  def _format_part_with_field_numbers(part, field_values):
    found = re.findall(r'{(\d+)}', part)
    if found:
      return field_values[int(found[0])]
    else:
      return part

  def _format_part_with_field_names(part, field_names, field_values):
    field_names_re = '|'.join(['{{({})}}'.format(x) for x in field_names])
    found = re.findall(field_names_re, part)
    if found:
      pos = field_names.index(''.join(found[0]))
      return field_values[int(pos)]
    else:
      return part

  def _format(format_string, field_names, field_values):
    if field_names is None:
      parts = [
          _format_part_with_field_numbers(p, field_values)
          for p in re.split(r'({\d+})', format_string)
      ]
    else:
      field_names_re = '(' + '|'.join(['{{{}}}'.format(x) for x in field_names
                                      ]) + ')'
      parts = [
          _format_part_with_field_names(p, field_names, field_values)
          for p in re.split(field_names_re, format_string)
      ]
    return tf.strings.join(parts)

  field_values = tf.io.decode_csv(
      line,
      record_defaults=[''] *
      (num_fields if field_names is None else len(field_names)),
      field_delim=field_delim,
      use_quote_delim=use_quote_delim)
  return {
      'inputs': _format(inputs_format, field_names, field_values),
      'targets': _format(targets_format, field_names, field_values)
  }


# ======================Token Preprocessors=====================================


def span_corruption(dataset,
                    sequence_length,
                    output_features,
                    mean_noise_span_length=3.0,
                    noise_density=0.15,
                    input_feature_key='inputs',
                    merge_examples_to_reduce_padding=True,
                    reserved_for_packing=None,
                    passthrough_feature_keys: Optional[Sequence[str]] = None):
  """Final pretraining objective used in Raffel et al., 2019.

  Args:
    dataset: A tf.data.Dataset with dictionaries containing the key
      `input_feature_key`.
    sequence_length: dict mapping of feature key to int length for that feature.
    output_features: mapping of keys to features.
    mean_noise_span_length: the mean number of tokens per masked span per
      example.
    noise_density: what fraction of the tokens to mask.
    input_feature_key: which feature to use from the dataset as the input text
      tokens.
    merge_examples_to_reduce_padding: if True, combines multiple input examples
      to reduce padding.
    reserved_for_packing: if specified, reduces the desired inputs length by the
      specified amount to enable multiple examples to be packed together
      downstream.
    passthrough_feature_keys: a sequence of feature names that should be passed
      through to the output of this preprocessor. eg: ["tokens"]. Only
      supported if `merge_examples_to_reduce_padding` is set to False.

  Returns:
    a dataset
  """
  inputs_length = sequence_length[input_feature_key]
  if reserved_for_packing:
    inputs_length -= reserved_for_packing

  input_length, targets_length = random_spans_helper(
      extra_tokens_per_span_inputs=1,
      extra_tokens_per_span_targets=1,
      inputs_length=inputs_length,
      mean_noise_span_length=mean_noise_span_length,
      noise_density=noise_density)

  if sequence_length['targets'] < targets_length:
    raise ValueError(
        f'Expected targets length for span corruption ({targets_length}) is '
        f'greater than configured targets length '
        f"({sequence_length['targets']})")

  ds = dataset
  ds = select_random_chunk(
      ds,
      output_features=output_features,
      feature_key='targets',
      max_length=65536,
      passthrough_feature_keys=passthrough_feature_keys)
  if merge_examples_to_reduce_padding:
    if passthrough_feature_keys:
      raise ValueError('passthrough_feature_keys not supported with '
                       'merge_examples_to_reduce_padding=True. '
                       f'Got: {passthrough_feature_keys}')
    ds = reduce_concat_tokens(ds, feature_key='targets', batch_size=128)
  ds = split_tokens(
      ds,
      feature_key='targets',
      min_tokens_per_segment=None,
      max_tokens_per_segment=input_length,
      passthrough_feature_keys=passthrough_feature_keys)
  ds = denoise(
      ds,
      output_features,
      inputs_fn=noise_span_to_unique_sentinel,
      targets_fn=nonnoise_span_to_unique_sentinel,
      noise_density=noise_density,
      noise_mask_fn=functools.partial(
          random_spans_noise_mask,
          mean_noise_span_length=mean_noise_span_length),
      input_feature_key=input_feature_key,
      passthrough_feature_keys=passthrough_feature_keys)
  return ds


# TODO(adarob): Add a test.
def iid_denoising(dataset, sequence_length, output_features):
  """Baseline pretraining objective used in Raffel et al., 2019."""
  ds = dataset
  ds = select_random_chunk(ds, output_features=output_features,
                           feature_key='targets', max_length=65536)
  ds = reduce_concat_tokens(ds, feature_key='targets', batch_size=128)
  ds = split_tokens_to_inputs_length(ds, output_features=output_features,
                                     sequence_length=sequence_length)
  ds = denoise(
      ds,
      output_features,
      inputs_fn=noise_span_to_unique_sentinel,
      targets_fn=nonnoise_span_to_unique_sentinel,
      noise_density=0.15,
      noise_mask_fn=iid_noise_mask
  )
  return ds


def prefix_lm(dataset, sequence_length, output_features):
  """Prefix language modeling objective used in Raffel et al. 2019."""
  ds = dataset
  ds = select_random_chunk(ds, output_features=output_features,
                           feature_key='targets', max_length=65536)
  ds = split_tokens_to_inputs_length(ds, output_features=output_features,
                                     sequence_length=sequence_length)
  ds = denoise(
      ds,
      output_features,
      inputs_fn=drop_nonnoise_tokens,
      targets_fn=drop_noise_tokens,
      noise_density=0.5,
      noise_mask_fn=random_prefix_noise_mask,
  )
  return ds


def full_lm(dataset, sequence_length, output_features):
  """Full language modeling objective with EOS only at document boundaries."""
  ds = dataset
  ds = select_random_chunk(ds, output_features=output_features,
                           feature_key='targets', max_length=65536)
  ds = seqio.preprocessors.append_eos(ds, output_features)
  ds = reduce_concat_tokens(ds, feature_key='targets', batch_size=128)
  # Don't use `split_tokens_to_targets_length` since we've alrady added EOS.
  ds = split_tokens(ds, max_tokens_per_segment=sequence_length['targets'])
  return ds


def single_example_select_random_chunk(
    features: FeatureType,
    seed: tf.Tensor,
    output_features: Mapping[str, seqio.Feature],
    max_length: Optional[int] = None,
    feature_key: str = 'targets',
    additional_feature_keys: Optional[Sequence[str]] = None,
    passthrough_feature_keys: Optional[Sequence[str]] = None,
    sequence_length: Optional[Mapping[str, int]] = None,
    uniform_random_start: bool = False,
    min_length: Optional[int] = None) -> FeatureType:
  """Token-preprocessor to extract one span of at most `max_length` tokens.

  If the token sequence is longer than `max_length`, then we return a random
  subsequence.  Otherwise, we return the full sequence.

  This is generally followed by split_tokens.

  Args:
    features: Single example with `feature_key` containing a tokenized sequence.
    seed: Random seed to be used.
    output_features: Mapping of keys to features.
    max_length: Typically specified in gin configs, takes priority over
      sequence_length.
    feature_key: Which feature to use from the dataset.
    additional_feature_keys: Additional features to use. The same chunk will be
      selected from these features as from the one specified in feature_key,
      so they should all have the same length.
    passthrough_feature_keys: Additional keys to pass through unchanged.
    sequence_length: Used if max_length is not specified. Typically passed in
      by the data pipeline. feature_key will be used to select the length.
    uniform_random_start: If True, will select a starting point in
      [-max_length + 1, n_tokens). If False, will select one of a set of chunks
      offset by max_length. Both of these starting points try to ensure each
      token has an equal probability of being included.
    min_length: If specified, lengths of chunks will be selected uniformly at
      random from [min_length, max_length]. Note that chunks can end up shorter
      than min_length if at the beginning or end of the sequence.

  Returns:
    The features of the selected chunk.
  """
  if passthrough_feature_keys:
    chunk_keys = set([feature_key] + (additional_feature_keys or []))
    overlap_keys = chunk_keys & set(passthrough_feature_keys)
    if overlap_keys:
      raise ValueError(
          f'chunk keys {overlap_keys} also included in passthrough keys')

  if max_length is None and sequence_length is not None:
    max_length = sequence_length[feature_key]
    if output_features[feature_key].add_eos:
      # Leave room to insert an EOS token.
      max_length -= 1
  if max_length is None:
    raise ValueError('Must specify max_length or sequence_length.')

  seeds = tf.unstack(tf.random.experimental.stateless_split(seed))
  tokens = features[feature_key]
  n_tokens = tf.shape(tokens)[0]
  if min_length is not None:
    length = tf.random.stateless_uniform([],
                                         minval=min_length,
                                         maxval=max_length,
                                         dtype=tf.int32,
                                         seed=seeds[0])
  else:
    length = max_length
  if uniform_random_start:
    start = tf.random.stateless_uniform(
        [],
        minval=-length + 1,  # pylint:disable=invalid-unary-operand-type
        maxval=n_tokens,
        dtype=tf.int32,
        seed=seeds[1])
    end = tf.minimum(start + length, n_tokens)
    start = tf.maximum(start, 0)
  else:
    num_segments = tf.cast(
        tf.math.ceil(
            tf.cast(n_tokens, tf.float32) / tf.cast(length, tf.float32)),
        tf.int32)
    start = length * tf.random.stateless_uniform(
        [], maxval=num_segments, dtype=tf.int32, seed=seeds[1])
    end = tf.minimum(start + length, n_tokens)
  chunk = {feature_key: tokens[start:end]}
  if additional_feature_keys is not None:
    for k in additional_feature_keys:
      with tf.control_dependencies([
          tf.assert_equal(
              tf.shape(tokens)[0],
              tf.shape(features[k])[0],
              message=(f'Additional feature {k} is not the same size as '
                       f'{feature_key} along axis 0 in select_random_chunk().'))
      ]):
        chunk[k] = features[k][start:end]
  if passthrough_feature_keys is not None:
    for k in passthrough_feature_keys:
      chunk[k] = features[k]
  return chunk


@gin.configurable
def select_random_chunk(dataset: tf.data.Dataset,
                        output_features: Mapping[str, seqio.Feature],
                        max_length: Optional[int] = None,
                        feature_key: str = 'targets',
                        additional_feature_keys: Optional[Sequence[str]] = None,
                        passthrough_feature_keys: Optional[
                            Sequence[str]] = None,
                        sequence_length: Optional[Mapping[str, int]] = None,
                        uniform_random_start: bool = False,
                        min_length: Optional[int] = None,
                        **unused_kwargs) -> tf.data.Dataset:
  """SeqIO wrapper for single_example_select_random_chunk()."""

  @seqio.map_over_dataset(num_seeds=1)
  def _my_fn(x, seed):
    return single_example_select_random_chunk(
        x,
        seed,
        output_features=output_features,
        max_length=max_length,
        feature_key=feature_key,
        additional_feature_keys=additional_feature_keys,
        passthrough_feature_keys=passthrough_feature_keys,
        sequence_length=sequence_length,
        uniform_random_start=uniform_random_start,
        min_length=min_length)

  # Filter empty examples.
  dataset = dataset.filter(lambda x: tf.not_equal(tf.size(x[feature_key]), 0))
  return _my_fn(dataset)


@gin.configurable
def reduce_concat_tokens(dataset,
                         feature_key='targets',
                         batch_size=128,
                         **unused_kwargs):
  """Token-preprocessor to concatenate multiple unrelated documents.

  If we want to generate examples of exactly the right length,
  (to avoid wasting space on padding), then we use this function, folowed by
  split_tokens.

  Args:
    dataset: a tf.data.Dataset with dictionaries containing the key feature_key.
    feature_key: an string
    batch_size: an integer - how many documents to concatenate into one

  Returns:
    a dataset
  """
  dataset = dataset.map(
      lambda x: {feature_key: x[feature_key]}, num_parallel_calls=AUTOTUNE)
  dataset = dataset.padded_batch(batch_size, padded_shapes={feature_key: [-1]})
  def _my_fn(x):
    tokens = tf.reshape(x[feature_key], [-1])
    # strip padding
    tokens = tf.boolean_mask(tokens, tf.cast(tokens, tf.bool))
    return {feature_key: tokens}

  return dataset.map(_my_fn, num_parallel_calls=AUTOTUNE)


@seqio.map_over_dataset
def trim_tokens_at_front(x,
                         sequence_length,
                         keys_to_trim=None,
                         **unused_kwargs):
  """Token-preprocessor to trim sequence at the beginning.

  Args:
    x: an example with dictionaries containing keys_to_trim.
    sequence_length: a dict of ints.
    keys_to_trim: a list of feature keys.

  Returns:
    A preprocessed example.
  """

  for key in (keys_to_trim or sequence_length.keys()):
    if key in x:
      # trim tokens, leaving room for EOS which gets added later
      x[key] = x[key][-(sequence_length[key] - 1):]
  return x


def trivia_qa_truncate_inputs(dataset, output_features, sequence_length):
  """Token preprocessor for the trivia QA dataset to truncate inputs.

  This function takes a dataset containing "targets" and "inputs". It searches
  for the "targets" in the "inputs" and truncates the "inputs" to
  `sequence_length` while ensuring that the "targets" are present in the
  "inputs". The function will randomly select a subset of "inputs".
  If "targets" are not found in the "inputs", then the example is
  is dropped from the dataset.

  E.g.
  Input dataset
  {
    "inputs": [0, 3, 5, 7, 9, 11, 13, 15, 17, 18]
    "targets": [5, 7, 9]
  }

  Output dataset (assuming sequence_length['inputs'] = 4)
  {
    "inputs": [3, 5, 7, 9]
    "targets": [5, 7, 9]
  }

  or

  {
     "inputs": [5, 7, 9, 11]
     "targets": [5, 7, 9]
  }
  Args:
    dataset: a tf.data.Dataset with dictionaries containing the "inputs" and
      "targets".
    output_features: unused by this function.
    sequence_length: a dict, with keys as "inputs" and "targets" indicating the
      maximum number of tokens in each of the sequences.

  Returns:
    a dataset

  """

  del output_features

  @seqio.map_over_dataset(num_seeds=1)
  def my_fn(features, seed):
    """Function to map original dataset to the new dataset."""
    inputs = features['inputs']
    targets = features['targets']
    ans_len = tf.shape(targets)[0]
    max_input_tokens = sequence_length['inputs']

    def truncate_inputs():
      """Helper function to truncate the inputs."""

      def answer_in_context(context, answer):
        """Helper function that checks if the answer is present in the context.

        Args:
          context: Tensor, tokenized representation of the context
          answer: Tensor, tokenized representation of the answer

        Returns:
          result: boolean, indicates if the answer was present in the context.
          pos_mask: boolean mask, a mask for every possible start position of
            the answer in the context. Indicates whether the answer starts at
            the particular position.
        """
        conv_inp = tf.reshape(tf.cast(context, tf.float32), [1, -1, 1])
        ans_len = tf.shape(answer)[0]
        filters = tf.eye(ans_len, dtype=tf.float32)

        # Assume context len is N and answer len is M.
        # Use a convolution to create a matrix of (N-M) x M elements where
        # each row of the matrix is a sequence of len M. This matrix contains
        # all possible contiguous sequences of length M from the context.
        # Every row of this matrix is compared with the answer to check if the
        # answer exists in the context.
        strided = tf.nn.conv1d(conv_inp,
                               tf.reshape(filters, [ans_len, 1, ans_len]), 1,
                               'VALID')
        strided = tf.cast(strided[0], answer.dtype)
        pos_mask = tf.reduce_all(
            tf.equal(strided, tf.reshape(answer, [1, -1])), 1)
        result = tf.reduce_any(pos_mask)
        return result, pos_mask

      def slice_inputs(inputs, answer_len, pos_mask, seed=None):
        """Helper function to slice inputs while keeping the answer."""
        ans_start_pos = tf.cast(tf.where(pos_mask)[0][0], tf.int32)
        inputs_len = tf.shape(inputs)[0]
        start_range_min = tf.maximum(
            0, ans_start_pos - (max_input_tokens - answer_len))
        start_range_max = tf.minimum(ans_start_pos,
                                     inputs_len - max_input_tokens) + 1

        start_pos = tf.random.stateless_uniform(
            [],
            minval=start_range_min,
            maxval=start_range_max,
            dtype=tf.int32,
            seed=seed)
        return inputs[start_pos:start_pos + max_input_tokens]

      result, pos_mask = answer_in_context(inputs, targets)

      if result:
        return slice_inputs(inputs, ans_len, pos_mask, seed=seed)
      else:
        return tf.constant([], dtype=inputs.dtype)

    if tf.greater(tf.shape(inputs)[0], max_input_tokens):
      inputs = truncate_inputs()
    return {'inputs': inputs, 'targets': features['targets']}

  dataset = my_fn(dataset)
  return dataset.filter(lambda x: tf.size(x['inputs']) > 0)


@gin.configurable()
def unsupervised(dataset,
                 preprocessors=None,
                 output_features=None,
                 sequence_length=None):
  """Configure this to point at unsupervised preprocessors.

   This function creates an extra level of indirection in case we want
   different unsupervised pretraining functions in the future which do not
   fit into the denoise() framework.

   This function should be used as a post-cache preprocessing function.

  Args:
    dataset: A tf.data.Dataset to process.
    preprocessors: a list of token-preprocessor functions. These functions
      should take unused kwargs if output_features or sequence_length is not
      used.
    output_features: dict(str, Feature), output features of the Task to be
      passed to the model.
    sequence_length: dict mapping feature key to int length for that feature.

  Returns:
    A preprocessed tf.data.Dataset.
  """
  if preprocessors is None:
    logging.warning(
        'unsupervised preprocessor got preprocessors=None; no preprocessing '
        'will be applied.'
    )
    return dataset

  kwargs = {}
  if output_features:
    kwargs['output_features'] = output_features
  if sequence_length:
    kwargs['sequence_length'] = sequence_length

  for p in preprocessors:
    dataset = p(dataset, **kwargs)
  return dataset

# ======================== split_tokens and helpers ============================


@gin.configurable
def split_tokens(dataset: tf.data.Dataset,
                 min_tokens_per_segment: Optional[int] = None,
                 max_tokens_per_segment: int = gin.REQUIRED,
                 feature_key: str = 'targets',
                 additional_feature_keys: Optional[Sequence[str]] = None,
                 passthrough_feature_keys: Optional[Sequence[str]] = None,
                 **unused_kwargs) -> tf.data.Dataset:
  """Split examples into multiple examples each.

  The intended use case is to break up long examples for use in unsupervised
  transfer-learning.

  This function is generally preceded by select_random_chunk.

  If min_tokens_per_segment is provided, the segment length is chosen randomly
  per document from a log-uniform distribution.  If min_tokens_per_segment is
  None, then the segment length is max_tokens_per_segment (except for a possibly
  shorter last segment in each document).

  Args:
    dataset: a tf.data.Dataset with dictionaries containing the key feature_key.
    min_tokens_per_segment: an optional integer
    max_tokens_per_segment: an integer, the maximum number of tokens in each
      segment. Only the final segment may be shorter.
    feature_key: a string, the feature to split
    additional_feature_keys: Additional features to split. The same chunk size
      will be used, so they should be the same size as feature_key.
    passthrough_feature_keys: Features to pass through without any splitting.

  Returns:
    a dataset
  """
  if passthrough_feature_keys:
    split_keys = set([feature_key] + (additional_feature_keys or []))
    overlap_keys = split_keys & set(passthrough_feature_keys)
    if overlap_keys:
      raise ValueError(
          f'split keys {overlap_keys} also included in passthrough keys')

  @seqio.map_over_dataset(num_seeds=1)
  def _split_tokens(x, seed):
    """Split one token sequence into multiple sequences."""
    tokens = x[feature_key]
    n_tokens = tf.shape(tokens)[0]
    if min_tokens_per_segment is None:
      length = max_tokens_per_segment
    else:
      # pick a length - log-uniformly distributed
      length = tf.cast(
          tf.exp(
              tf.random.stateless_uniform(
                  [],
                  minval=math.log(min_tokens_per_segment),
                  maxval=math.log(max_tokens_per_segment),
                  seed=seed
              )
          ),
          tf.int32)

    # Pad to a multiple of length, then use tf.reshape to split up the tokens
    # into num_segments segments each of the given length.
    num_segments = tf.cast(
        tf.math.ceil(
            tf.cast(n_tokens, tf.float32) / tf.cast(length, tf.float32))
        ,
        tf.int32)
    padding = num_segments * length - tf.shape(tokens)[0]
    feature_keys_to_split = [feature_key]
    orig_lengths = {}
    outputs = {}
    if additional_feature_keys is not None:
      feature_keys_to_split.extend(additional_feature_keys)
    for k in feature_keys_to_split:
      with tf.control_dependencies([
          tf.assert_equal(
              tf.shape(tokens)[0],
              tf.shape(x[k])[0],
              message=(f'Additional feature {k} is not the same size as '
                       f'{feature_key} along axis 0 in split_tokens().')
          )
      ]):
        shape = tf.shape(x[k])[1:]
        shape_list = x[k].shape[1:]
        padded = tf.pad(
            x[k],
            tf.concat([[[0, padding]],
                       tf.zeros([len(shape_list), 2], dtype=tf.int32)],
                      axis=0))
        orig_lengths[k] = tf.concat(
            [tf.repeat(length, num_segments - 1), [length - padding]], axis=0)
        outputs[k] = tf.reshape(
            padded, tf.concat([[-1, length], shape], axis=0))
    return outputs, orig_lengths

  def _strip_padding(inputs, orig_lengths):
    output = {}
    for k, v in inputs.items():
      output[k] = v[:orig_lengths[k]]
    return output

  # Filter empty examples.
  dataset = dataset.filter(lambda x: tf.not_equal(tf.size(x[feature_key]), 0))

  if passthrough_feature_keys:
    # Extract passthrough keys into a separate dataset.
    def _extract_passthrough_fields(inputs):
      return {
          k: v for k, v in inputs.items() if k in passthrough_feature_keys
      }
    passthrough_ds = dataset.map(
        _extract_passthrough_fields, num_parallel_calls=AUTOTUNE)

  dataset = _split_tokens(dataset)

  if passthrough_feature_keys:
    # Get number of segments from each example in original dataset.
    def _extract_num_segments(inputs, orig_lengths):
      del orig_lengths
      return tf.shape(inputs[feature_key], out_type=tf.int64)[0]
    num_segments_ds = dataset.map(
        _extract_num_segments, num_parallel_calls=AUTOTUNE)

    # Construct a dataset where the passthrough fields are repeated once for
    # each segment.
    def _repeat_passthrough_fields(inputs, num_segments):
      return tf.data.Dataset.from_tensors(inputs).repeat(num_segments)
    passthrough_ds = tf.data.Dataset.zip(
        (passthrough_ds, num_segments_ds)).flat_map(_repeat_passthrough_fields)

  dataset = dataset.unbatch()
  dataset = dataset.map(_strip_padding, num_parallel_calls=AUTOTUNE)

  if passthrough_feature_keys:
    # Add the passthrough fields back to the original dataset.
    def _merge_passthrough_fields(inputs, passthrough_inputs):
      outputs = {}
      outputs.update(inputs)
      outputs.update(passthrough_inputs)
      return outputs
    dataset = tf.data.Dataset.zip((dataset, passthrough_ds)).map(
        _merge_passthrough_fields, num_parallel_calls=AUTOTUNE)

  return dataset


@gin.configurable
def split_tokens_to_inputs_length(dataset, sequence_length,
                                  output_features, **kwargs):
  max_tokens = sequence_length['inputs']
  if output_features['inputs'].add_eos:
    # Leave room to insert an EOS token.
    max_tokens -= 1

  return split_tokens(dataset, max_tokens_per_segment=max_tokens, **kwargs)


@gin.configurable
def split_tokens_to_targets_length(dataset, sequence_length,
                                   output_features, **kwargs):
  max_tokens = sequence_length['targets']
  if output_features['targets'].add_eos:
    # Leave room to insert an EOS token.
    max_tokens -= 1

  return split_tokens(dataset, max_tokens_per_segment=max_tokens, **kwargs)


@gin.configurable
def split_tokens_to_random_length(dataset, sequence_length,
                                  output_features, **kwargs):
  max_tokens = sequence_length['inputs']
  if output_features['inputs'].add_eos:
    # Leave room to insert an EOS token.
    max_tokens -= 1

  return split_tokens(dataset,
                      min_tokens_per_segment=8,
                      max_tokens_per_segment=max_tokens,
                      **kwargs)


@gin.configurable
def concatenate_and_split_to_fixed_length(dataset,
                                          sequence_length,
                                          output_features,
                                          feature_key='targets',
                                          **unused_kwargs):
  """Concatenate tokens across examples, then split to fixed-size chunks.

  Chunk length is determined by sequence_length[feature_key].

  Args:
    dataset: a tf.data.Dataset
    sequence_length: a dict of ints.
    output_features: a dict mapping feature name to t5.data.Feature.
    feature_key: a string
  Returns:
    a tf.data.Dataset
  """
  dataset = dataset.map(lambda x: {feature_key: x[feature_key]})
  max_tokens = sequence_length[feature_key]
  if output_features[feature_key].add_eos:
    # Leave room to insert an EOS token.
    max_tokens -= 1
  return dataset.unbatch().batch(max_tokens)


@gin.configurable
def filter_by_string_length(dataset,
                            feature_key='targets',
                            min_length=1,
                            max_length=1000000,
                            **unused_kwargs):
  """Filter examples by string length.

  Args:
    dataset: a tf.data.Dataset (not tokenized)
    feature_key: a string
    min_length: an integer
    max_length: an integer
  Returns:
    a tf.data.Dataset
  """
  def my_fn(x):
    l = tf.strings.length(x[feature_key])
    return tf.logical_and(tf.greater_equal(l, min_length),
                          tf.less_equal(l, max_length))
  return dataset.filter(my_fn)


@gin.configurable
def random_spans_helper(inputs_length=gin.REQUIRED,
                        noise_density=gin.REQUIRED,
                        mean_noise_span_length=gin.REQUIRED,
                        extra_tokens_per_span_inputs=gin.REQUIRED,
                        extra_tokens_per_span_targets=gin.REQUIRED,
                        verbose=False):
  """Training parameters to avoid padding with random_spans_noise_mask.

  When training a model with random_spans_noise_mask, we would like to set the
  other training hyperparmeters in a way that avoids padding.  This function
  helps us compute these hyperparameters.

  We assume that each noise span in the input is replaced by
  extra_tokens_per_span_inputs sentinel tokens, and each non-noise span in the
  targets is replaced by extra_tokens_per_span_targets sentinel tokens.

  This function tells us the required number of tokens in the raw example (for
  split_tokens()) as well as the length of the encoded targets.

  Note that this function assumes the inputs and targets will have EOS appended
  and includes that in the reported length.

  Args:
    inputs_length: an integer - desired length of the tokenized inputs sequence
    noise_density: a float
    mean_noise_span_length: a float
    extra_tokens_per_span_inputs: an integer
    extra_tokens_per_span_targets: an integer
    verbose: a bool indicating whether to log sequence lengths
  Returns:
    tokens_length: length of original text in tokens
    targets_length: an integer - length in tokens of encoded targets sequence
  """
  def _tokens_length_to_inputs_length_targets_length(tokens_length):
    num_noise_tokens = int(round(tokens_length * noise_density))
    num_nonnoise_tokens = tokens_length - num_noise_tokens
    num_noise_spans = int(round(num_noise_tokens / mean_noise_span_length))
    # inputs contain all nonnoise tokens, sentinels for all noise spans
    # and one EOS token.
    return (
        num_nonnoise_tokens +
        num_noise_spans * extra_tokens_per_span_inputs + 1,
        num_noise_tokens +
        num_noise_spans * extra_tokens_per_span_targets + 1)

  tokens_length = inputs_length - 1
  while (_tokens_length_to_inputs_length_targets_length(tokens_length + 1)[0]
         <= inputs_length):
    tokens_length += 1
  inputs_length, targets_length = (
      _tokens_length_to_inputs_length_targets_length(tokens_length))
  # minor hack to get the targets length to be equal to inputs length
  # which is more likely to have been set to a nice round number.
  if noise_density == 0.5 and targets_length > inputs_length:
    tokens_length -= 1
    targets_length -= 1
  if verbose:
    logging.info(
        'tokens_length=%s inputs_length=%s targets_length=%s '
        'noise_density=%s mean_noise_span_length=%s ',
        tokens_length, inputs_length, targets_length,
        noise_density, mean_noise_span_length)
  return tokens_length, targets_length


@gin.configurable
def random_spans_tokens_length():
  """Helper for gin-configuring split_tokens with random_spans_noise_mask."""
  return random_spans_helper()[0]


@gin.configurable
def random_spans_targets_length():
  """Helper for gin-configuring the targets sequence length."""
  return random_spans_helper()[1]


# ========================== denoise and helpers ===============================


class DenoiseNoiseMaskFn(Protocol):

  def __call__(self, num_tokens: tf.Tensor, noise_density: float,
               seeds: tf.Tensor) -> tf.Tensor:
    """Computes the boolean makes. Seeds should have shape [2, 2]."""


class DenoiseInputsFn(Protocol):

  def __call__(self, tokens: tf.Tensor, noise_mask: tf.Tensor, vocabulary,
               seeds: tf.Tensor) -> tf.Tensor:
    """Computes the input tokens. Seeds should have shape [2, 2]."""


class DenoiseTargetsFn(Protocol):

  def __call__(self, tokens: tf.Tensor, noise_mask: tf.Tensor, vocabulary,
               seeds: tf.Tensor) -> tf.Tensor:
    """Computes the target tokens. Seeds should have shape [2, 2]."""


def single_example_denoise(features: FeatureType,
                           seed: tf.Tensor,
                           *,
                           output_features: Mapping[str, Any],
                           noise_density: float,
                           noise_mask_fn: DenoiseNoiseMaskFn,
                           inputs_fn: DenoiseInputsFn,
                           targets_fn: Optional[DenoiseTargetsFn] = None,
                           passthrough_feature_keys: Optional[
                               Sequence[str]] = None,
                           input_feature_key: str = 'inputs') -> FeatureType:
  """Preprocessing function for self-supervised denoising tasks.

  This function takes a dataset containing "targets" sequences,
  and turns each sequence into a dictionary containing:
  {
     "inputs": noisy version of the original sequence
     "targets": the full original sequence or missing parts of original sequence
  }

  In particular, for each sequence, we choose a boolean noise_mask identifying
  which tokens in the sequence to corrupt, as defined by the given
  noise_mask_fn.

  Given the sequence and the noise mask, we generate the inputs and targets
  using the given inputs_fn and targets_fn respectively.

  The self-supervised tasks vary along these axes:
    - noise_density: What fraction of the tokens to select as noise
    - noise_mask_fn: What pattern should the noise mask follow
         (iid, regular segments, etc.)
    - inputs_fn: How to apply the noise
         (drop noise tokens, replace with sentinels, etc.)
    - targets_fn: How to represent the output
         (full sequence, only non-noise tokens, etc.)

  Note: Some functionality has been deleted, which we may or may not want to
  restore at a later date.  The code for this functionality can be found in
  the deleted code for this CL.  In particular:
    - mixture of masking and random replacement
    - task labels prepended to the inputs

  Args:
    features: Flat dictionary of features.
    seed: Random seed to use.
    output_features: a dict mapping feature name to t5.data.Feature.
    noise_density: a float
    noise_mask_fn: a function from (length, noise_density) -> boolean mask
    inputs_fn: a function from (tokens, noise_mask, vocabulary) -> tokens
    targets_fn: a function from (tokens, noise_mask, vocabulary) -> tokens
    passthrough_feature_keys: names of additional features to include in output
    input_feature_key: name of feature to use as inputs

  Returns:
    A preprocessed features.
  """
  if passthrough_feature_keys and (input_feature_key in passthrough_feature_keys
                                   or 'targets' in passthrough_feature_keys):
    raise ValueError(
        f"passthrough keys cannot contain '{input_feature_key}' or 'targets'")

  seeds = tf.unstack(tf.random.experimental.stateless_split(seed, 6))
  tokens = features['targets']
  vocabulary = output_features['targets'].vocabulary
  if (input_feature_key in output_features and
      vocabulary != output_features[input_feature_key].vocabulary):
    raise ValueError(
        'denoise creates inputs based on tokenized targets but was applied '
        'to a task that uses different vocabularies for inputs and targets.')
  noise_mask = noise_mask_fn(tf.size(tokens), noise_density, seeds=seeds[:2])
  inputs = inputs_fn(tokens, noise_mask, vocabulary, seeds=seeds[2:4])
  if targets_fn:
    targets = targets_fn(tokens, noise_mask, vocabulary, seeds=seeds[4:6])
  else:
    targets = tokens
  return {
      input_feature_key: inputs,
      'targets': targets,
      **{
          k: features[k]
          for k in features
          if passthrough_feature_keys and k in passthrough_feature_keys
      }
  }


@gin.configurable()
def denoise(dataset,
            output_features,
            noise_density=gin.REQUIRED,
            noise_mask_fn=gin.REQUIRED,
            inputs_fn=gin.REQUIRED,
            targets_fn=None,
            passthrough_feature_keys: Optional[Sequence[str]] = None,
            input_feature_key='inputs',
            **unused_kwargs):
  """SeqIO wrapper for single_example_denoise()."""

  @seqio.map_over_dataset(num_seeds=1)
  def my_fn(features, seed):
    return single_example_denoise(
        features,
        seed,
        output_features=output_features,
        noise_density=noise_density,
        noise_mask_fn=noise_mask_fn,
        inputs_fn=inputs_fn,
        targets_fn=targets_fn,
        passthrough_feature_keys=passthrough_feature_keys,
        input_feature_key=input_feature_key)

  return my_fn(dataset)


@gin.configurable()
def iid_noise_mask(length, noise_density, seeds):
  """Independent and identically distributed token noise.

  Args:
    length: an int32 scalar.
    noise_density: a float - approximate density of output mask.
    seeds: an int32 Tensor, shaped (1, 2), the random seed.

  Returns:
    a boolean tensor with shape [length].
  """
  return tf.random.stateless_uniform([length], seed=seeds[0]) < noise_density


@gin.configurable()
def regular_noise_mask(length,
                       noise_density,
                       seeds,
                       min_span_length=1,
                       max_span_length=5):
  """Noise mask consisting of equally spaced spans of equal length.

  The span length and the offset are chosen randomly per-example.
  The beginning and end of the sequence may be part of shorter spans of noise.
  For example, if noise_density=0.25 and a span length of 2 is chosen,
  then the output might be:

  [T F F F F F F T T F F F F F F T T F F F F F F T T F F]

  Args:
    length: an int32 scalar.
    noise_density: a float - approximate density of output mask.
    seeds: an int32 Tensor, shaped (2, 2), the random seeds.
    min_span_length: an integer.
    max_span_length: an integer.

  Returns:
    a boolean tensor with shape [length].
  """
  span_length = tf.random.stateless_uniform(
      [],
      minval=min_span_length,
      maxval=max_span_length + 1,
      dtype=tf.int32,
      seed=seeds[0])
  period = tf.cast(
      tf.round(tf.cast(span_length, tf.float32) / noise_density), tf.int32)
  offset = tf.random.stateless_uniform(
      [],
      maxval=period,
      dtype=tf.int32,
      seed=seeds[1])
  return (tf.range(length, dtype=tf.int32) + offset) % period < span_length


@gin.configurable()
def random_spans_noise_mask(length,
                            noise_density,
                            seeds,
                            mean_noise_span_length=3.0,
                            random_roll=False):
  """Noise mask consisting of random spans of noise tokens.

  The number of noise tokens and the number of noise spans and non-noise spans
  are determined deterministically as follows:

    num_noise_tokens = round(length * noise_density)
    num_nonnoise_spans = num_noise_spans = round(
       num_noise_tokens / mean_noise_span_length)

  Spans alternate between non-noise and noise, beginning with non-noise.
  Subject to the above restrictions, all masks are equally likely.

  Args:
    length: an int32 scalar (length of the incoming token sequence)
    noise_density: a float - approximate density of output mask
    seeds: an int32 Tensor, shaped (2, 2)
    mean_noise_span_length: a number
    random_roll: bool, whether to roll the mask by a random integer offset in
      [0, length). Set random_roll to True to get a more uniform distribution
      of masked positions. Specifically, when random_roll is False (default) and
      a single span is enough to satisfy the noise density requirement, this
      fuction masks only the last few positions.

  Returns:
    a boolean tensor with shape [length]
  """

  if noise_density == 0.0:
    return tf.zeros(length, tf.bool)

  orig_length = length
  # increase length to avoid degeneracy
  length = tf.maximum(length, 2)
  def to_int(x):
    return tf.cast(x, tf.int32)
  def to_float(x):
    return tf.cast(x, tf.float32)
  num_noise_tokens = to_int(tf.round(to_float(length) * noise_density))
  # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
  num_noise_tokens = tf.minimum(tf.maximum(num_noise_tokens, 1), length - 1)
  num_noise_spans = to_int(
      tf.round(to_float(num_noise_tokens) / mean_noise_span_length))
  # avoid degeneracy by ensuring positive number of noise spans
  num_noise_spans = tf.maximum(num_noise_spans, 1)
  num_nonnoise_tokens = length - num_noise_tokens
  # pick the lengths of the noise spans and the non-noise spans
  def _random_segmentation(num_items, num_segments, seed):
    """Partition a sequence of items randomly into non-empty segments.

    Args:
      num_items: an integer scalar > 0
      num_segments: an integer scalar in [1, num_items]
      seed: an integer seed
    Returns:
      a Tensor with shape [num_segments] containing positive integers that add
      up to num_items
    """
    first_in_segment = tf.pad(
        seqio.stateless_shuffle(
            to_int(tf.range(num_items - 1) < num_segments - 1),
            seed),
        [[1, 0]])
    segment_id = tf.cumsum(first_in_segment)
    segment_length = tf.math.segment_sum(tf.ones_like(segment_id), segment_id)
    return segment_length
  noise_span_lengths = _random_segmentation(
      num_noise_tokens, num_noise_spans, seeds[0])
  nonnoise_span_lengths = _random_segmentation(
      num_nonnoise_tokens, num_noise_spans, seeds[1])
  interleaved_span_lengths = tf.reshape(
      tf.stack([nonnoise_span_lengths, noise_span_lengths], axis=1),
      [num_noise_spans * 2])
  span_starts = tf.cumsum(interleaved_span_lengths)[:-1]
  span_start_indicator = tf.math.unsorted_segment_sum(
      tf.ones_like(span_starts), span_starts, length)
  span_num = tf.cumsum(span_start_indicator)
  is_noise = tf.equal(span_num % 2, 1)

  mask = is_noise[:orig_length]

  if random_roll:
    roll_seed = (seeds[0][0]+seeds[1][1], seeds[0][1]-seeds[1][0])  # new seed.
    # Roll the mask by a random offset e.g. for offset=2: [1,2,3,4] => [3,4,1,2]
    offset = tf.random.stateless_uniform(
        [1], seed=roll_seed, dtype=tf.int32, minval=0, maxval=length)[0]
    mask = tf.roll(mask, shift=offset, axis=0)

  return mask


@gin.configurable()
def random_prefix_noise_mask(length, noise_density, seeds):
  """First part of the sequence is noise (for prefix_lm).

  The length of the prefix is chosen uniformly between [1, length)
  noise_density must be 0.5.
  TODO(noam): figure out some distribution to use if noise_density != 0.5.

  Args:
    length: an int32 scalar.
    noise_density: a float - must equal 0.5.
    seeds: an int32 Tensor, shaped (1, 2), the random seed.

  Returns:
    a boolean tensor with shape [length].
  """
  if noise_density != 0.5:
    raise NotImplementedError(
        'noise density must equal 0.5 for random_prefix_noise_mask')
  max_input_tokens = length - 1
  min_input_tokens = tf.minimum(max_input_tokens, 1)
  num_input_tokens = tf.random.stateless_uniform(
      [],
      minval=min_input_tokens,
      maxval=max_input_tokens + 1,
      dtype=tf.int32,
      seed=seeds[0])
  return tf.range(length, dtype=tf.int32) < num_input_tokens


@gin.configurable()
def sentinel_id(vocabulary, return_value=None):
  """Token ID to use as a sentinel.

  By default, we use the last token in the vocabulary.

  Args:
    vocabulary: a t5.data.vocabularies.Vocabulary
    return_value: an optional integer
  Returns:
    an integer
  """
  if return_value is not None:
    return return_value
  return vocabulary.vocab_size - 1


@gin.configurable()
def noise_token_to_sentinel(tokens, noise_mask, vocabulary, seeds):
  """Replace each noise token with the given sentinel.

  Args:
    tokens: a 1d integer Tensor
    noise_mask: a boolean Tensor with the same shape as tokens
    vocabulary: a vocabulary.Vocabulary
    seeds: an unused int32 Tensor

  Returns:
    a Tensor with the same shape and dtype as tokens
  """
  del seeds
  return tf.where(noise_mask,
                  tf.cast(sentinel_id(vocabulary), tokens.dtype),
                  tokens)


@gin.configurable()
def noise_span_to_sentinel(tokens, noise_mask, vocabulary, seeds):
  """Replace each run of consecutive noise tokens with a single sentinel.

  Args:
    tokens: a 1d integer Tensor
    noise_mask: a boolean Tensor with the same shape as tokens
    vocabulary: a vocabulary.Vocabulary
    seeds: an unused int32 Tensor
  Returns:
    a Tensor with the same shape and dtype as tokens
  """
  del seeds
  tokens = tf.where(noise_mask,
                    tf.cast(sentinel_id(vocabulary), tokens.dtype),
                    tokens)
  prev_token_is_noise = tf.pad(noise_mask[:-1], [[1, 0]])
  subsequent_noise_tokens = tf.logical_and(noise_mask, prev_token_is_noise)
  return tf.boolean_mask(tokens, tf.logical_not(subsequent_noise_tokens))


@gin.configurable()
def nonnoise_span_to_sentinel(tokens, noise_mask, vocabulary, seeds):
  return noise_span_to_sentinel(
      tokens, tf.logical_not(noise_mask), vocabulary, seeds)


@gin.configurable()
def noise_span_to_unique_sentinel(tokens, noise_mask, vocabulary, seeds):
  """Replace each run of consecutive noise tokens with a different sentinel.

  The idea here is to be able to align the dropped spans in the inputs
  with the markers in the targets.

  We want to generate training examples like
  "We hold X to be Y that" -> "X these truths Y self evident Z"

  Sentinels assigned in decreasing order within the sequence starting at
  vocabulary.size - 1.  That is, we appropriate the last tokens in the
  vocabulary for additional use as sentinels.

  TODO(noam): we may want to try enlarging the vocabulary and leaving room
  for the sentinels instead.  However, this requires enlarging the embedding
  tables in the model, so that is a bigger change.

  Args:
    tokens: a 1d integer Tensor
    noise_mask: a boolean Tensor with the same shape as tokens
    vocabulary: a vocabulary.Vocabulary
    seeds: an unused int32 Tensor
  Returns:
    a Tensor with the same shape and dtype as tokens
  """
  del seeds

  prev_token_is_noise = tf.pad(noise_mask[:-1], [[1, 0]])

  first_noise_tokens = tf.logical_and(
      noise_mask, tf.logical_not(prev_token_is_noise))
  subsequent_noise_tokens = tf.logical_and(noise_mask, prev_token_is_noise)

  sentinel = sentinel_id(vocabulary) + 1 - tf.cumsum(
      tf.cast(first_noise_tokens, tokens.dtype))

  tokens = tf.where(first_noise_tokens, sentinel, tokens)
  return tf.boolean_mask(tokens, tf.logical_not(subsequent_noise_tokens))


@gin.configurable()
def nonnoise_span_to_unique_sentinel(tokens, noise_mask, vocabulary, seeds):
  return noise_span_to_unique_sentinel(
      tokens, tf.logical_not(noise_mask), vocabulary, seeds)


@gin.configurable()
def drop_noise_tokens(tokens, noise_mask, vocabulary, seeds):
  """Drop noise tokens without inserting a sentinel.

  Args:
    tokens: a 1d integer Tensor
    noise_mask: a boolean Tensor with the same shape as tokens
    vocabulary: an unused vocabulary.Vocabulary
    seeds: an unused int32 Tensor

  Returns:
    a Tensor with the same shape and dtype as tokens
  """
  del vocabulary
  del seeds
  return tf.boolean_mask(tokens, tf.logical_not(noise_mask))


@gin.configurable()
def drop_nonnoise_tokens(tokens, noise_mask, vocabulary, seeds):
  """Drop non-noise tokens without inserting a sentinel.

  Args:
    tokens: a 1d integer Tensor
    noise_mask: a boolean Tensor with the same shape as tokens
    vocabulary: an unused vocabulary.Vocabulary
    seeds: an unused int32 Tensor
  Returns:
    a Tensor with the same shape and dtype as tokens
  """
  del vocabulary
  del seeds
  return tf.boolean_mask(tokens, noise_mask)


@gin.configurable()
def permute_noise_tokens(tokens, noise_mask, vocabulary, seeds):
  """Permute the noise tokens, keeping the non-noise tokens where they are.

  Args:
    tokens: a 1d integer Tensor
    noise_mask: a boolean Tensor with the same shape as tokens
    vocabulary: an unused vocabulary.Vocabulary
    seeds: an int32 Tensor, sized (1, 2)
  Returns:
    a Tensor with the same shape and dtype as tokens
  """
  del vocabulary

  masked_only = tf.boolean_mask(tokens, noise_mask)
  permuted = seqio.stateless_shuffle(masked_only, seeds[0])
  # pad to avoid errors when it has size 0
  permuted = tf.pad(permuted, [[0, 1]])
  indices = tf.cumsum(tf.cast(noise_mask, tf.int32), exclusive=True)
  return tf.where(noise_mask,
                  tf.gather(permuted, indices),
                  tokens)


@gin.configurable()
def noise_token_to_gathered_token(tokens, noise_mask, vocabulary, seeds):
  """Replace each noise token with a random token from the sequence.

  Args:
    tokens: a 1d integer Tensor
    noise_mask: a boolean Tensor with the same shape as tokens
    vocabulary: an unused vocabulary.Vocabulary
    seeds: an int32 Tensor, sized (1, 2)
  Returns:
    a Tensor with the same shape and dtype as tokens
  """
  del vocabulary

  indices = tf.random.stateless_uniform(
      shape=tf.shape(tokens),
      maxval=tf.size(tokens),
      dtype=tf.int32,
      seed=seeds[0])
  return tf.where(noise_mask,
                  tf.gather(tokens, indices),
                  tokens)


@gin.configurable()
def noise_token_to_random_token(
    tokens,
    noise_mask,
    vocabulary,
    seeds,
    num_reserved_tokens=3):
  """Replace each noise token with a random token from the vocabulary.



  Args:
    tokens: a 1d integer Tensor
    noise_mask: a boolean Tensor with the same shape as tokens
    vocabulary: a vocabulary.Vocabulary
    seeds: an int32 Tensor, shaped (1, 2)
    num_reserved_tokens: an integer
  Returns:
    a Tensor with the same shape and dtype as tokens
  """
  return tf.where(noise_mask,
                  tf.random.stateless_uniform(
                      tf.shape(tokens),
                      minval=num_reserved_tokens,
                      maxval=vocabulary.vocab_size,
                      dtype=tokens.dtype,
                      seed=seeds[0]),
                  tokens)


@gin.configurable()
def noise_token_to_random_token_or_sentinel(
    tokens,
    noise_mask,
    vocabulary,
    seeds,
    random_prob=0.1):
  """Replace each noise token with a random token or a sentinel.

  For each masked token, with probability random_prob, we replace it by a
  random token from the vocabulary.  Otherwise, we replace it with a sentinel.

  Args:
    tokens: a 1d integer Tensor
    noise_mask: a boolean Tensor with the same shape as tokens
    vocabulary: a vocabulary.Vocabulary
    seeds: an int32 Tensor, shaped (2, 2).
    random_prob: a float
  Returns:
    a Tensor with the same shape and dtype as tokens
  """
  use_random = (
      tf.random.stateless_uniform(tf.shape(tokens), seed=seeds[0]) <
      random_prob)
  return tf.where(
      use_random,
      noise_token_to_random_token(
          tokens, noise_mask, vocabulary, seeds=seeds[1:]),
      noise_token_to_sentinel(
          tokens, noise_mask, vocabulary, seeds=()))


# =============== EXPERIMENTAL preprocessors (not used for the T5 paper) =======


def trim_and_pad_dataset(dataset, sequence_length):
  """A wrapper to use `seqio.utils.trim_and_pad_dataset` as a preprocessor."""
  return seqio.utils.trim_and_pad_dataset(
      dataset, feature_lengths=sequence_length)


def targets_for_prefix_lm_objective(dataset, sequence_length, output_features):
  """Prepares targets to be used for prefix LM objective."""
  dataset = select_random_chunk(
      dataset, output_features, max_length=65536, feature_key='targets')
  dataset = seqio.preprocessors.append_eos(dataset, output_features)
  dataset = reduce_concat_tokens(dataset, batch_size=128)
  dataset = split_tokens(
      dataset, max_tokens_per_segment=sequence_length['targets'])
  dataset = trim_and_pad_dataset(dataset, sequence_length)
  return dataset


def pack_prefix_lm_encoder_decoder(ds, sequence_length, pad_id=0):
  """Pack two examples into one with the prefix LM objective."""
  packed_length = next(iter(sequence_length.values()))
  assert packed_length % 2 == 0
  assert all(l == packed_length for l in sequence_length.values())

  @seqio.utils.map_over_dataset(num_seeds=1)
  def pack_examples(example_pair, seed):
    split_point = tf.random.stateless_uniform((),
                                              minval=1,
                                              maxval=packed_length,
                                              seed=seed,
                                              dtype=tf.int32)
    inputs = tf.concat([
        example_pair['targets'][0][:split_point],
        example_pair['targets'][1][:packed_length - split_point]
    ],
                       axis=0)
    inputs = tf.reshape(inputs, (packed_length,))
    targets = tf.concat([
        example_pair['targets'][0][split_point:],
        example_pair['targets'][1][packed_length - split_point:]
    ],
                        axis=0)
    targets = tf.reshape(targets, (packed_length,))

    encoder_segment_ids = tf.cast(
        tf.range(packed_length) >= split_point, tf.int32) + 1
    decoder_segment_ids = tf.cast(
        tf.range(packed_length) >= (packed_length - split_point), tf.int32) + 1

    decoder_input_tokens = seqio.utils.make_autoregressive_inputs(
        targets, sequence_id=decoder_segment_ids)

    encoder_positions = tf.concat(
        [tf.range(split_point),
         tf.range(packed_length - split_point)], axis=0)
    encoder_positions = tf.reshape(encoder_positions, (packed_length,))
    decoder_positions = tf.concat(
        [tf.range(packed_length - split_point),
         tf.range(split_point)], axis=0)
    decoder_positions = tf.reshape(decoder_positions, (packed_length,))
    decoder_loss_weights = tf.cast(
        tf.not_equal(targets, pad_id), dtype=tf.int32)
    return {
        'encoder_input_tokens': inputs,
        'decoder_target_tokens': targets,
        'decoder_input_tokens': decoder_input_tokens,
        'encoder_segment_ids': encoder_segment_ids,
        'encoder_positions': encoder_positions,
        'decoder_segment_ids': decoder_segment_ids,
        'decoder_positions': decoder_positions,
        'decoder_loss_weights': decoder_loss_weights,
    }

  # Note that the batch requires the lengths to be the same.
  return pack_examples(ds.batch(2))


def pack_prefix_lm_decoder_only(ds,
                                sequence_length,
                                loss_on_targets_only=True,
                                pad_id=0):
  """Randomly split the tokens for the prefix LM objective."""
  packed_length = next(iter(sequence_length.values()))
  assert packed_length % 2 == 0
  assert all(l == packed_length for l in sequence_length.values())

  @seqio.utils.map_over_dataset(num_seeds=1)
  def pack_examples(example, seed):
    split_point = tf.random.stateless_uniform((),
                                              minval=1,
                                              maxval=packed_length,
                                              seed=seed,
                                              dtype=tf.int32)
    decoder_target_tokens = example['targets']
    decoder_input_tokens = seqio.utils.make_autoregressive_inputs(
        decoder_target_tokens)

    if loss_on_targets_only:
      decoder_loss_weights = tf.cast(
          tf.range(packed_length) >= split_point, tf.int32)
    else:
      decoder_loss_weights = tf.ones((packed_length,), dtype=tf.int32)

    padding_mask = tf.cast(
        tf.not_equal(decoder_target_tokens, pad_id), dtype=tf.int32)
    decoder_loss_weights *= padding_mask

    decoder_causal_attention = tf.cast(
        tf.range(packed_length) <= split_point, tf.int32)

    return {
        'decoder_target_tokens': decoder_target_tokens,
        'decoder_input_tokens': decoder_input_tokens,
        'decoder_loss_weights': decoder_loss_weights,
        'decoder_causal_attention': decoder_causal_attention,
    }

  return pack_examples(ds)
