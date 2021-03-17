# Copyright 2021 The T5 Authors.
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

"""Experimental utilities for SeqIO."""
import inspect
from typing import Callable, Iterable, Mapping, Optional, Sequence

from absl import logging
from t5.seqio import dataset_providers
from t5.seqio import utils
import tensorflow as tf


CacheDatasetPlaceholder = dataset_providers.CacheDatasetPlaceholder
Mixture = dataset_providers.Mixture
MixtureRegistry = dataset_providers.MixtureRegistry
ShardInfo = dataset_providers.ShardInfo
Task = dataset_providers.Task
TaskRegistry = dataset_providers.TaskRegistry


def _get_fully_cached_name(
    original_name: str,
    sequence_length: Mapping[str, int]
) -> str:
  """Generates name for fully-cached task or mixture."""
  new_name = f'{original_name}_'
  # Find shortest unique prefix.
  prefix_len = 0
  while (len(set(feat[:prefix_len] for feat in sequence_length)) !=
         len(sequence_length)):
    prefix_len += 1
  new_name += '_'.join(
      f'{feat[:prefix_len]}{sequence_length[feat]}' for feat in sequence_length)
  return new_name


def add_fully_cached_task(
    task_name: str,
    sequence_length: Mapping[str, int],
    disallow_shuffling: bool = False
) -> Task:
  """Adds fully-cached version of the task for given sequence lengths."""
  task = TaskRegistry.get(task_name)
  new_name = _get_fully_cached_name(task_name, sequence_length)

  try:
    return TaskRegistry.get(new_name)
  except ValueError:
    pass

  # Rename the sequence lengths to differentiate from the preprocessor kwarg.
  fixed_sequence_length = sequence_length

  new_preprocessors = []
  for prep in task.preprocessors:
    if isinstance(prep, CacheDatasetPlaceholder):
      continue

    def wrapped_prep(ds, output_features, prep=prep):
      prep_args = inspect.signature(prep).parameters.keys()
      extra_kwargs = {}
      if 'sequence_length' in prep_args:
        extra_kwargs['sequence_length'] = fixed_sequence_length
      if 'output_features' in prep_args:
        extra_kwargs['output_features'] = output_features
      return prep(ds, **extra_kwargs)

    new_preprocessors.append(wrapped_prep)

  # Cache at the end of the pipeline.
  new_preprocessors.append(CacheDatasetPlaceholder(required=True))

  # Add post-cache preprocessor to ensure the runtime sequence length is valid.
  def validate_sequence_length(ds, sequence_length):
    if sequence_length is not None and sequence_length != fixed_sequence_length:
      raise ValueError(
          f"Fully-cached task '{new_name}' can only be loaded with "
          f'`sequence_length={fixed_sequence_length}` or `None`. '
          f'Given sequence_length={sequence_length}.'
      )
    return ds
  new_preprocessors.append(validate_sequence_length)

  logging.info("Registering fully cached Task '%s' with sequence lengths %s.",
               new_name, sequence_length)

  return TaskRegistry.add(
      new_name,
      source=task.source,
      preprocessors=new_preprocessors,
      output_features=task.output_features,
      metric_fns=task.metric_fns,
      shuffle_buffer_size=
      None if disallow_shuffling else dataset_providers.SHUFFLE_BUFFER_SIZE
  )


def add_fully_cached_mixture(
    mixture_name: str,
    sequence_length: Mapping[str, int],
    disallow_shuffling: bool = False
) -> Mixture:
  """Adds fully-cached version of the mixture for given sequence lengths."""
  mixture = MixtureRegistry.get(mixture_name)
  new_name = _get_fully_cached_name(mixture_name, sequence_length)

  # Register fully-cached tasks for the mixture.
  new_tasks = [
      add_fully_cached_task(task.name, sequence_length, disallow_shuffling)
      for task in mixture.tasks]

  logging.info(
      "Registering fully cached Mixture '%s' with sequence lengths %s.",
      new_name, sequence_length)
  return MixtureRegistry.add(
      new_name,
      [(new_t.name, mixture._task_to_rate[old_t.name])  # pylint:disable=protected-access
       for old_t, new_t in zip(mixture.tasks, new_tasks)])


class FewshotDataSource(dataset_providers.DataSource):
  """Combines two splits of another `DataSource` to provide fewshot examples.

  Output examples are a dictionary containing a single eval example and a batch
  of train examples. For example, with `num_shots=2`:

  {
    'train': {
        'inputs': [
            'How many Beatles are there?', 'How many Beatles are alive in 2020?'
        ],
        'targets': ['4', '2']
    },
    'eval': {
        'inputs': 'What city were the Beatles from?'
        'targets': 'Liverpool'
    }
  }

  Note that if `num_shots` is 0, the 'train' entry will not be included in the
  resulting examples.
  """

  def __init__(
      self,
      original_source: dataset_providers.DataSource,
      num_shots: int,
      train_preprocessors:
      Iterable[Callable[[tf.data.Dataset], tf.data.Dataset]] = (),
      eval_preprocessors:
      Iterable[Callable[[tf.data.Dataset], tf.data.Dataset]] = (),
      train_split: str = 'train',
      train_feature_keys: Iterable[str] = ('inputs', 'targets')
  ):
    """Initializes FewshotDataSource.

    Args:
      original_source: a DataSource to produce fewshot examples from.
      num_shots: A non-negative integer specifying how many training examples to
        include in the inputs.
      train_preprocessors: an iterable of preprocessors to run on the train
        split before zipping with the eval split.
      eval_preprocessors: an iterable of preprocessors to run on the eval
        split before zipping with the train split.
      train_split: the split to use as training examples.
      train_feature_keys: the features to retain in the train split after
        preprocessing but before batching zipping with the eval split. This is
        necessary to remove variable-length sequences, which cannot be batched.
    """
    self._original_source = original_source
    self._num_shots = num_shots
    self._train_preprocessors = train_preprocessors
    self._eval_preprocessors = eval_preprocessors
    self._train_split = train_split
    self._train_feature_keys = train_feature_keys

    # Override split in property since it may need to be loaded lazily (e.g.,
    # for TfdsSource)
    super().__init__(splits=())

  @property
  def splits(self) -> Sequence[str]:
    return tuple(
        s for s in self._original_source.splits if s != self._train_split)

  def list_shards(self, split: str) -> Sequence[str]:
    return self._original_source.list_shards(split)

  def get_dataset(
      self,
      split: str,
      shuffle: bool = True,
      seed: Optional[int] = None,
      shard_info: Optional[ShardInfo] = None
  ) -> tf.data.Dataset:
    shard_info: ShardInfo = shard_info or ShardInfo(0, 1)
    if self._train_split not in self._original_source.splits:
      raise ValueError(
          f"Train split '{self._train_split}' is not one of the original "
          f"source splits: {self._original_source.splits}")

    if not self._num_shots:
      logging.warning(
          'Train examples will not be included in the provided dataset since '
          '`num_shots` is 0.')

    def _apply_preprocessors(ds, preprocessors):
      for prep_fn in preprocessors:
        ds = prep_fn(ds)
      return ds

    def _get_maybe_sharded_dataset(split_: str) -> tf.data.Dataset:
      """Shard at source if possible, but fall back to examples if not."""
      num_shards = len(self._original_source.list_shards(split_))
      if num_shards >= shard_info.num_shards:
        # Shard at the source.
        return self._original_source.get_dataset(
            split=split_, shuffle=shuffle, seed=seed, shard_info=shard_info)
      else:
        # Shard the examples.
        return self._original_source.get_dataset(
            split=split_, shuffle=shuffle, seed=seed).shard(
                shard_info.num_shards, shard_info.index)

    datasets = {}
    if self._num_shots:
      train_ds = _get_maybe_sharded_dataset(self._train_split)
      train_ds = _apply_preprocessors(train_ds, self._train_preprocessors)
      train_ds = train_ds.map(
          lambda x: {k: x[k] for k in self._train_feature_keys},
          num_parallel_calls=tf.data.experimental.AUTOTUNE)
      train_ds = train_ds.repeat().batch(self._num_shots)
      datasets['train'] = train_ds

    eval_ds = _get_maybe_sharded_dataset(split)
    eval_ds = _apply_preprocessors(eval_ds, self._eval_preprocessors)
    datasets['eval'] = eval_ds

    return tf.data.Dataset.zip(datasets)


def fewshot_preprocessor(
    ds,
    inputs_prefix='',
    targets_prefix='',
    example_separator='\n\n'):
  """Create 'inputs' and 'targets' strings for (zero/few)-shot evaluation.

  Inputs and targets will be formatted using the given prefixes along with a
  separator between each pair. The few-shot examples from the train set will
  include both inputs and targets, whereas the eval example (at the end) will
  contain only the input followed by the targets prefix.

  NOTE: The final target prefix will be right-stripped so that the input does
  not end with whitepsace.

  For example, a 2-shot output might look like:
  output: {
    'inputs':
      '0 How many states in the US? X 1 50 X 0 How many cents in a dollar? X '
      '1 100 X 0 Who was in the Beatles? X 1',
    'targets': 'John',
    'answers': ['John', 'Paul', 'George', 'Ringo']
  }

  Args:
    ds: A dictionary of zipped eval and train tf.data.Datasets, each
      preprocessed with at least the fields 'inputs' and 'targets'. Note that
      the train dataset will not exist in the 0-shot case.
    inputs_prefix: Prefix string for inputs.
    targets_prefix: Prefix string for targets.
    example_separator: The string separator to delimit different examples.
  Returns:
    A tf.data.Dataset containing 'inputs', 'targets', and any other features
    from the evaluation dataset.
  """

  @utils.map_over_dataset
  def fewshot_map(ex):
    if 'train' in ex:
      train_examples = tf.reshape(
          tf.stack(
              [
                  inputs_prefix + ex['train']['inputs'],
                  targets_prefix + ex['train']['targets'] + example_separator
              ],
              axis=1),
          [-1])
      shots = tf.strings.reduce_join(train_examples)
    else:
      shots = ''

    new_ex = {
        'inputs':
            shots + inputs_prefix + ex['eval']['inputs'] +
            targets_prefix.rstrip(),
        'targets': ex['eval']['targets'],
    }
    # Pass through other eval features unchanged.
    new_ex.update(
        {k: v for k, v in ex['eval'].items() if k not in ('inputs', 'targets')}
    )
    return new_ex

  ds = fewshot_map(ds)
  if ds.element_spec['inputs'].shape.rank:
    # Unbatch if not a scalar. This is useful for fewshot eval.
    ds = ds.unbatch()
  return ds

