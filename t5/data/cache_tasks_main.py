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

r"""Dumps preprocessed, tokenized tasks as TFRecord of tf.Examples.

Usage:
====================
t5_cache_tasks \
--tasks=my_task_*,your_task \
--excluded_tasks=my_task_5 \
--output_cache_dir=/path/to/cache_dir \
--module_import=my.tasks \
--alsologtostderr

"""

import importlib
import json
import os
import re

from absl import app
from absl import flags
from absl import logging

import apache_beam as beam
import apache_beam.metrics as metrics
import t5
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds


tf.disable_v2_behavior()

# Significantly speeds up preprocessing.
tf.enable_eager_execution()

FLAGS = flags.FLAGS

flags.DEFINE_list(
    "tasks", None,
    "Regexes matching task(s) to build a preprocessed dataset for. Will build "
    "all registered if not specified.")
flags.DEFINE_list(
    "excluded_tasks", None,
    "Regexes matching task(s) to skip.")
flags.DEFINE_string(
    "output_cache_dir", None,
    "The directory to output cached tasks to.")
flags.DEFINE_integer(
    "max_input_examples", None,
    "The maximum number of input examples to use. No limit if None.")
flags.DEFINE_list(
    "tasks_additional_cache_dirs", [],
    "Additional directories to search for cached Tasks after checking the "
    "global caches and `output_cache_dir`.")
flags.DEFINE_multi_string(
    "module_import", [],
    "Modules to import. Use this, for example, to add new `Task`s to the "
    "global `TaskRegistry`.")
flags.DEFINE_list(
    "pipeline_options", ["--runner=DirectRunner"],
    "A comma-separated list of command line arguments to be used as options "
    "for the Beam Pipeline.")
flags.DEFINE_boolean(
    "overwrite", False,
    "If true, overwrite the cached task even if it exists in the cached "
    "directories.")


def _import_modules(modules):
  for module in modules:
    if module:
      importlib.import_module(module)


class PreprocessAndTokenize(beam.PTransform):
  """Preprocesses and tokenizes a Task.

  Expects input to be a PCollection of sharded file paths.

  Returns a PCollection of tokenized example dicts containing Tensors.
  """

  def __init__(
      self, task, split, max_input_examples=None, modules_to_import=()):
    """PreprocessAndTokenize constructor.

    Args:
      task: Task, the task to process.
      split: string, the split to process.
      max_input_examples: (Optional) int, the maximum number of input examples
        to use.
      modules_to_import: (Optional) list, modules to import.
    """
    self._task = task
    self._max_input_examples = max_input_examples
    self._split = split
    self._modules_to_import = modules_to_import
    self.files = self._task.tfds_dataset.files(split)
    logging.info(
        "%s %s files: %s", task.name, split, ", ".join(
            ["%s" % inst for inst in self.files]))

  def _increment_counter(self, name):
    metrics.Metrics.counter(
        str("%s_%s" % (self._task.name, self._split)), name).inc()

  def _emit_tokenized_examples(self, shard_instruction):
    """Emits examples keyed by shard path and index for a single shard."""
    _import_modules(self._modules_to_import)
    logging.info("Processing shard: %s", shard_instruction)
    self._increment_counter("input-shards")

    ds = self._task.tfds_dataset.load_shard(shard_instruction)

    if self._max_input_examples:
      num_shard_examples = int(
          self._max_input_examples / len(self.files))
      ds = ds.repeat().take(num_shard_examples)

    ds = self._task.preprocess_text(ds)
    ds = t5.data.encode_string_features(
        ds, self._task.get_vocabulary(), keys=self._task.output_features,
        copy_plaintext=True)

    for ex in tfds.as_numpy(ds):
      self._increment_counter("examples")
      yield ex

  def expand(self, pipeline):
    return (
        pipeline
        | beam.Create(self.files)
        | beam.FlatMap(self._emit_tokenized_examples)
        | beam.Reshuffle())  # Allows for additional parallelization.


class WriteExampleTfRecord(beam.PTransform):
  """Writes examples (dicts) to a TFRecord of tf.Example protos."""

  def __init__(self, output_path, num_shards=None):
    """WriteExampleTfRecord constructor.

    Args:
      output_path: string, path to the output TFRecord file (w/o shard suffix).
      num_shards: (optional) int, number of shards to output or None to use
        liquid sharding.
    """
    self._output_path = output_path
    self._num_shards = num_shards

  def expand(self, pcoll):
    return (
        pcoll
        | beam.Map(t5.data.dict_to_tfexample)
        | beam.Reshuffle()
        | beam.io.tfrecordio.WriteToTFRecord(
            self._output_path,
            num_shards=self._num_shards,
            coder=beam.coders.ProtoCoder(tf.train.Example)))


class WriteJson(beam.PTransform):
  """Writes datastructures to file as JSON(L)."""

  def __init__(self, output_path, prettify=True):
    """WriteJson constructor.

    Args:
      output_path: string, path to the output JSON(L) file.
      prettify: bool, whether to write the outputs with sorted keys and
        indentation. Note this not be used if there are multiple records being
        written to the file (JSONL).
    """
    self._output_path = output_path
    self._prettify = prettify

  def _jsonify(self, el):
    if self._prettify:
      return json.dumps(el, sort_keys=True, indent=2)
    else:
      return json.dumps(el)

  def expand(self, pcoll):
    return (
        pcoll
        | beam.Map(self._jsonify)
        | "write_info" >> beam.io.WriteToText(
            self._output_path,
            num_shards=1,
            shard_name_template=""))


class GetInfo(beam.PTransform):
  """Computes info for dataset examples.

  Expects a single PCollections of examples.
  Returns a dictionary with information needed to read the data (number of
  shards, feature shapes and types)
  """

  def __init__(self, num_shards):
    self._num_shards = num_shards

  def _info_dict(self, ex):
    if not ex:
      return {}
    assert len(ex) == 1
    ex = ex[0]
    info = {"num_shards": self._num_shards, "features": {}}
    feature_dict = info["features"]
    for k, v in ex.items():
      t = tf.constant(v)
      # Change int32 to int64 since the tf.Example proto will store it this way.
      dtype = "int64" if t.dtype.name == "int32" else t.dtype.name
      shape = [None] * len(t.shape)
      feature_dict[k] = {"shape": shape, "dtype": dtype}
    return info

  def expand(self, pcoll):
    return (
        pcoll
        | beam.combiners.Sample.FixedSizeGlobally(1)
        | beam.Map(self._info_dict))


class GetStats(beam.PTransform):
  """Computes stastistics for dataset examples.

  Expects a dictionary of string identifiers mapped to PCollections of examples.
  Returns a dictionary with statistics (number of examples, number of tokens)
  prefixed by the identifiers.
  """

  def __init__(self, output_features):
    self._output_features = output_features

  def expand(self, pcoll):
    to_dict = lambda x: {x[0]: x[1]}
    example_counts = (
        pcoll
        | "count_examples" >> beam.combiners.Count.Globally()
        | "key_example_counts" >> beam.Map(
            lambda x: ("examples", x))
        | "example_count_dict" >> beam.Map(to_dict))
    def _count_tokens(pcoll, feat):
      return (
          pcoll
          | "key_%s_toks" % feat >> beam.Map(
              lambda x:  # pylint:disable=g-long-lambda
              ("%s_tokens" % feat, int(sum(x[feat] > 1)) if feat in x else 0)))
    token_counts = (
        [_count_tokens(pcoll, feat)
         for feat in self._output_features]
        | "flatten_tokens" >> beam.Flatten()
        | "count_tokens" >> beam.CombinePerKey(sum)
        | "token_count_dict" >> beam.Map(to_dict))

    def _merge_dicts(dicts):
      merged_dict = {}
      for d in dicts:
        assert not set(merged_dict).intersection(d)
        merged_dict.update(d)
      return merged_dict
    return (
        [example_counts, token_counts]
        | "flatten_counts" >> beam.Flatten()
        | "merge_stats" >> beam.CombineGlobally(_merge_dicts))


def run_pipeline(
    pipeline, task_names, cache_dir, max_input_examples=None,
    excluded_tasks=None, modules_to_import=(), overwrite=False):
  """Run preprocess pipeline."""
  output_dirs = []
  # Includes all names by default.
  included_regex = re.compile(r"(%s\Z)" % r"\Z|".join(task_names or [".*"]))
  # Excludes only empty names by default.
  excluded_regex = re.compile(r"(%s\Z)" % r"\Z|".join(excluded_tasks or []))
  task_names = [
      t for t in t5.data.TaskRegistry.names()
      if included_regex.match(t) and not excluded_regex.match(t)]
  for task_name in task_names:
    task = t5.data.TaskRegistry.get(task_name)
    if not isinstance(task, t5.data.TfdsTask):
      # TODO(adarob): Add support for not TfdsTasks.
      logging.info("Skipping non-`TfdsTask`: '%s'", task.name)
      continue

    task_cache_dir = task.cache_dir
    output_dir = os.path.join(cache_dir, task.name)

    if task_cache_dir and not overwrite:
      logging.info("Skipping task '%s', which exists in cache dir: %s",
                   task.name, task_cache_dir)
      continue

    if task_cache_dir and overwrite:
      if task_cache_dir == output_dir:
        # We were asked to overwrite the data, and the given directory that we
        # should generate the data in already has the data, then delete it.
        logging.warning(
            "Overwriting already cached data for task '%s' in cache_dir %s",
            task.name, output_dir)
        tf.io.gfile.rmtree(output_dir)
      else:
        # Cannot overwrite, since cache_dir isn't same as task.cache_dir.
        logging.warning("Not overwriting data in task.cache_dir since it is "
                        "different from cache_dir - %s vs %s", task.cache_dir,
                        output_dir)
        continue

    if not task.splits:
      logging.warning("Skipping task '%s' with no splits.", task.name)
      continue

    # Log this task to the terminal.
    print("Caching task '%s' with splits: %s" % (task.name, task.splits))

    output_dirs.append(output_dir)

    for split in task.splits:
      label = "%s_%s" % (task.name, split)
      pat = PreprocessAndTokenize(
          task, split, max_input_examples, modules_to_import)
      num_shards = len(pat.files)
      examples = (
          pipeline
          | "%s_pat" % label >> pat)
      _ = (examples
           | "%s_write_tfrecord" % label >> WriteExampleTfRecord(
               t5.data.get_tfrecord_prefix(output_dir, split),
               # Liquid sharding produces uneven shards, so use the same
               # number of shards as input.
               num_shards=num_shards))
      _ = (
          examples
          | "%s_info" % label >> GetInfo(num_shards)
          | "%s_write_info" % label >> WriteJson(
              t5.data.get_info_path(output_dir, split)))
      _ = (
          examples
          | "%s_stats" % label >> GetStats(task.output_features)
          | "%s_write_stats" % label >> WriteJson(
              t5.data.get_stats_path(output_dir, split)))
  return output_dirs


def main(_):
  flags.mark_flags_as_required(["output_cache_dir"])

  _import_modules(FLAGS.module_import)

  if FLAGS.tasks_additional_cache_dirs:
    t5.data.add_global_cache_dirs(
        [FLAGS.output_cache_dir] + FLAGS.tasks_additional_cache_dirs)

  output_dirs = []
  pipeline_options = beam.options.pipeline_options.PipelineOptions(
      FLAGS.pipeline_options)
  with beam.Pipeline(options=pipeline_options) as pipeline:
    tf.io.gfile.makedirs(FLAGS.output_cache_dir)
    output_dirs = run_pipeline(
        pipeline, FLAGS.tasks, FLAGS.output_cache_dir,
        FLAGS.max_input_examples, FLAGS.excluded_tasks, FLAGS.module_import,
        FLAGS.overwrite)

  # TODO(adarob): Figure out a way to write these when each task completes.
  for output_dir in output_dirs:
    with tf.io.gfile.GFile(os.path.join(output_dir, "COMPLETED"), "w") as f:
      f.write("")


def console_entry_point():
  app.run(main)

if __name__ == "__main__":
  console_entry_point()
