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
r"""Utility to print the text or tokens in a task.

Example usage:
python -m t5.scripts.dump_task \
    --task=glue_mnli_v002 \
    --max_examples=100


"""

import importlib

from absl import app
from absl import flags
from absl import logging

# from mesh_tensorflow.transformer import utils
import gin
import t5
import tensorflow.compat.v1 as tf


_DEFAULT_MODULE_IMPORTS = [
]

FLAGS = flags.FLAGS

flags.DEFINE_string("task", None,
                    "A registered Task.")
flags.DEFINE_integer("max_examples", -1,
                     "maximum number of examples. -1 for no limit")
flags.DEFINE_string(
    "format_string",
    "{inputs}\t{targets}", "format for printing examples")
flags.DEFINE_multi_string(
    "module_import", _DEFAULT_MODULE_IMPORTS,
    "Modules to import. Use this when your Task or is defined outside "
    "of the T5 codebase so that it is registered.")
flags.DEFINE_string(
    "split", "train", "which split of the dataset, e.g. train or validation")
flags.DEFINE_bool(
    "tokenize", False, "tokenize and print out token ids")


@gin.configurable
def sequence_length(value=512):
  """Sequence length used when tokenizing.

  Args:
    value: an integer or dictionary
  Returns:
    a dictionary
  """
  if isinstance(value, int):
    return {"inputs": value, "targets": value}
  else:
    return value


def import_modules(modules):
  for module in modules:
    importlib.import_module(module)


def main(_):
  flags.mark_flags_as_required(["task"])

  if FLAGS.module_import:
    import_modules(FLAGS.module_import)

  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)

  total_examples = 0
  tf.enable_eager_execution()
  task = t5.data.TaskRegistry.get(FLAGS.task)
  files = task.tfds_dataset.files(FLAGS.split)
  def _example_to_string(ex):
    key_to_string = {}
    for k in ("inputs", "targets"):
      if k in ex:
        v = ex[k].numpy()
        key_to_string[k] = (
            " ".join(str(i) for i in v) if FLAGS.tokenize
            else v.decode("utf-8"))
      else:
        key_to_string[k] = ""
    return FLAGS.format_string.format(**key_to_string)

  for shard_path in files:
    logging.info("Processing shard: %s", shard_path)
    ds = task.tfds_dataset.load_shard(shard_path)
    ds = task.preprocess_text(ds)
    if FLAGS.tokenize:
      ds = t5.data.encode_string_features(
          ds, task.output_features, keys=task.output_features,
          copy_plaintext=True)
      ds = task.preprocess_tokens(ds, sequence_length())

    for ex in ds:
      print(_example_to_string(ex))
      total_examples += 1
      if total_examples == FLAGS.max_examples:
        return


if __name__ == "__main__":
  tf.disable_v2_behavior()
  app.run(main)
