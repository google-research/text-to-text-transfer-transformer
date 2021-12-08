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

# Lint as: python3
r"""Utility to print the text or tokens in a task.

Example usage:
python -m t5.scripts.dump_task \
    --task=glue_mnli_v002 \
    --max_examples=100


"""

import importlib
import re

from absl import app
from absl import flags

# from mesh_tensorflow.transformer import utils
import gin
import seqio

import tensorflow.compat.v1 as tf
tf.compat.v1.enable_eager_execution()

try:
  tf.flags.DEFINE_multi_string("gin_file", None, "Path to a Gin file.")
  tf.flags.DEFINE_multi_string("gin_param", None, "Gin parameter binding.")
  tf.flags.DEFINE_list("gin_location_prefix", [], "Gin file search path.")
except tf.flags.DuplicateFlagError:
  pass


_DEFAULT_MODULE_IMPORTS = [
]

FLAGS = flags.FLAGS

flags.DEFINE_string("task", None,
                    "A registered Task.")
flags.DEFINE_string("mixture", None,
                    "A registered Mixture.")
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

flags.DEFINE_bool("detokenize", False, "If True, then decode ids to strings.")

flags.DEFINE_bool("shuffle", True, "Whether to shuffle dataset or not.")
flags.DEFINE_bool("apply_postprocess_fn", False,
                  "Whether to apply the postprocess function or not.")


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

  # Load gin parameters if they've been defined.
  try:
    for gin_file_path in FLAGS.gin_location_prefix:
      gin.add_config_file_search_path(gin_file_path)
    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
  except AttributeError:
    # Otherwise, use default settings.
    gin.parse_config_files_and_bindings(None, None)

  total_examples = 0
  if FLAGS.task is not None:
    task_or_mixture = seqio.TaskRegistry.get(FLAGS.task)
  elif FLAGS.mixture is not None:
    task_or_mixture = seqio.MixtureRegistry.get(FLAGS.mixture)

  ds = task_or_mixture.get_dataset(
      sequence_length=sequence_length(),
      split=FLAGS.split,
      use_cached=False,
      shuffle=FLAGS.shuffle)

  keys = re.findall(r"{([\w+]+)}", FLAGS.format_string)
  def _example_to_string(ex):
    key_to_string = {}
    for k in keys:
      if k in ex:
        v = ex[k].numpy().tolist()
        if (FLAGS.detokenize
            and v and isinstance(v, list)
            and isinstance(v[0], int)):
          s = task_or_mixture.output_features[k].vocabulary.decode(
              [abs(i) for i in v])
          if (FLAGS.apply_postprocess_fn and k == "targets"
              and hasattr(task_or_mixture, "postprocess_fn")):
            s = task_or_mixture.postprocess_fn(s)
        elif isinstance(v, bytes):
          s = v.decode("utf-8")
        else:
          s = " ".join(str(i) for i in v)
        key_to_string[k] = s
      else:
        key_to_string[k] = ""
    return FLAGS.format_string.format(**key_to_string)

  for ex in ds:
    print(_example_to_string(ex))
    total_examples += 1
    if total_examples == FLAGS.max_examples:
      break
  return

if __name__ == "__main__":
  app.run(main)
