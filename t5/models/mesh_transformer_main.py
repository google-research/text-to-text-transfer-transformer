# Copyright 2019 The T5 Authors.
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

r"""Main file for launching training/eval/inference of mesh-transformer model.

#TODO(sharannarang): Update instructions for running the model in the README.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
import os
import sys

from absl import app
from absl import flags
import gin
from mesh_tensorflow.transformer import utils
import pkg_resources
import t5
import tensorflow.compat.v1 as tf

flags.DEFINE_string(
    "tpu_job_name", None,
    "Name of TPU worker binary. Only necessary if job name is changed from "
    "default tpu_worker.")
flags.DEFINE_string(
    "model_dir", "/tmp/transformer_standalone", "Estimator model_dir")


flags.DEFINE_string(
    "tpu", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.")

flags.DEFINE_string(
    "gcp_project",
    None,
    "Project name for the Cloud TPU-enabled project. If not specified, we "
    "will attempt to automatically detect the GCE project from metadata.")

flags.DEFINE_string(
    "tpu_zone", None,
    "GCE zone where the Cloud TPU is located in. If not specified, we "
    "will attempt to automatically detect the GCE project from metadata.")

# TFDS Module Import
flags.DEFINE_multi_string(
    "module_import", None,
    "Modules to import. Use this when your DatasetBuilder is defined outside "
    "of tensorflow_datasets so that it is registered.")

flags.DEFINE_multi_string(
    "tasks_cache_dir", None,
    "Directories to search for cached Tasks.")

FLAGS = flags.FLAGS


def main(_):
  if FLAGS.module_import:
    for module in FLAGS.module_import:
      importlib.import_module(module)

  if FLAGS.tasks_cache_dir:
    t5.data.set_global_cache_dirs(FLAGS.tasks_cache_dir)

  # Add search path for gin files stored in package.
  gin.add_config_file_search_path(
      pkg_resources.resource_filename(__name__, "gin"))

  tf.io.gfile.makedirs(FLAGS.model_dir)
  suffix = 0
  command_filename = os.path.join(FLAGS.model_dir, "command")
  while tf.io.gfile.exists(command_filename):
    suffix += 1
    command_filename = os.path.join(
        FLAGS.model_dir, "command.{}".format(suffix))
  with tf.io.gfile.GFile(command_filename, "w") as f:
    f.write(" ".join(sys.argv))

  utils.parse_gin_defaults_and_flags()
  utils.run(
      tpu_job_name=FLAGS.tpu_job_name,
      tpu=FLAGS.tpu,
      gcp_project=FLAGS.gcp_project,
      tpu_zone=FLAGS.tpu_zone,
      model_dir=FLAGS.model_dir)


def console_entry_point():
  tf.disable_v2_behavior()
  tf.logging.set_verbosity(tf.logging.INFO)
  app.run(main)

if __name__ == "__main__":
  console_entry_point()
