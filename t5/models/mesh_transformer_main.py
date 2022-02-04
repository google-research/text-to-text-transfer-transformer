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

r"""Main file for launching training/eval/predictions of mesh-transformer model."""

import importlib
import os
import sys

from absl import app
from absl import flags
from absl import logging
import gin
from mesh_tensorflow.transformer import utils
import pkg_resources
import t5
from t5.models import mesh_transformer
from t5.models import mtf_model
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

flags.DEFINE_multi_string(
    "module_import", "t5.data.mixtures",
    "Modules to import. Use this, for example, to add new `Task`s to the "
    "global `TaskRegistry`.")

flags.DEFINE_string(
    "t5_tfds_data_dir", None,
    "If set, this directory will be used to store datasets prepared by "
    "TensorFlow Datasets that are not available in the public TFDS GCS bucket. "
    "Note that this flag overrides the `tfds_data_dir` attribute of all "
    "`Task`s.")

flags.DEFINE_list(
    "additional_task_cache_dirs", [],
    "Directories to search for Tasks in addition to defaults.")

flags.DEFINE_boolean("use_model_api", False,
                     "Use Model API instead of utils.run.")

flags.DEFINE_list("additional_deprecated_gin_references", [],
                  "Deprecated gin configs to be ignored.")

flags.DEFINE_boolean("skip_all_gin_unknowns", False,
                     "Don't throw any errors if any gin config params are "
                     "not found. Overrides the specific list of names in "
                     "--additional_deprecated_gin_references and "
                     "DEPRECATED_GIN_REFERENCES.")

# Note: All the args from here on are only used when use_model_api is set
flags.DEFINE_enum(
    "mode", None, ["train", "finetune", "eval", "predict",
                   "export_predict", "export_score", "score"],
    "Mode with which to run the model.")
flags.DEFINE_integer("batch_size", 1,
                     "Number of sequences per batch.")
flags.DEFINE_integer("input_sequence_length", 512,
                     "Number of tokens in input sequence.")
flags.DEFINE_integer("target_sequence_length", 512,
                     "Number of tokens in target sequence.")

# TPU-specific args.
flags.DEFINE_string("tpu_topology", "v2-8",
                    "The TPU topology being used. Ignored if --tpu not set.")
flags.DEFINE_integer("model_parallelism", 8,
                     "The number of cores per model replica. Ignored if --tpu "
                     "not set.")

# Train mode args
flags.DEFINE_integer("train_steps", 1000, "Number of training iterations.")
flags.DEFINE_string("mixture_or_task", "wmt_t2t_ende_v003",
                    "Name of Mixture or Task to use for training/evaluation.")
flags.DEFINE_string("pretrained_model_dir", "",
                    "Pretrained model dir for finetuning a model.")

# Eval mode args
flags.DEFINE_enum(
    "checkpoint_mode", "latest", ["all", "latest", "specific"],
    "Checkpoint steps to use when running 'eval', 'predict', 'finetune', and "
    "'export' modes. Can specify a list of checkpoints or all or the latest "
    "checkpoint. 'finetune' and 'export' modes work with 'latest' or "
    "'specific' with a single checkpoint.")
flags.DEFINE_list(
    "checkpoint_steps", [],
    "Checkpoint step numbers used for 'eval', 'predict', and 'finetune' modes. "
    "This argument is only used when which_checkpoint='specific'. "
    "For the 'finetune' mode, only a single checkpoint must be specified.")

flags.DEFINE_string("eval_summary_dir", "", "Path to save eval summaries")
flags.DEFINE_string("eval_split", "validation",
                    "Dataset split to use for evaluation.")

# Predict mode args
flags.DEFINE_string("input_file", "",
                    "Path to input file for decoding or scoring.")
flags.DEFINE_string("target_file", "", "Path to target file for scoring.")
flags.DEFINE_string("output_file", "", "Path to output file to save decodes.")

# Export mode args
flags.DEFINE_string(
    "export_dir", "",
    "Directory to export SavedModels to. Will use `model_dir` if unspecified.")


# Decoding strategy args, used in export and predict modes.
flags.DEFINE_integer("beam_size", 1, "Beam size for predict or export mode.")
flags.DEFINE_float("temperature", 0.0,
                   "Sampling emperature for predict or export mode.")
flags.DEFINE_integer("keep_top_k", -1,
                     "Top-k value for predict or export mode.")

FLAGS = flags.FLAGS


def main(_):
  if FLAGS.module_import:
    for module in FLAGS.module_import:
      importlib.import_module(module)

  if FLAGS.t5_tfds_data_dir:
    t5.data.set_tfds_data_dir_override(FLAGS.t5_tfds_data_dir)

  # Add search path for gin files stored in package.
  gin.add_config_file_search_path(
      pkg_resources.resource_filename(__name__, "gin"))
  try:
    suffix = 0
    command_dir = os.path.join(FLAGS.model_dir, "commands")
    tf.io.gfile.makedirs(command_dir)
    command_filename = os.path.join(command_dir, "command")
    while tf.io.gfile.exists(command_filename):
      suffix += 1
      command_filename = os.path.join(command_dir, "command.{}".format(suffix))
    with tf.io.gfile.GFile(command_filename, "w") as f:
      f.write(" ".join(sys.argv))
  except (tf.errors.PermissionDeniedError, tf.errors.InvalidArgumentError):
    logging.info(
        "No write access to model directory. Skipping command logging.")

  utils.parse_gin_defaults_and_flags(
      skip_unknown=(FLAGS.skip_all_gin_unknowns or (
          mesh_transformer.DEPRECATED_GIN_REFERENCES +
          tuple(FLAGS.additional_deprecated_gin_references))),
      finalize_config=False)
  # We must overide this binding explicitly since it is set to a deprecated
  # function or class in many existing configs.
  gin.bind_parameter("run.vocabulary", mesh_transformer.get_vocabulary())
  gin.finalize()

  # Set cache dir after loading gin to avoid unintentionally overriding it.
  t5.data.add_global_cache_dirs(FLAGS.additional_task_cache_dirs)

  if FLAGS.use_model_api:
    model = mtf_model.MtfModel(
        tpu_job_name=FLAGS.tpu_job_name,
        tpu=FLAGS.tpu,
        gcp_project=FLAGS.gcp_project,
        tpu_zone=FLAGS.tpu_zone,
        tpu_topology=FLAGS.tpu_topology,
        model_parallelism=FLAGS.model_parallelism,
        model_dir=FLAGS.model_dir,
        batch_size=FLAGS.batch_size,
        sequence_length={"inputs": FLAGS.input_sequence_length,
                         "targets": FLAGS.target_sequence_length}
    )

    if FLAGS.checkpoint_mode != "specific" and FLAGS.checkpoint_steps:
      raise ValueError("checkpoint_mode is set to %s and checkpoint_steps is "
                       "also set. To use a particular checkpoint, please set "
                       "checkpoint_mode to 'specific'. For other modes, please "
                       "ensure that checkpoint_steps is not set."
                       % FLAGS.checkpoint_mode)

    if FLAGS.checkpoint_mode == "latest":
      checkpoint_steps = -1
    elif FLAGS.checkpoint_mode == "all":
      checkpoint_steps = "all"
    else:
      checkpoint_steps = [int(c) for c in FLAGS.checkpoint_steps]

    if FLAGS.mode == "train":
      model.train(mixture_or_task_name=FLAGS.mixture_or_task,
                  steps=FLAGS.train_steps)
    elif FLAGS.mode == "eval":
      model.eval(mixture_or_task_name=FLAGS.mixture_or_task,
                 checkpoint_steps=checkpoint_steps,
                 summary_dir=FLAGS.eval_summary_dir,
                 split=FLAGS.eval_split)
    elif FLAGS.mode == "finetune":
      if not (FLAGS.checkpoint_mode == "latest" or
              (FLAGS.checkpoint_mode == "specific" and
               len(FLAGS.checkpoint_steps) == 1)):
        raise ValueError(
            "Must specify a single checkpoint for finetuning a model.")

      if isinstance(checkpoint_steps, list):
        checkpoint_steps = checkpoint_steps[0]

      model.finetune(
          mixture_or_task_name=FLAGS.mixture_or_task,
          steps=FLAGS.train_steps,
          pretrained_model_dir=FLAGS.pretrained_model_dir,
          checkpoint_steps=checkpoint_steps)
    elif FLAGS.mode == "predict":
      model.predict(
          checkpoint_steps=checkpoint_steps,
          input_file=FLAGS.input_file,
          output_file=FLAGS.output_file,
          beam_size=FLAGS.beam_size,
          temperature=FLAGS.temperature,
          keep_top_k=FLAGS.keep_top_k,)
    elif FLAGS.mode == "score":
      model.score(
          FLAGS.input_file,
          FLAGS.target_file,
          scores_file=FLAGS.output_file,
          checkpoint_steps=checkpoint_steps)
    elif FLAGS.mode in ("export_predict", "export_score"):
      if not (FLAGS.checkpoint_mode == "latest" or
              (FLAGS.checkpoint_mode == "specific" and
               len(FLAGS.checkpoint_steps) == 1)):
        raise ValueError(
            "Must specify a single checkpoint for exporting a model.")

      if isinstance(checkpoint_steps, list):
        checkpoint_steps = checkpoint_steps[0]

      model.export(
          export_dir=FLAGS.export_dir,
          checkpoint_step=checkpoint_steps,
          beam_size=FLAGS.beam_size,
          temperature=FLAGS.temperature,
          keep_top_k=FLAGS.keep_top_k,
          eval_with_score=(FLAGS.mode == "export_score"))
    else:
      raise ValueError("--mode flag must be set when using Model API.")
  else:
    if FLAGS.mode:
      raise ValueError("--mode flag should only be set when using Model API.")
    if not FLAGS.tpu:
      with gin.unlock_config():
        gin.bind_parameter("utils.get_variable_dtype.slice_dtype", "float32")
        gin.bind_parameter(
            "utils.get_variable_dtype.activation_dtype", "float32")
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
