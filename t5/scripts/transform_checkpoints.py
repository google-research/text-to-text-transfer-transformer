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

r"""Script to transform one or more checkpoints into a new checkpoint.

One operation supported is to average several checkpoints.

Other operations deal with ensemble models.  An ensemble is represented as
a single model where all variables (other than the step number) have an extra
leading dimension.

Operations supported are:
  average: take the arithmetic mean of multiple models
  ensemble: create an ensemble from multiple (similarly shaped) checkpoints
  autoensemble: create an ensemble of identical models from one checkpoint
    (broadcast each variable to give it a leading dimension)
  extract_first: extract the first element of an ensemble checkpoint
  average_last_n: average the last n checkpoints in the given model dir

This script takes a list of model directories as inputs and picks the latest
checkpoint from each of the directories. If ckpt files are provided, it will use
them directly instead of looking for the latest checkpoint.


The global step of the output checkpoint is set to either the value of the
global_step flag (if nonzero) or the global step in the first input checkpoint.
"""

import os
import re
from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf

FLAGS = flags.FLAGS

flags.DEFINE_list("model_dirs_or_checkpoints", [],
                  "Model directories or checkpoints to use for ensembling.")
flags.DEFINE_string("output_dir", "/tmp/",
                    "Directory to output the ensembled checkpoint to.")
flags.DEFINE_integer(
    "global_step", 0,
    "Global step to use for writing the output checkpoint file.")
flags.DEFINE_enum(
    "operation",
    "average",
    ["average", "ensemble", "autoensemble", "extract_first", "average_last_n"],
    "what to do to the input checkpoints to produce the output checkpoints")

flags.DEFINE_integer(
    "autoensemble_size", 4, "ensemble size for 'autoensemble'")

flags.DEFINE_integer("number_of_checkpoints", 4,
                     "number of last checkpoints for 'average_last_n'")


def average_tensors(tensors):
  result = tensors[0]
  for t in tensors[1:]:
    result += t
  return result / len(tensors)


def main(_):
  assert FLAGS.model_dirs_or_checkpoints

  if not tf.io.gfile.exists(FLAGS.output_dir):
    tf.io.gfile.makedirs(FLAGS.output_dir)

  if (FLAGS.operation == "average_last_n" and
      len(FLAGS.model_dirs_or_checkpoints) > 1):
    raise ValueError("Need only 1 directory for %s operation" % FLAGS.operation)

  checkpoints = []

  for path in FLAGS.model_dirs_or_checkpoints:
    if tf.io.gfile.isdir(path):
      # Grab the latest checkpoint for all the provided model dirs
      checkpoint_state = tf.train.get_checkpoint_state(path)
      if FLAGS.operation == "average_last_n":
        ckpt_paths = tf.io.gfile.glob(os.path.join(path, "model.ckpt*index"))
        def sort_fn(ckpt):
          return int(re.sub(".*ckpt-", "", ckpt))

        ckpts = sorted([c.replace(".index", "") for c in ckpt_paths],
                       key=sort_fn)
        checkpoints.extend(ckpts[-FLAGS.number_of_checkpoints:])
      else:
        checkpoints.append(checkpoint_state.all_model_checkpoint_paths[-1])
    else:
      if FLAGS.operation == "average_last_n":
        raise ValueError("need a directory while running %s operation" %
                         FLAGS.operation)
      checkpoints.append(path)

  logging.info("Using checkpoints %s", checkpoints)

  if FLAGS.operation in ["ensemble", "average", "average_last_n"]:
    if len(checkpoints) == 1:
      raise ValueError("no point in ensebling/averaging one checkpoint")
  else:
    if len(checkpoints) != 1:
      raise ValueError(
          "operation %s requires exactly one checkpoint" % FLAGS.operation)

  var_values = {}
  var_dtypes = {}

  for i in range(0, len(checkpoints)):
    checkpoint = checkpoints[i]
    logging.info("loading checkpoint %s", checkpoint)
    reader = tf.train.load_checkpoint(checkpoint)
    var_list = tf.train.list_variables(checkpoint)
    for (name, _) in var_list:
      if i:
        assert name in var_values
        tensor = reader.get_tensor(name)
        assert tensor.dtype == var_dtypes[name]
        var_values[name].append(tensor)
      else:
        tensor = reader.get_tensor(name)
        var_dtypes[name] = tensor.dtype
        var_values[name] = [tensor]
        if not FLAGS.global_step:
          if name == "global_step":
            FLAGS.global_step = tensor

    logging.info("Read from checkpoint %s", checkpoint)

  new_var_values = {}

  # stack the list of tensors along the 0th dimension.
  for name, tensors in var_values.items():
    tensor = tensors[0]
    if name == "global_step":
      new_val = np.int32(FLAGS.global_step)
    elif FLAGS.operation == "ensemble":
      new_val = np.stack(tensors)
    elif FLAGS.operation == "autoensemble":
      new_val = np.stack([tensor] * FLAGS.autoensemble_size)
    elif FLAGS.operation == "average" or FLAGS.operation == "average_last_n":
      new_val = average_tensors(tensors)
    elif FLAGS.operation == "extract_first":
      new_val = tensor[0]
    else:
      raise ValueError("unknown FLAGS.operation=%s" % FLAGS.operation)
    new_var_values[name] = new_val

  var_values = new_var_values

  with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
    tf_vars = [
        tf.get_variable(v, shape=var_values[v].shape, dtype=var_dtypes[v])
        for v in var_values
    ]

  placeholders = [tf.placeholder(v.dtype, shape=v.shape) for v in tf_vars]
  assign_ops = [tf.assign(v, p) for (v, p) in zip(tf_vars, placeholders)]
  saver = tf.train.Saver(tf.all_variables())

  output_file = "model.ckpt-" + str(FLAGS.global_step)
  output_path = os.path.join(FLAGS.output_dir, output_file)

  # Build a model consisting only of variables, set them to the average values.
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for p, assign_op, (name, value) in zip(
        placeholders, assign_ops, var_values.items()):
      sess.run(assign_op, {p: value})
    # Use the built saver to save the averaged checkpoint.
    saver.save(sess, output_path)

  logging.info("Transformed checkpoints saved in %s", output_path)


if __name__ == "__main__":
  tf.disable_v2_behavior()
  app.run(main)
