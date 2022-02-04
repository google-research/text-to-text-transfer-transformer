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

r"""Prepare a file for submission to the (Super)GLUE leaderboard.

This script assumes that you have already generated predictions on a given
split. The predictions should be saved line-by-line in a text file, and should
be written with postprocessing applied. This is the format that gets written out
when you run the Mesh TensorFlow Transformer in eval mode. Note that the order
of this line must exactly match the order of examples returned by loading the
splits from the task. So, for example, you should run eval on the cached test
set and then run this script with the split flag set to test and the cached flag
set to True.
"""

import ast
import collections
import csv
import json
import os

from absl import app
from absl import flags
import t5.data
import t5.data.tasks
import tensorflow as tf
import tensorflow_datasets as tfds

FLAGS = flags.FLAGS

flags.DEFINE_string("predictions_file", None, "Path to model predictions.")
flags.DEFINE_string("task", None, "T5 task name for this benchmark.")
flags.DEFINE_string("tfds_name", None, "Short name of tfds (e.g. 'cb').")
flags.DEFINE_string("out_dir", None, "Path to write output file.")
flags.DEFINE_string("split", "test", "Split, should typically be test.")
flags.DEFINE_boolean("super", False, "Whether to make SuperGLUE-style file.")
flags.DEFINE_boolean("cached", True, "Whether to used cached dataset.")
flags.DEFINE_list("additional_task_cache_dirs", [], "Dirs with cached tasks.")

FILE_NAME_MAP = {
    "boolq": "BoolQ",
    "cb": "CB",
    "copa": "COPA",
    "multirc": "MultiRC",
    "record": "ReCoRD",
    "rte": "RTE",
    "wic": "WiC",
    "cola": "CoLA",
    "sst2": "SST-2",
    "mrpc": "MRPC",
    "stsb": "STS-B",
    "qqp": "QQP",
    "mnli_matched": "MNLI-m",
    "mnli_mismatched": "MNLI-mm",
    "qnli": "QNLI",
    "wnli": "WNLI",
    "wsc": "WSC",
    "axb": "AX-b",
    "axg": "AX-g",
}
USES_TEXT = [
    "cb", "rte", "mnli_matched", "mnli_mismatched", "qnli", "axb", "axg"
]
# Placeholder for seq len - required by get_dataset but not used
_FAKE_LEN = {"inputs": 512, "targets": 512}


def main(_):
  t5.data.add_global_cache_dirs(FLAGS.additional_task_cache_dirs)
  out_file = os.path.join(
      FLAGS.out_dir, "{}.{{extension}}".format(FILE_NAME_MAP[FLAGS.tfds_name])
  )

  ds = t5.data.TaskRegistry.get_dataset(
      FLAGS.task, _FAKE_LEN, FLAGS.split, use_cached=FLAGS.cached, shuffle=False
  )
  examples = [{k: v.numpy() for k, v in ex.items()} for ex in ds]

  with tf.io.gfile.GFile(FLAGS.predictions_file) as f:
    prediction_lines = f.readlines()
  if FLAGS.tfds_name == "record":
    # record just uses raw strings
    predictions = [l.strip() for l in prediction_lines]
  else:
    # everything else uses Python code strings
    predictions = [ast.literal_eval(l.strip()) for l in prediction_lines]

  if FLAGS.tfds_name in USES_TEXT:
    if FLAGS.super:
      builder_configs = tfds.text.super_glue.SuperGlue.builder_configs
    else:
      builder_configs = tfds.text.glue.Glue.builder_configs
    label_classes = builder_configs[FLAGS.tfds_name].label_classes
    predictions = [label_classes[p] for p in predictions]
  elif FLAGS.tfds_name in ["boolq", "wic"]:
    predictions = [("false", "true")[p] for p in predictions]
  elif FLAGS.tfds_name == "wsc":
    predictions = [("False", "True")[p] for p in predictions]
  elif FLAGS.tfds_name == "multirc":
    # multirc is so different from the rest that we special-case everything
    rows = collections.defaultdict(lambda: collections.defaultdict(dict))
    predictions = [int(p["value"]) for p in predictions]
    for p, e in zip(predictions, examples):
      e = {k: int(e["idx/" + k]) for k in ["paragraph", "question", "answer"]}
      rows[e["paragraph"]][e["question"]][e["answer"]] = p
    with tf.io.gfile.GFile(out_file.format(extension="jsonl"), "w") as f:
      for pidx, passage in rows.items():
        qs = [
            {"idx": i, "answers": [{"idx": j, "label": q[j]} for j in q]}
            for i, q in passage.items()
        ]
        f.write(
            json.dumps({"idx": pidx, "passage": {"questions": qs}}) + os.linesep
        )
    return

  if len(predictions) != len(examples):
    raise ValueError(
        "Number of predictions in {} ({}) != of examples in {} split of {} "
        "({}).".format(
            FLAGS.predictions_file,
            len(predictions),
            FLAGS.split,
            FLAGS.task,
            len(examples),
        )
    )

  if "record" in FLAGS.task:
    indices = [ex["idx/query"] for ex in examples]
  else:
    indices = [ex.get("idx", None) for ex in examples]

  if FLAGS.super:
    lines = [
        json.dumps({"idx": int(i), "label": p}) + os.linesep
        for i, p in zip(indices, predictions)
    ]
    with tf.io.gfile.GFile(out_file.format(extension="jsonl"), "w") as f:
      for line in lines:
        f.write(line)
  else:
    with tf.io.gfile.GFile(out_file.format(extension="tsv"), "w") as out_file:
      tsv_writer = csv.writer(out_file, delimiter="\t")
      tsv_writer.writerow(["index", "prediction"])
      tsv_writer.writerows([i, p] for i, p in zip(indices, predictions))

if __name__ == "__main__":
  app.run(main)
