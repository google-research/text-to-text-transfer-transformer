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

r"""Parse TensorBoard events files and (optionally) log results to csv.


Note that `--summary_dir` *must* point directly to the directory with .events
files (e.g. `/validation_eval/`), not a parent directory.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging

from t5.evaluation import eval_utils

FLAGS = flags.FLAGS

flags.DEFINE_string("summary_dir", None, "Where to search for .events files.")
flags.DEFINE_string("out_file", None, "Output file to write TSV.")


def main(_):
  events = eval_utils.parse_events_files(FLAGS.summary_dir)
  scores = eval_utils.get_eval_metric_values(events)
  if not scores:
    logging.info("No evaluation events found in %s", FLAGS.summary_dir)
    return
  df = eval_utils.scores_to_df(scores)
  df = eval_utils.compute_avg_glue(df)
  df = eval_utils.sort_columns(df)
  eval_utils.log_csv(df, output_file=FLAGS.out_file)

if __name__ == "__main__":
  app.run(main)
