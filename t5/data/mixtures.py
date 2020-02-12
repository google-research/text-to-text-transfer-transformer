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

"""Add Mixtures to the registry.

This module contains different mixtures for training T5 models.
"""

import functools

import t5.data.tasks  # pylint: disable=unused-import
from t5.data.utils import MixtureRegistry
from t5.data.utils import rate_num_examples
from t5.data.utils import rate_unsupervised
import tensorflow_datasets as tfds


# We omit WNLI because we train on WSC/DPR simple instead
_glue_tasks = [
    "glue_%s_v002" % b.name
    for b in tfds.text.glue.Glue.builder_configs.values()
    if "wnli" not in b.name
]

_wsc_dpr_tasks = [
    "dpr_v001_simple",
    "super_glue_wsc_v102_simple_train",
    "super_glue_wsc_v102_simple_eval",
]
_super_glue_tasks = _wsc_dpr_tasks + [
    "super_glue_%s_v102" % b.name
    for b in tfds.text.super_glue.SuperGlue.builder_configs.values()
    if "wsc" not in b.name
]

_supervised_tasks = (
    _glue_tasks + _super_glue_tasks +
    ["cnn_dailymail_v002",
     "squad_v010_allanswers",
     "wmt_t2t_ende_v003",
     "wmt15_enfr_v003",
     "wmt16_enro_v003"]
)

# Tasks is not the best description. For example, glue_v002_equal refers to a
# mixture. Calling it "finetune tasks" because we consider all glue tasks as
# a single dataset to train on.
_finetune_tasks = [
    "glue_v002_proportional",  # mixture
    "super_glue_v102_proportional",  # mixture
    "cnn_dailymail_v002",
    "squad_v010_allanswers",
    "wmt_t2t_ende_v003",
    "wmt15_enfr_v003",
    "wmt16_enro_v003"
]

# ========================== GLUE and SuperGLUE ================================

MixtureRegistry.add(
    "glue_v002_proportional",
    _glue_tasks, default_rate=rate_num_examples)


MixtureRegistry.add(
    "super_glue_v102_proportional",
    _super_glue_tasks,
    default_rate=rate_num_examples)


# mnli and its associated dev sets: mnli_matched and mnli_mismatched
MixtureRegistry.add(
    "glue_mnli_and_dev_v002",
    [t for t in _glue_tasks if "mnli" in t],
    default_rate=1.0)

# ============================== Co-training ===================================


# C4, glue, squad, superglue
#  The supervised tasks here are all small datasets
#  Mix them proportionally to their dataset sizes.
# TODO(noam): This should be called "small_mix" or something, but we will
#   keep it as en_mix to avoid restarting experiments.
# TODO(noam): some rates should be reduced - but not now to avoid restarting
#     experiments.   They are:
#  - Tasks duplicated between glue and superglue (see _dedupe)
#  - squad and glue_qnli are duplicates
#  - glue_sst2 may contain overlapping phrases (related examples with itself)
#  - we seem to overtrain on super_glue_record - don't know why
MixtureRegistry.add(
    "en_mix",
    [("c4_v020_unsupervised", rate_unsupervised)] +
    _glue_tasks + _super_glue_tasks +
    ["squad_v010_allanswers"],
    default_rate=rate_num_examples)

MixtureRegistry.add(
    "all_equal",
    _supervised_tasks + ["c4_v020_unsupervised"],
    default_rate=1.,
)


def _dedupe(name):
  if "glue" in name and "rte" in name:
    return functools.partial(rate_num_examples, scale=0.5)
  return rate_num_examples

MixtureRegistry.add(
    "all_proportional",
    [(t, _dedupe(t)) for t in _supervised_tasks + ["c4_v020_unsupervised"]],
)

# all_mix is the same as all_proportional except it uses rate_unsupervised
# for c4_v020_unsupervised. This is useful if you want to specify a specific
# rate for the unsupervised task which is different from the global value for
# rate_num_examples.maximum
# If you use this task, you should set a maximum rate value via gin e.g.
# --gin_param="dataset_utils.rate_num_examples.maximum = 1e6"
MixtureRegistry.add(
    "all_mix",
    ([("c4_v020_unsupervised", rate_unsupervised)] +
     [(t, _dedupe(t)) for t in _supervised_tasks]),
)

# ================== Leave-one-out cotrain then finetune =======================

for task_name in _finetune_tasks:
  task_names = set(_supervised_tasks + ["c4_v020_unsupervised"])

  # Special case to treat all GLUE tasks as one task.
  if task_name == "glue_v002_proportional":
    task_names -= set(_glue_tasks)
    # No de-duping needed
    tasks = [(t, rate_num_examples) for t in task_names]
  # Special case to treat all Super GLUE tasks as one task.
  elif task_name == "super_glue_v102_proportional":
    task_names -= set(_super_glue_tasks)
    # No de-duping needed
    tasks = [(t, rate_num_examples) for t in task_names]
  else:
    task_names -= {task_name}
    # Use de-duping since we have GLUE and SuperGLUE
    tasks = [(t, _dedupe(t)) for t in task_names]

  MixtureRegistry.add("leave_one_out_{}".format(task_name), tasks)

# ================= Pre-train on supervised tasks ==============================

_large_translation_tasks = ["wmt_t2t_ende_v003",
                            "wmt15_enfr_v003"]

_large_supervised_tasks = _large_translation_tasks + ["cnn_dailymail_v002"]

MixtureRegistry.add(
    "large_supervised_equal",
    _large_supervised_tasks,
    default_rate=1.0)

MixtureRegistry.add(
    "large_supervised_proportional",
    _large_supervised_tasks,
    default_rate=rate_num_examples)

MixtureRegistry.add(
    "large_translation_equal",
    _large_translation_tasks,
    default_rate=1.0)

# =========================== Squad + Trivia QA ================================
MixtureRegistry.add(
    "squad_trivia_qa_equal",
    ["squad_v010_allanswers", "trivia_qa_v010"],
    default_rate=1.0)

# ================================= WSC + DPR ==================================
MixtureRegistry.add(
    "wsc_dpr_simple_proportional",
    _wsc_dpr_tasks,
    default_rate=rate_num_examples)
