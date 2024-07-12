# Copyright 2024 The T5 Authors.
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

"""Utilities for GLUE and SuperGLUE tasks."""

import collections
import functools

import seqio
from t5.data import postprocessors
from t5.data import preprocessors
from t5.evaluation import metrics
import tensorflow_datasets as tfds

# These weights are based on the number of examples in each dataset.
SUPER_GLUE_WEIGHT_MAPPING = {
    "dpr_v001_simple": 1_322.,
    "super_glue_wsc_v102_simple_train": 259.,
    "super_glue_wsc_v102_simple_eval": 0.,
    "super_glue_boolq_v102": 9_427.,
    "super_glue_cb_v102": 250.,
    "super_glue_copa_v102": 400.,
    "super_glue_multirc_v102": 27_243.,
    "super_glue_record_v102": 138_854.,
    "super_glue_rte_v102": 2_490.,
    "super_glue_wic_v102": 5_428.,
    "super_glue_axb_v102": 0.,
    "super_glue_axg_v102": 0.,
}

# Mappings for the SuperGLUE tasks with sentinel tokens added.
SUPER_GLUE_WEIGHT_MAPPING_SENTINEL = {
    "dpr_v001_simple_1_sentinel": 1_322.,
    "super_glue_wsc_v102_simple_1_sentinel_train": 259.,
    "super_glue_wsc_v102_simple_1_sentinel_eval": 0.,
    "super_glue_boolq_v102_1_sentinel": 9_427.,
    "super_glue_cb_v102_1_sentinel": 250.,
    "super_glue_copa_v102_1_sentinel": 400.,
    "super_glue_multirc_v102_1_sentinel": 27_243.,
    "super_glue_record_v102_1_sentinel": 138_854.,
    "super_glue_rte_v102_1_sentinel": 2_490.,
    "super_glue_wic_v102_1_sentinel": 5_428.,
    "super_glue_axb_v102_1_sentinel": 0.,
    "super_glue_axg_v102_1_sentinel": 0.,
}

# These weights are based on the number of examples in each dataset.
# We omit WNLI because we train on WSC/DPR simple instead
GLUE_WEIGHT_MAPPING = {
    "glue_cola_v002": 8_551.,
    "glue_sst2_v002": 67_349.,
    "glue_mrpc_v002": 3_668.,
    "glue_qqp_v002": 363_849.,
    "glue_stsb_v002": 5_749.,
    "glue_mnli_v002": 392_702.,
    "glue_qnli_v002": 104_743.,
    "glue_rte_v002": 2_490.,
    "glue_mnli_mismatched_v002": 0.,
    "glue_mnli_matched_v002": 0.,
    "glue_ax_v002": 0.,
}


def get_glue_weight_mapping():
  return GLUE_WEIGHT_MAPPING


def get_super_glue_weight_mapping():
  return SUPER_GLUE_WEIGHT_MAPPING


def get_super_glue_weight_mapping_sentinel():
  return SUPER_GLUE_WEIGHT_MAPPING_SENTINEL


def get_glue_text_preprocessor(builder_config):
  """Return the glue preprocessor.

  Args:
    builder_config: a BuilderConfig

  Returns:
    a preprocessor function
  """
  # stsb uses a floating point target, so use special preprocessor
  if builder_config.name == "stsb":
    return preprocessors.stsb
  elif builder_config.name == "wsc.fixed":
    return preprocessors.wsc
  elif builder_config.name == "record":
    return preprocessors.record
  else:
    if "mnli" in builder_config.name or builder_config.name == "ax":
      # Cast the GLUE diagnostic task as MNLI.
      benchmark_name = "mnli"
    elif builder_config.name in ["axb", "axg"]:
      # Cast the SuperGLUE diagnostic tasks as RTE.
      benchmark_name = "rte"
    else:
      benchmark_name = builder_config.name
    if builder_config.name == "multirc":
      feature_names = ("question", "answer", "paragraph")
    elif builder_config.name == "wic":
      # This ignores the start/end indices which show where in each sentence the
      # word appears.
      # TODO(craffel): Investigate using those indices.
      feature_names = ("sentence1", "sentence2", "word")
    else:
      feature_names = None
    return functools.partial(
        preprocessors.glue,
        benchmark_name=benchmark_name,
        label_names=builder_config.label_classes,
        feature_names=feature_names)


def get_glue_postprocess_fn(builder_config):
  if builder_config.name == "stsb":
    return postprocessors.string_to_float
  elif builder_config.name == "multirc":
    return postprocessors.multirc
  elif builder_config.name == "record":
    return postprocessors.record
  else:
    return functools.partial(
        postprocessors.string_label_to_class_id,
        label_classes=builder_config.label_classes,
    )

GLUE_METRICS = collections.OrderedDict([
    ("cola", [metrics.sklearn_metrics_wrapper(
        "matthews_corrcoef", metric_post_process_fn=lambda x: 100 * x)]),
    ("sst2", [metrics.accuracy]),
    ("mrpc", [metrics.f1_score_with_invalid, metrics.accuracy]),
    ("stsb", [metrics.pearson_corrcoef, metrics.spearman_corrcoef]),
    ("qqp", [metrics.f1_score_with_invalid, metrics.accuracy]),
    ("mnli", [metrics.accuracy]),
    ("mnli_matched", [metrics.accuracy]),
    ("mnli_mismatched", [metrics.accuracy]),
    ("qnli", [metrics.accuracy]),
    ("rte", [metrics.accuracy]),
    ("wnli", [metrics.accuracy]),
    ("ax", []),  # Only test set available.
])


def get_glue_metric(task_name):
  return GLUE_METRICS[task_name]

SUPERGLUE_METRICS = collections.OrderedDict([
    ("boolq", [metrics.accuracy]),
    ("cb", [metrics.mean_multiclass_f1(num_classes=3), metrics.accuracy]),
    ("copa", [metrics.accuracy]),
    ("multirc", [
        metrics.multirc_f1_over_all_answers,
        metrics.mean_group_metric(metrics.all_match)
    ]),
    ("record", [metrics.deduplicate_metric(metrics.squad)]),
    ("rte", [metrics.accuracy]),
    ("wic", [metrics.accuracy]),
    ("axb", []),  # Only test set available.
    ("axg", []),  # Only test set available.
])


def get_super_glue_metric(task_name):
  return SUPERGLUE_METRICS[task_name]


def register_super_glue(vocab, add_inputs_eos, add_targets_eos):
  """Registers SeqIO Tasks and Mixture for SuperGlue given vocabulary."""
  output_features = {
      "inputs":
          seqio.Feature(
              vocabulary=vocab, add_eos=add_inputs_eos, required=False),
      "targets":
          seqio.Feature(vocabulary=vocab, add_eos=add_targets_eos)
  }

  # ================================ SuperGlue =================================

  for b in tfds.text.super_glue.SuperGlue.builder_configs.values():
    # We use a simplified version of WSC, defined below
    if "wsc" in b.name:
      continue
    if b.name == "axb":
      glue_preprocessors = [
          functools.partial(
              preprocessors.rekey,
              key_map={
                  "premise": "sentence1",
                  "hypothesis": "sentence2",
                  "label": "label",
                  "idx": "idx",
              }),
          get_glue_text_preprocessor(b),
          seqio.preprocessors.tokenize,
          seqio.CacheDatasetPlaceholder(),
          seqio.preprocessors.append_eos_after_trim,
      ]
    else:
      glue_preprocessors = [
          get_glue_text_preprocessor(b),
          seqio.preprocessors.tokenize,
          seqio.CacheDatasetPlaceholder(),
          seqio.preprocessors.append_eos_after_trim,
      ]
    seqio.TaskRegistry.add(
        "super_glue_%s_v102" % b.name,
        source=seqio.TfdsDataSource(
            tfds_name="super_glue/%s:1.0.2" % b.name,
            splits=["test"] if b.name in ["axb", "axg"] else None),
        preprocessors=glue_preprocessors,
        metric_fns=get_super_glue_metric(b.name),
        output_features=output_features,
        postprocess_fn=get_glue_postprocess_fn(b))

    # Create SuperGLUE tasks with 1 sentinel token added.
    seqio.experimental.add_task_with_sentinels("super_glue_%s_v102" % b.name,
                                               num_sentinels=1)

  # ======================= Definite Pronoun Resolution ========================
  seqio.TaskRegistry.add(
      "dpr_v001_simple",
      source=seqio.TfdsDataSource(
          tfds_name="definite_pronoun_resolution:1.1.0"),
      preprocessors=[
          preprocessors.definite_pronoun_resolution_simple,
          seqio.preprocessors.tokenize,
          seqio.CacheDatasetPlaceholder(),
          seqio.preprocessors.append_eos_after_trim,
      ],
      metric_fns=[metrics.accuracy],
      output_features=output_features)

  # Create SuperGLUE tasks with 1 sentinel token added.
  seqio.experimental.add_task_with_sentinels("dpr_v001_simple", num_sentinels=1)

  # ================================== WSC =====================================
  seqio.TaskRegistry.add(
      "super_glue_wsc_v102_simple_train",
      source=seqio.TfdsDataSource(
          tfds_name="super_glue/wsc.fixed:1.0.2", splits=["train"]),
      preprocessors=[
          functools.partial(
              preprocessors.wsc_simple, correct_referent_only=True),
          seqio.preprocessors.tokenize,
          seqio.CacheDatasetPlaceholder(),
          seqio.preprocessors.append_eos_after_trim,
      ],
      metric_fns=[],
      output_features=output_features)

  # Create SuperGLUE tasks with 1 sentinel token added.
  seqio.experimental.add_task_with_sentinels("super_glue_wsc_v102_simple_train",
                                             num_sentinels=1)

  seqio.TaskRegistry.add(
      "super_glue_wsc_v102_simple_eval",
      source=seqio.TfdsDataSource(
          tfds_name="super_glue/wsc.fixed:1.0.2", splits=["validation",
                                                          "test"]),
      preprocessors=[
          functools.partial(
              preprocessors.wsc_simple, correct_referent_only=False),
          seqio.preprocessors.tokenize,
          seqio.CacheDatasetPlaceholder(),
          seqio.preprocessors.append_eos_after_trim,
      ],
      postprocess_fn=postprocessors.wsc_simple,
      metric_fns=[metrics.accuracy],
      output_features=output_features)

  # Create SuperGLUE tasks with 1 sentinel token added.
  seqio.experimental.add_task_with_sentinels("super_glue_wsc_v102_simple_eval",
                                             num_sentinels=1)

  # ============================== Mixtures ====================================
  super_glue_tasks_with_weight = list(
      get_super_glue_weight_mapping().items())
  super_glue_tasks_with_weight_sentinel = list(
      get_super_glue_weight_mapping_sentinel().items())

  seqio.MixtureRegistry.add("super_glue_v102_proportional",
                            super_glue_tasks_with_weight)
  seqio.MixtureRegistry.add("super_glue_v102_proportional_sentinel",
                            super_glue_tasks_with_weight_sentinel)
