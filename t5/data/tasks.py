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

"""Add Tasks to registry."""
import collections
import functools

from t5.data import postprocessors
from t5.data import preprocessors
from t5.data.utils import DEFAULT_SPM_PATH
from t5.data.utils import set_global_cache_dirs
from t5.data.utils import TaskRegistry
from t5.data.utils import TfdsTask
from t5.evaluation import metrics
import tensorflow_datasets as tfds




# ==================================== C4 ======================================
_c4_config_suffixes = ["", ".noclean", ".realnewslike", ".webtextlike"]
for config_suffix in _c4_config_suffixes:
  TaskRegistry.add(
      "c4{name}_v020_unsupervised".format(
          name=config_suffix.replace(".", "_")),
      TfdsTask,
      tfds_name="c4/en{config}:2.2.0".format(config=config_suffix),
      text_preprocessor=functools.partial(
          preprocessors.rekey, key_map={"inputs": None, "targets": "text"}),
      token_preprocessor=preprocessors.unsupervised,
      sentencepiece_model_path=DEFAULT_SPM_PATH,
      metric_fns=[])

# ================================ Wikipedia ===================================
TaskRegistry.add(
    "wikipedia_20190301.en_v003_unsupervised",
    TfdsTask,
    tfds_name="wikipedia/20190301.en:1.0.0",
    text_preprocessor=functools.partial(
        preprocessors.rekey, key_map={"inputs": None, "targets": "text"}),
    token_preprocessor=preprocessors.unsupervised,
    sentencepiece_model_path=DEFAULT_SPM_PATH,
    metric_fns=[])


# =================================== GLUE =====================================
def _get_glue_text_preprocessor(builder_config):
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


def _get_glue_postprocess_fn(builder_config):
  if builder_config.name == "stsb":
    return postprocessors.string_to_float
  elif builder_config.name == "multirc":
    return postprocessors.multirc
  elif builder_config.name == "record":
    return postprocessors.qa
  else:
    return functools.partial(
        postprocessors.string_label_to_class_id,
        label_classes=builder_config.label_classes,
    )


GLUE_METRICS = collections.OrderedDict([
    ("cola", [metrics.matthews_corrcoef]),
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

for b in tfds.text.glue.Glue.builder_configs.values():
  TaskRegistry.add(
      "glue_%s_v002" % b.name,
      TfdsTask,
      tfds_name="glue/%s:1.0.0" % b.name,
      text_preprocessor=_get_glue_text_preprocessor(b),
      metric_fns=GLUE_METRICS[b.name],
      sentencepiece_model_path=DEFAULT_SPM_PATH,
      postprocess_fn=_get_glue_postprocess_fn(b),
      splits=["test"] if b.name == "ax" else None,
  )

# =============================== CNN DailyMail ================================
TaskRegistry.add(
    "cnn_dailymail_v002",
    TfdsTask,
    tfds_name="cnn_dailymail/plain_text:1.0.0",
    text_preprocessor=functools.partial(preprocessors.summarize,
                                        article_key="article",
                                        summary_key="highlights"),
    metric_fns=[metrics.rouge],
    sentencepiece_model_path=DEFAULT_SPM_PATH)

# ==================================== WMT =====================================
# Format: year, tfds builder config, tfds version
b_configs = [
    ("14", tfds.translate.wmt14.Wmt14Translate.builder_configs["de-en"], "1.0.0"
    ),
    ("14", tfds.translate.wmt14.Wmt14Translate.builder_configs["fr-en"], "1.0.0"
    ),
    ("16", tfds.translate.wmt16.Wmt16Translate.builder_configs["ro-en"], "1.0.0"
    ),
    ("15", tfds.translate.wmt15.Wmt15Translate.builder_configs["fr-en"], "1.0.0"
    ),
    ("19", tfds.translate.wmt19.Wmt19Translate.builder_configs["de-en"], "1.0.0"
    ),
]

for prefix, b, tfds_version in b_configs:
  TaskRegistry.add(
      "wmt%s_%s%s_v003" % (prefix, b.language_pair[1], b.language_pair[0]),
      TfdsTask,
      tfds_name="wmt%s_translate/%s:%s" % (prefix, b.name, tfds_version),
      text_preprocessor=functools.partial(
          preprocessors.translate,
          source_language=b.language_pair[1],
          target_language=b.language_pair[0],
          ),
      metric_fns=[metrics.bleu],
      sentencepiece_model_path=DEFAULT_SPM_PATH)

# Special case for t2t ende.
b = tfds.translate.wmt_t2t.WmtT2tTranslate.builder_configs["de-en"]
TaskRegistry.add(
    "wmt_t2t_ende_v003",
    TfdsTask,
    tfds_name="wmt_t2t_translate/de-en:1.0.0",
    text_preprocessor=functools.partial(
        preprocessors.translate,
        source_language=b.language_pair[1],
        target_language=b.language_pair[0],
        ),
    metric_fns=[metrics.bleu],
    sentencepiece_model_path=DEFAULT_SPM_PATH)

# ================================= SuperGlue ==================================
SUPERGLUE_METRICS = collections.OrderedDict([
    ("boolq", [metrics.accuracy]),
    ("cb", [
        metrics.mean_multiclass_f1(num_classes=3),
        metrics.accuracy
    ]),
    ("copa", [metrics.accuracy]),
    ("multirc", [
        metrics.multirc_f1_over_all_answers,
        metrics.mean_group_metric(metrics.exact_match)
    ]),
    ("record", [metrics.squad]),
    ("rte", [metrics.accuracy]),
    ("wic", [metrics.accuracy]),
    ("axb", []),  # Only test set available.
    ("axg", []),  # Only test set available.
])

for b in tfds.text.super_glue.SuperGlue.builder_configs.values():
  # We use a simplified version of WSC, defined below
  if "wsc" in b.name:
    continue
  if b.name == "axb":
    text_preprocessor = [
        functools.partial(
            preprocessors.rekey,
            key_map={
                "premise": "sentence1",
                "hypothesis": "sentence2",
                "label": "label",
                "idx": "idx",
            }),
        _get_glue_text_preprocessor(b)
    ]
  else:
    text_preprocessor = _get_glue_text_preprocessor(b)
  TaskRegistry.add(
      "super_glue_%s_v102" % b.name,
      TfdsTask,
      tfds_name="super_glue/%s:1.0.2" % b.name,
      text_preprocessor=text_preprocessor,
      metric_fns=SUPERGLUE_METRICS[b.name],
      sentencepiece_model_path=DEFAULT_SPM_PATH,
      postprocess_fn=_get_glue_postprocess_fn(b),
      splits=["test"] if b.name in ["axb", "axg"] else None)

# ======================== Definite Pronoun Resolution =========================
TaskRegistry.add(
    "dpr_v001_simple",
    TfdsTask,
    tfds_name="definite_pronoun_resolution/plain_text:1.0.0",
    text_preprocessor=preprocessors.definite_pronoun_resolution_simple,
    metric_fns=[metrics.accuracy],
    sentencepiece_model_path=DEFAULT_SPM_PATH)

# =================================== WSC ======================================
TaskRegistry.add(
    "super_glue_wsc_v102_simple_train",
    TfdsTask,
    tfds_name="super_glue/wsc.fixed:1.0.2",
    text_preprocessor=functools.partial(
        preprocessors.wsc_simple, correct_referent_only=True),
    metric_fns=[],
    sentencepiece_model_path=DEFAULT_SPM_PATH,
    splits=["train"])
TaskRegistry.add(
    "super_glue_wsc_v102_simple_eval",
    TfdsTask,
    tfds_name="super_glue/wsc.fixed:1.0.2",
    text_preprocessor=functools.partial(
        preprocessors.wsc_simple, correct_referent_only=False),
    postprocess_fn=postprocessors.wsc_simple,
    metric_fns=[metrics.accuracy],
    sentencepiece_model_path=DEFAULT_SPM_PATH,
    splits=["validation", "test"])

# =================================== WNLI =====================================
TaskRegistry.add(
    "glue_wnli_v002_simple_eval",
    TfdsTask,
    tfds_name="glue/wnli:1.0.0",
    text_preprocessor=preprocessors.wnli_simple,
    postprocess_fn=postprocessors.wsc_simple,
    metric_fns=[metrics.accuracy],
    sentencepiece_model_path=DEFAULT_SPM_PATH,
    splits=["validation", "test"])

# =================================== Squad ====================================
# Maximized evaluation metrics over all answers.
TaskRegistry.add(
    "squad_v010_allanswers",
    TfdsTask,
    tfds_name="squad/plain_text:1.0.0",
    text_preprocessor=preprocessors.squad,
    postprocess_fn=postprocessors.qa,
    metric_fns=[metrics.squad],
    sentencepiece_model_path=DEFAULT_SPM_PATH)

# Maximized evaluation metrics over all answers.
TaskRegistry.add(
    "squad_v010_context_free",
    TfdsTask,
    tfds_name="squad/plain_text:1.0.0",
    text_preprocessor=functools.partial(
        preprocessors.squad, include_context=False),
    postprocess_fn=postprocessors.qa,
    metric_fns=[metrics.squad],
    sentencepiece_model_path=DEFAULT_SPM_PATH)

# Squad span prediction task instead of text.
TaskRegistry.add(
    "squad_v010_allanswers_span",
    TfdsTask,
    tfds_name="squad/plain_text:1.0.0",
    text_preprocessor=preprocessors.squad_span_space_tokenized,
    postprocess_fn=postprocessors.span_qa,
    metric_fns=[metrics.span_squad],
    sentencepiece_model_path=DEFAULT_SPM_PATH)

# Deprecated: Use `squad_v010_allanswers` instead.
TaskRegistry.add(
    "squad_v010",
    TfdsTask,
    tfds_name="squad/plain_text:1.0.0",
    text_preprocessor=preprocessors.squad,
    metric_fns=[metrics.squad],
    sentencepiece_model_path=DEFAULT_SPM_PATH)

# ================================= TriviaQA ===================================
TaskRegistry.add(
    "trivia_qa_v010",
    TfdsTask,
    tfds_name="trivia_qa:1.1.0",
    text_preprocessor=preprocessors.trivia_qa,
    metric_fns=[],
    token_preprocessor=preprocessors.trivia_qa_truncate_inputs,
    sentencepiece_model_path=DEFAULT_SPM_PATH)
