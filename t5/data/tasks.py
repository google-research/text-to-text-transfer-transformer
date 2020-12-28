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
import functools

from t5.data import postprocessors
from t5.data import preprocessors
from t5.data.dataset_providers import Feature
from t5.data.dataset_providers import TaskRegistry
from t5.data.dataset_providers import TfdsTask
from t5.data.glue_utils import get_glue_metric
from t5.data.glue_utils import get_glue_postprocess_fn
from t5.data.glue_utils import get_glue_text_preprocessor
from t5.data.glue_utils import get_super_glue_metric
from t5.data.utils import get_default_vocabulary
from t5.data.utils import set_global_cache_dirs
from t5.evaluation import metrics
import tensorflow_datasets as tfds



DEFAULT_OUTPUT_FEATURES = {
    "inputs": Feature(
        vocabulary=get_default_vocabulary(), add_eos=True, required=False),
    "targets": Feature(vocabulary=get_default_vocabulary(), add_eos=True)
}

# ==================================== C4 ======================================
# Final pretraining task used in Raffel et al., 2019.
TaskRegistry.add(
    "c4_v220_span_corruption",
    TfdsTask,
    tfds_name="c4/en:2.2.0",
    text_preprocessor=functools.partial(
        preprocessors.rekey, key_map={"inputs": None, "targets": "text"}),
    token_preprocessor=preprocessors.span_corruption,
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[])


# Baseline pretraining task used in Raffel et al., 2019.
TaskRegistry.add(
    "c4_v220_iid_denoising",
    TfdsTask,
    tfds_name="c4/en:2.2.0",
    text_preprocessor=functools.partial(
        preprocessors.rekey, key_map={"inputs": None, "targets": "text"}),
    token_preprocessor=preprocessors.iid_denoising,
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[])


# Prefix language modeling pretraining task used in Raffel et al., 2019.
TaskRegistry.add(
    "c4_v220_prefix_lm",
    TfdsTask,
    tfds_name="c4/en:2.2.0",
    text_preprocessor=functools.partial(
        preprocessors.rekey, key_map={"inputs": None, "targets": "text"}),
    token_preprocessor=preprocessors.prefix_lm,
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[])


# Configurable tasks used for comparisons in Raffel et al., 2019.
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
      output_features=DEFAULT_OUTPUT_FEATURES,
      metric_fns=[])


# ================================ Wikipedia ===================================
TaskRegistry.add(
    "wikipedia_20190301.en_v003_unsupervised",
    TfdsTask,
    tfds_name="wikipedia/20190301.en:1.0.0",
    text_preprocessor=functools.partial(
        preprocessors.rekey, key_map={"inputs": None, "targets": "text"}),
    token_preprocessor=preprocessors.unsupervised,
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[])


# =================================== GLUE =====================================
for b in tfds.text.glue.Glue.builder_configs.values():
  TaskRegistry.add(
      "glue_%s_v002" % b.name,
      TfdsTask,
      tfds_name="glue/%s:1.0.0" % b.name,
      text_preprocessor=get_glue_text_preprocessor(b),
      metric_fns=get_glue_metric(b.name),
      output_features=DEFAULT_OUTPUT_FEATURES,
      postprocess_fn=get_glue_postprocess_fn(b),
      splits=["test"] if b.name == "ax" else None,
  )

# =============================== CNN DailyMail ================================
TaskRegistry.add(
    "cnn_dailymail_v002",
    TfdsTask,
    tfds_name="cnn_dailymail:1.0.0",
    text_preprocessor=functools.partial(preprocessors.summarize,
                                        article_key="article",
                                        summary_key="highlights"),
    metric_fns=[metrics.rouge],
    output_features=DEFAULT_OUTPUT_FEATURES)

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
      output_features=DEFAULT_OUTPUT_FEATURES)

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
    output_features=DEFAULT_OUTPUT_FEATURES)

# ================================= SuperGlue ==================================
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
        get_glue_text_preprocessor(b)
    ]
  else:
    text_preprocessor = get_glue_text_preprocessor(b)
  TaskRegistry.add(
      "super_glue_%s_v102" % b.name,
      TfdsTask,
      tfds_name="super_glue/%s:1.0.2" % b.name,
      text_preprocessor=text_preprocessor,
      metric_fns=get_super_glue_metric(b.name),
      output_features=DEFAULT_OUTPUT_FEATURES,
      postprocess_fn=get_glue_postprocess_fn(b),
      splits=["test"] if b.name in ["axb", "axg"] else None)

# ======================== Definite Pronoun Resolution =========================
TaskRegistry.add(
    "dpr_v001_simple",
    TfdsTask,
    tfds_name="definite_pronoun_resolution:1.1.0",
    text_preprocessor=preprocessors.definite_pronoun_resolution_simple,
    metric_fns=[metrics.accuracy],
    output_features=DEFAULT_OUTPUT_FEATURES)

# =================================== WSC ======================================
TaskRegistry.add(
    "super_glue_wsc_v102_simple_train",
    TfdsTask,
    tfds_name="super_glue/wsc.fixed:1.0.2",
    text_preprocessor=functools.partial(
        preprocessors.wsc_simple, correct_referent_only=True),
    metric_fns=[],
    output_features=DEFAULT_OUTPUT_FEATURES,
    splits=["train"])
TaskRegistry.add(
    "super_glue_wsc_v102_simple_eval",
    TfdsTask,
    tfds_name="super_glue/wsc.fixed:1.0.2",
    text_preprocessor=functools.partial(
        preprocessors.wsc_simple, correct_referent_only=False),
    postprocess_fn=postprocessors.wsc_simple,
    metric_fns=[metrics.accuracy],
    output_features=DEFAULT_OUTPUT_FEATURES,
    splits=["validation", "test"])

# =================================== WNLI =====================================
TaskRegistry.add(
    "glue_wnli_v002_simple_eval",
    TfdsTask,
    tfds_name="glue/wnli:1.0.0",
    text_preprocessor=preprocessors.wnli_simple,
    postprocess_fn=postprocessors.wsc_simple,
    metric_fns=[metrics.accuracy],
    output_features=DEFAULT_OUTPUT_FEATURES,
    splits=["validation", "test"])

# =================================== Squad ====================================
# Maximized evaluation metrics over all answers.
TaskRegistry.add(
    "squad_v010_allanswers",
    TfdsTask,
    tfds_name="squad/v1.1:2.0.0",
    text_preprocessor=preprocessors.squad,
    postprocess_fn=postprocessors.qa,
    metric_fns=[metrics.squad],
    output_features=DEFAULT_OUTPUT_FEATURES)


# Maximized evaluation metrics over all answers.
TaskRegistry.add(
    "squad_v010_context_free",
    TfdsTask,
    tfds_name="squad/v1.1:2.0.0",
    text_preprocessor=functools.partial(
        preprocessors.squad, include_context=False),
    postprocess_fn=postprocessors.qa,
    metric_fns=[metrics.squad],
    output_features=DEFAULT_OUTPUT_FEATURES)

# Squad span prediction task instead of text.
TaskRegistry.add(
    "squad_v010_allanswers_span",
    TfdsTask,
    tfds_name="squad/v1.1:2.0.0",
    text_preprocessor=preprocessors.squad_span_space_tokenized,
    postprocess_fn=postprocessors.span_qa,
    metric_fns=[metrics.span_squad],
    output_features=DEFAULT_OUTPUT_FEATURES)

# Deprecated: Use `squad_v010_allanswers` instead.
TaskRegistry.add(
    "squad_v010",
    TfdsTask,
    tfds_name="squad/v1.1:2.0.0",
    text_preprocessor=preprocessors.squad,
    metric_fns=[metrics.squad],
    output_features=DEFAULT_OUTPUT_FEATURES)

# ================================= TriviaQA ===================================
TaskRegistry.add(
    "trivia_qa_v010",
    TfdsTask,
    tfds_name="trivia_qa/rc:1.1.0",
    text_preprocessor=preprocessors.trivia_qa,
    metric_fns=[],
    token_preprocessor=preprocessors.trivia_qa_truncate_inputs,
    output_features=DEFAULT_OUTPUT_FEATURES)
