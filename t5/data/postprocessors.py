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

"""Functions which process model output strings to make them ready for eval."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf

# Note: postprocessors must either accept an `example` and `is_target` kwargs
# or include `**unused_kwargs` in their signature. The `example` will be the
# full example.


def string_to_float(string, default=-1., **unused_kwargs):
  """Converts string to float, using default when conversion not possible."""
  try:
    return float(string)
  except ValueError:
    return default


def as_text(string, **unused_kwargs):
  """Converts bytes or string to a string."""
  return tf.compat.as_text(string)


def string_label_to_class_id(
    string_label, label_classes, default=-1, **unused_kwargs):
  """Returns index of string_label in label_classes or default if not found."""
  string_label = tf.compat.as_text(string_label)
  if string_label in label_classes:
    return label_classes.index(string_label)
  else:
    return default


def multirc(string_label, example=None, is_target=False):
  """Returns dict containing the class with the question index for grouping."""
  res = {
      "value": string_label_to_class_id(
          string_label, example=example, label_classes=("False", "True"))
  }
  # Add the group, if present, since the model outputs will not have it.
  if is_target:
    res["group"] = example["idx/question"]
  return res


def qa(answer, example=None, is_target=False):
  """Returns answer, or all answers if the full example is provided."""
  if is_target:
    return example["answers"]
  return answer


def span_qa(answer, example=None, is_target=False):
  """Returns answer, or a dict with answers and context if the example is provided."""

  if is_target:
    return {"answers": example["answers"], "context": example["context"]}

  return answer


def wsc_simple(prediction, example=None, is_target=False):
  """Sees whether we predicted the referent or not."""
  if is_target:
    return example["label"]

  determiners = {
      "a", "an", "few", "her", "his", "each", "every", "many", "much", "my",
      "our", "some", "that", "the", "their", "these", "this", "those", "which",
      "whose", "your"
  }

  def clean(s):
    """Ignore capitalization and determiners."""
    s = s.strip().lower()
    return " ".join([w for w in s.split(" ") if w not in determiners])

  # We aren't using the label but rather using the extracted referent so that we
  # can see if the prediction is equivalent to the referent.
  referent = clean(example["targets_plaintext"].decode("utf-8"))
  prediction = clean(prediction)

  if ("'" in prediction) != ("'" in referent):
    # Make sure we don't mark cases where the prediction is "Bob" and the
    # referent is "Bob's hat" as predicting the referent.
    predicted_referent = False
  else:
    prediction_words = set(prediction.split(" "))
    referent_words = set(referent.split(" "))

    # Handle cases where the prediction is "fuzzy bunny" and the referent is
    # "bunny".
    predicted_referent = prediction_words.issubset(
        referent_words) or referent_words.issubset(prediction_words)

  return int(predicted_referent)
