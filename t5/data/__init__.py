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

"""Import data modules."""

import t5.data.mixtures
import t5.data.postprocessors
import t5.data.preprocessors
# For backwards compatibility with an old import path
import t5.data.sentencepiece_vocabulary
# For backwards compatibility with an old import path
from t5.data.sentencepiece_vocabulary import SentencePieceVocabulary
import t5.data.tasks
import t5.data.test_utils
from t5.data.utils import *  # pylint:disable=wildcard-import
import t5.data.vocabularies
