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

from t5.data.dataset_providers import *  # pylint:disable=wildcard-import
from t5.data.feature_converters import *  # pylint:disable=wildcard-import
from t5.data.glue_utils import *  # pylint:disable=wildcard-import
import t5.data.postprocessors
import t5.data.preprocessors
from t5.data.test_utils import assert_dataset
from t5.data.utils import *  # pylint:disable=wildcard-import
from t5.data.vocabularies import *  # pylint:disable=wildcard-import
