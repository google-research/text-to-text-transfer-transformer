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

"""Import data modules."""
# pylint:disable=wildcard-import,g-bad-import-order
from t5.data.dataset_providers import *
from t5.data.glue_utils import *
import t5.data.postprocessors
import t5.data.preprocessors
from t5.data.utils import *

# For backward compatibility
# TODO(adarob): Remove need for these imports.
from seqio.dataset_providers import *
from t5.data.dataset_providers import TaskRegistry
from t5.data.dataset_providers import FunctionTask as Task
from seqio.test_utils import assert_dataset
from seqio.utils import *
from seqio.vocabularies import *
