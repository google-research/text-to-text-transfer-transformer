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

# Lint as: python3
"""Abstract Vocabulary."""

import abc
import gin


@gin.configurable
class Vocabulary(object):
  """Base class for all vocabularies."""

  def __init__(self, extra_ids=0):
    self._extra_ids = extra_ids

  @abc.abstractmethod
  def encode(self, s):
    raise NotImplementedError

  @abc.abstractmethod
  def decode(self, ids):
    raise NotImplementedError

  @abc.abstractmethod
  def encode_tf(self, s):
    raise NotImplementedError

  @abc.abstractmethod
  def decode_tf(self, ids):
    raise NotImplementedError

  @property
  def extra_ids(self):
    return self._extra_ids
