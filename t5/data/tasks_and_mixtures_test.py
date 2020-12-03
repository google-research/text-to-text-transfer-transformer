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
"""Tests for creating tasks and mixtures."""

from absl import flags
from absl import logging
from absl.testing import absltest
from absl.testing import parameterized

from t5.data import dataset_providers
import t5.data.mixtures  # pylint:disable=unused-import
import t5.data.tasks  # pylint:disable=unused-import

FLAGS = flags.FLAGS

MixtureRegistry = dataset_providers.MixtureRegistry
TaskRegistry = dataset_providers.TaskRegistry
_SEQUENCE_LENGTH = {'inputs': 1024, 'targets': 512}

flags.DEFINE_bool(
    'load_data', False,
    'Whether to test loading data in addition to creating task.')


class TasksAndMixturesTest(parameterized.TestCase):

  @parameterized.parameters(((name,) for name in TaskRegistry.names()))
  def test_task(self, name):
    logging.info('task=%s', name)
    task = TaskRegistry.get(name)
    if FLAGS.load_data:
      ds = task.get_dataset(_SEQUENCE_LENGTH, 'train')
      for d in ds:
        logging.info(d)
        break

  @parameterized.parameters(((name,) for name in MixtureRegistry.names()))
  def test_mixture(self, name):
    logging.info('mixture=%s', name)
    mixture = MixtureRegistry.get(name)
    if FLAGS.load_data:
      ds = mixture.get_dataset(_SEQUENCE_LENGTH, 'train')
      for d in ds:
        logging.info(d)
        break


if __name__ == '__main__':
  absltest.main()
