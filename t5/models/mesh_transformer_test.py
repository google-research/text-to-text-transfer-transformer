# Copyright 2022 The T5 Authors.
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

"""Tests for t5.models.mesh_transformer."""

from absl.testing import absltest
import seqio
from t5.models import mesh_transformer
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds

tf.enable_eager_execution()


class MeshDatasetFnsTest(seqio.test_utils.FakeMixtureTest):

  def check_ds_shape(self, ds, sequence_length):
    for k, v in tf.data.get_output_shapes(ds).items():
      feat = k.split("_")[0]
      if len(v) == 0:  # pylint:disable=g-explicit-length-test
        expected_shape = []
      elif feat in sequence_length:
        expected_shape = [sequence_length[feat]]
      else:
        expected_shape = [None]
      self.assertEqual(expected_shape, v.as_list())

  def verify_mesh_dataset_fn(self, mixture_name, train, use_cached):
    if train:
      dataset_fn = mesh_transformer.mesh_train_dataset_fn
      split = tfds.Split.TRAIN
    else:
      dataset_fn = mesh_transformer.mesh_eval_dataset_fn
      split = tfds.Split.VALIDATION
    sequence_length = {"inputs": 13, "targets": 13}
    output = dataset_fn(
        mixture_name,
        sequence_length=sequence_length,
        dataset_split=split,
        use_cached=use_cached)
    if train:
      ds = output
      self.check_ds_shape(ds, sequence_length)
      # Materialize a few batches to test for errors.
      list(zip(range(10), tfds.as_numpy(ds)))
    else:
      self.assertLen(output, 1)
      output = output[0]
      (name, dsfn, postprocess_fn, metric_fns) = output
      self.assertEqual("cached_task" if use_cached else "uncached_task", name)
      ds = dsfn()
      self.check_ds_shape(ds, sequence_length)
      # No postprocess_fn is supplied so it should function as a pass-through
      self.assertEqual("test", postprocess_fn("test"))
      # test_utils task has empty metric_fns list
      self.assertEmpty(metric_fns)
      # Materialize the full dataset to test for errors.
      list(tfds.as_numpy(ds))

  def test_mesh_train_dataset_fn(self):
    self.verify_mesh_dataset_fn(
        mixture_name="cached_mixture", train=True, use_cached=True,
    )
    self.verify_mesh_dataset_fn(
        mixture_name="uncached_mixture", train=True, use_cached=False,
    )

  def test_mesh_eval_dataset_fn(self):
    self.verify_mesh_dataset_fn(
        mixture_name="cached_mixture", train=False, use_cached=True,
    )
    self.verify_mesh_dataset_fn(
        mixture_name="uncached_mixture", train=False, use_cached=False,
    )

if __name__ == "__main__":
  absltest.main()
