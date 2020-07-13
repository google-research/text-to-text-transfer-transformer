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

"""Tests for t5.models.mesh_transformer."""

from absl.testing import absltest
import t5.data
from t5.data import test_utils
from t5.models import mesh_transformer
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds

tf.disable_v2_behavior()
tf.enable_eager_execution()


class MeshDatasetFnsTest(test_utils.FakeMixtureTest):

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
    vocabulary = t5.data.MixtureRegistry.get(mixture_name).get_vocabulary()
    sequence_length = {"inputs": 13, "targets": 13}
    output = dataset_fn(
        mixture_name,
        sequence_length=sequence_length,
        vocabulary=vocabulary,
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
      self.assertEqual([], metric_fns)
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

  def test_maybe_shuffle_and_subsample_dataset_no_shuffle(self):
    ds = tf.data.Dataset.range(100)

    num_eval_examples = 10
    shuffle_eval_examples = False
    num_repeat = 2
    ds = mesh_transformer.maybe_shuffle_and_subsample_dataset(
        ds, num_eval_examples, shuffle_eval_examples)
    ds = ds.repeat(num_repeat)

    list_examples = list(tfds.as_numpy(ds))

    # Assert on the number of examples.
    self.assertLen(list_examples, num_eval_examples * num_repeat)
    # Since `shuffle_eval_examples` is false, we will get the same examples
    # repeated `num_repeat` times.
    # Ex: [0, 1, 2, 3, 0, 1, 2, 3]
    self.assertEqual(list_examples, list(range(num_eval_examples)) * num_repeat)

  def test_maybe_shuffle_and_subsample_dataset_shuffle(self):
    ds = tf.data.Dataset.range(100)

    num_eval_examples = 10
    shuffle_eval_examples = True
    num_repeat = 2
    ds = mesh_transformer.maybe_shuffle_and_subsample_dataset(
        ds, num_eval_examples, shuffle_eval_examples,
        num_repeat * num_eval_examples)  # shuffle buffer size.
    ds = ds.repeat(num_repeat)

    list_examples = list(tfds.as_numpy(ds))

    # With high probability, not every slice of `num_eval_examples` in
    # `list_examples` will be the same.
    self.assertNotEqual(list_examples[:num_eval_examples],
                        list_examples[num_eval_examples:2 * num_eval_examples])

if __name__ == "__main__":
  absltest.main()
