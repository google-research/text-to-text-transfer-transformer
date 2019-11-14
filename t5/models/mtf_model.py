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

# Lint as: python3
"""Mesh Tensorflow T5 Model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import gin
import gin.tf

import mesh_tensorflow as mtf

from mesh_tensorflow import optimize
from mesh_tensorflow.transformer import learning_rate_schedules
from mesh_tensorflow.transformer import utils

from t5.data.sentencepiece_vocabulary import SentencePieceVocabulary
from t5.models.mesh_transformer import mesh_eval_dataset_fn
from t5.models.mesh_transformer import mesh_train_dataset_fn
from t5.models.t5_model import T5Model

import tensorflow.compat.v1 as tf


@gin.configurable
class MtfModel(T5Model):
  """Wrapper class for Mesh-TF models."""

  def __init__(
      self,
      model_dir,
      tpu,
      tpu_job_name,
      tpu_zone,
      gcp_project,
      batch_size=("tokens_per_batch", 1024),
      sequence_length=None,
      vocabulary=None,
      model_type="bitransformer",
      layout_rules="ensemble:ensemble,batch:batch,d_ff:model,heads:model,vocab:model,experts:batch",
      mesh_shape=None,
      autostack=True,
      learning_rate_schedule=None,
      keep_checkpoint_max=None,
      save_checkpoints_steps=5000,
      optimizer=None,
      predict_fn=None,
      variable_filter=None,
      ensemble_inputs=None,
      iterations_per_loop=100,
      init_checkpoint=None):
    """Constructor for MtfModel class.

    Args:
      model_dir: a string, directory to save the model.
      tpu: string, the Cloud TPU to use for training.
      tpu_job_name: string, name of the TPU worker binary.
      tpu_zone: string, GCE zone where the Cloud TPU is located
      gcp_project: string, project name for the Cloud TPU-enabled project.
      batch_size: An integer or a (method, value) pair to pass to
        compute_batch_size(). Note that this is the global batch size and not
        the per-shard batch size.
      sequence_length: an integer or a dict from feature-key to integer
        the (packed) sequence length, e.g. {"inputs": 512, "targets": 128}
      vocabulary: a vocabulary.Vocabulary or (inputs_vocabulary,
        targets_vocabulary) tuple.
      model_type: string, a model type from mesh tf models.
      layout_rules: an input to mtf.convert_to_layout_rules()
      mesh_shape: a function that returns mtf.shape
      autostack: boolean, internally combine variables.
      learning_rate_schedule: an optional function taking the scalar name
        argument `step` and the numeric argument `total_train_steps` and return
        the scalar learning rate.
      keep_checkpoint_max: an integer, maximum number of checkpoints to keep.
      save_checkpoints_steps: an integer, steps per checkpoint.
      optimizer: a class extending optimize.Optimizer, required for training.
      predict_fn: an optional function that can be used to override the default
        transformer prediction behavior. Must return a tensor of shape
        [batch_dim, length_dim] that will be the prediction for each example.
        Must accept the following arguments:
          - model: a Unitransformer or Bitransformer
          - features: a dict representing an example. Every value will be an
            mtf.Tensor with shape [batch_dim, length_dim].
          - variable_dtype: an mtf.VariableDType
      variable_filter: a string, a variable will only be trained if its name
        matches this regex. If None (default), train all trainable variables.
      ensemble_inputs: an integer, see `train_model` docstring for details.
      iterations_per_loop: integer, steps per train loop
      init_checkpoint: a string, if not None the read in varialbes from this
        checkpoint path when initializing variables.
    """

    mesh_shape = mesh_shape or []
    vocabulary = vocabulary or SentencePieceVocabulary()

    sequence_length = sequence_length or {"inputs": 512, "targets": 512}

    if isinstance(sequence_length, int):
      sequence_length = {"inputs": sequence_length,
                         "targets": sequence_length}

    if not isinstance(batch_size, int):
      self._batch_size = utils.compute_batch_size(
          sequence_length, mesh_shape, layout_rules, batch_size)
    else:
      self._batch_size = batch_size

    learning_rate_schedule = (
        learning_rate_schedule or
        learning_rate_schedules.learning_rate_schedule_noam)

    optimizer = optimizer or optimize.AdafactorOptimizer

    self._sequence_length = sequence_length
    self._vocabulary = vocabulary
    self._model_dir = model_dir
    self._init_checkpoint = init_checkpoint
    self._model_type = model_type
    self._ensemble_inputs = ensemble_inputs

    cluster = tf.distribute.cluster_resolver.TPUClusterResolver(
        tpu if (tpu) else "", zone=tpu_zone, project=gcp_project)

    self._estimator = utils.get_estimator(
        model_type=model_type,
        input_vocab_size=utils.inputs_vocabulary(vocabulary).vocab_size,
        output_vocab_size=utils.targets_vocabulary(vocabulary).vocab_size,
        layout_rules=mtf.convert_to_layout_rules(layout_rules),
        mesh_shape=mtf.convert_to_shape(mesh_shape),
        model_dir=model_dir,
        batch_size=self._batch_size,
        sequence_length=sequence_length,
        autostack=autostack,
        learning_rate_schedule=learning_rate_schedule,
        keep_checkpoint_max=keep_checkpoint_max,
        save_checkpoints_steps=save_checkpoints_steps,
        optimizer=optimizer,
        predict_fn=predict_fn,
        variable_filter=variable_filter,
        ensemble_inputs=ensemble_inputs,
        use_tpu=tpu,
        tpu_job_name=tpu_job_name,
        iterations_per_loop=iterations_per_loop,
        cluster=cluster,
        init_checkpoint=init_checkpoint)

  def train(self, mixture_or_task_name, steps):
    dataset_fn = functools.partial(
        mesh_train_dataset_fn, mixture_or_task_name=mixture_or_task_name)
    utils.train_model(self._estimator, self._vocabulary, self._sequence_length,
                      self._batch_size, dataset_fn, steps,
                      self._ensemble_inputs)

  def eval(self, mixture_or_task_name, checkpoint_step, summary_dir, split):
    dataset_fn = functools.partial(
        mesh_eval_dataset_fn, mixture_or_task_name=mixture_or_task_name)
    utils.eval_model(self._estimator, self._vocabulary, self._sequence_length,
                     self._batch_size, split, self._model_dir, dataset_fn,
                     summary_dir, checkpoint_step)

  def predict(self, checkpoint_step, input_file, output_file):
    # TODO(sharannarang) : Add the ability to decode from a collection of
    # strings instead of always requiring an input file.
    # TODO(sharannarang) : It would be nice to have a function like
    # load_checkpoint that loads the model once and then call decode_from_file
    # multiple times without having to restore the checkpoint weights again.
    # This would be particularly useful in colab demo.
    utils.infer_model(self._estimator, self._vocabulary, self._sequence_length,
                      self._batch_size, self._model_type, self._model_dir,
                      checkpoint_step, input_file, output_file)

