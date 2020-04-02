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
"""Mesh Tensorflow T5 Model."""

import functools

import os
import re
import gin
import gin.tf

import mesh_tensorflow as mtf

from mesh_tensorflow import optimize
from mesh_tensorflow.transformer import learning_rate_schedules
from mesh_tensorflow.transformer import utils

import t5.data
from t5.models.mesh_transformer import mesh_eval_dataset_fn
from t5.models.mesh_transformer import mesh_train_dataset_fn
from t5.models.t5_model import T5Model

import tensorflow.compat.v1 as tf


def _get_latest_checkpoint_from_dir(model_dir):
  """Helper function to return the latest checkpoint number from a directory.

  Args:
    model_dir: str, Directory with checkpoint files.

  Returns:
    an int, latest checkpoint number.

  Raises:
    ValueError: if no checkpoints are found.
  """
  ckpt = tf.train.latest_checkpoint(model_dir)
  if ckpt is None:
    raise ValueError("No checkpoints found in model directory: %s" % model_dir)
  return int(re.sub(".*ckpt-", "", ckpt))


def _operative_config_path(model_dir):
  return os.path.join(model_dir, "operative_config.gin")


@gin.configurable
class MtfModel(T5Model):
  """Wrapper class for Mesh-TF models."""

  def __init__(
      self,
      model_dir,
      tpu,
      tpu_job_name=None,
      tpu_zone=None,
      gcp_project=None,
      tpu_topology="2x2",
      model_parallelism=8,
      batch_size=("tokens_per_batch", 1024),
      sequence_length=None,
      model_type="bitransformer",
      layout_rules="ensemble:ensemble,batch:batch,d_ff:model,heads:model,vocab:model,experts:batch",
      mesh_shape=None,
      mesh_devices=None,
      autostack=True,
      learning_rate_schedule=None,
      keep_checkpoint_max=None,
      save_checkpoints_steps=5000,
      optimizer=None,
      predict_fn=None,
      variable_filter=None,
      ensemble_inputs=None,
      iterations_per_loop=100):
    """Constructor for MtfModel class.

    Args:
      model_dir: str, directory to save the model.
      tpu: str, the TPU address to use.
      tpu_job_name: str, name of the TPU worker binary.
      tpu_zone: str, GCE zone where the Cloud TPU is located
      gcp_project: str, project name for the Cloud TPU-enabled project.
      tpu_topology: str, e.g. "2x2".
      model_parallelism: integer, the number of cores per model replica.
      batch_size: An integer or a (method, value) pair to pass to
        compute_batch_size(). Note that this is the global batch size and not
        the per-shard batch size.
      sequence_length: an integer or a dict from feature-key to integer
        the (packed) sequence length, e.g. {"inputs": 512, "targets": 128}
      model_type: str, a model type from mesh tf models.
      layout_rules: an input to mtf.convert_to_layout_rules()
      mesh_shape: an mtf.Shape or string (e.g., "model:2,batch:4") specifying
        how the data/model should be parallelized. If None (default), the mesh
        shape will be constructed using the supplied `tpu_topology` and
        `model_parallelism` arguments.
      mesh_devices: a list of strings, the device names to use for each mesh
        slice. Only required for GPU.
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
      variable_filter: a str, a variable will only be trained if its name
        matches this regex. If None (default), train all trainable variables.
      ensemble_inputs: an integer, see `train_model` docstring for details.
      iterations_per_loop: integer, steps per train loop
    """
    mesh_shape = mesh_shape or (
        utils.tpu_mesh_shape(tpu_topology, model_parallelism) if tpu else "")

    sequence_length = sequence_length or {"inputs": 512, "targets": 512}

    if isinstance(sequence_length, int):
      sequence_length = {"inputs": sequence_length,
                         "targets": sequence_length}
    self._learning_rate_schedule = (
        learning_rate_schedule or
        learning_rate_schedules.learning_rate_schedule_noam)

    self._optimizer = optimizer or optimize.AdafactorOptimizer

    self._sequence_length = sequence_length
    self._model_dir = model_dir
    self._model_type = model_type
    self._ensemble_inputs = ensemble_inputs

    self._layout_rules = mtf.convert_to_layout_rules(layout_rules)
    self._mesh_shape = mtf.convert_to_shape(mesh_shape)
    self._mesh_devices = mesh_devices

    self._autostack = autostack
    self._keep_checkpoint_max = keep_checkpoint_max
    self._save_checkpoints_steps = save_checkpoints_steps
    self._predict_fn = predict_fn
    self._variable_filter = variable_filter
    self._ensemble_inputs = ensemble_inputs
    self._iterations_per_loop = iterations_per_loop

    self._cluster = tf.distribute.cluster_resolver.TPUClusterResolver(
        tpu if (tpu) else "", zone=tpu_zone, project=gcp_project)
    self._tpu = tpu
    self._tpu_job_name = tpu_job_name
    self._estimator = None

    # Must be called after _sequence_length, _mesh_shape, and _layout_rules are
    # set.
    self.batch_size = batch_size

  @property
  def batch_size(self):
    return self._batch_size

  @batch_size.setter
  def batch_size(self, batch_size):
    if not isinstance(batch_size, int):
      self._batch_size = utils.compute_batch_size(
          self._sequence_length, self._mesh_shape, self._layout_rules,
          batch_size)
    else:
      self._batch_size = batch_size

  def estimator(self, vocabulary, init_checkpoint=None, disable_tpu=False):

    if not self._tpu or disable_tpu:
      with gin.unlock_config():
        gin.bind_parameter("utils.get_variable_dtype.slice_dtype", "float32")
        gin.bind_parameter(
            "utils.get_variable_dtype.activation_dtype", "float32")

    return utils.get_estimator(
        model_type=self._model_type,
        vocabulary=vocabulary,
        layout_rules=self._layout_rules,
        mesh_shape=mtf.Shape([]) if disable_tpu else self._mesh_shape,
        mesh_devices=self._mesh_devices,
        model_dir=self._model_dir,
        batch_size=self.batch_size,
        sequence_length=self._sequence_length,
        autostack=self._autostack,
        learning_rate_schedule=self._learning_rate_schedule,
        keep_checkpoint_max=self._keep_checkpoint_max,
        save_checkpoints_steps=self._save_checkpoints_steps,
        optimizer=self._optimizer,
        predict_fn=self._predict_fn,
        variable_filter=self._variable_filter,
        ensemble_inputs=self._ensemble_inputs,
        use_tpu=None if disable_tpu else self._tpu,
        tpu_job_name=self._tpu_job_name,
        iterations_per_loop=self._iterations_per_loop,
        cluster=self._cluster,
        init_checkpoint=init_checkpoint)

  def train(self, mixture_or_task_name, steps, init_checkpoint=None,
            split="train"):
    """Train the model on the given Mixture or Task.

    Args:
      mixture_or_task_name: str, the name of the Mixture or Task to train on.
        Must be pre-registered in the global `TaskRegistry` or
        `MixtureRegistry.`
      steps: int, the total number of steps to train for.
      init_checkpoint: a string, if not None then read in variables from this
        checkpoint path when initializing variables. Will only initialize
        variables that appear both in the current graph and the checkpoint.
      split: str, the mixture/task split to train on.
    """
    vocabulary = t5.data.get_mixture_or_task(
        mixture_or_task_name).get_vocabulary()
    dataset_fn = functools.partial(
        mesh_train_dataset_fn, mixture_or_task_name=mixture_or_task_name)
    utils.train_model(self.estimator(vocabulary, init_checkpoint), vocabulary,
                      self._sequence_length, self.batch_size, dataset_fn,
                      steps, self._ensemble_inputs, dataset_split=split)

  def eval(self, mixture_or_task_name, checkpoint_steps=None, summary_dir=None,
           split="validation"):
    """Evaluate the model on the given Mixture or Task.

    Args:
      mixture_or_task_name: str, the name of the Mixture or Task to evaluate on.
        Must be pre-registered in the global `TaskRegistry` or
        `MixtureRegistry.`
      checkpoint_steps: int, list of ints, or None. If an int or list of ints,
        evaluation will be run on the checkpoint files in `model_dir` whose
        global steps are closest to the global steps provided. If None, run eval
        continuously waiting for new checkpoints. If -1, get the latest
        checkpoint from the model directory.
      summary_dir: str, path to write TensorBoard events file summaries for
        eval. If None, use model_dir/eval_{split}.
      split: str, the mixture/task split to evaluate on.
    """
    if checkpoint_steps == -1:
      checkpoint_steps = _get_latest_checkpoint_from_dir(self._model_dir)
    vocabulary = t5.data.get_mixture_or_task(
        mixture_or_task_name).get_vocabulary()
    dataset_fn = functools.partial(
        mesh_eval_dataset_fn, mixture_or_task_name=mixture_or_task_name)
    with gin.unlock_config():
      gin.parse_config_file(_operative_config_path(self._model_dir))
    utils.eval_model(self.estimator(vocabulary), vocabulary,
                     self._sequence_length, self.batch_size, split,
                     self._model_dir, dataset_fn, summary_dir, checkpoint_steps)

  def finetune(self, mixture_or_task_name, finetune_steps, pretrained_model_dir,
               pretrained_checkpoint_step=-1, split="train"):
    """Finetunes a model from an existing checkpoint.

    Args:
      mixture_or_task_name: str, the name of the Mixture or Task to evaluate on.
        Must be pre-registered in the global `TaskRegistry` or
        `MixtureRegistry.`
      finetune_steps: int, the number of additional steps to train for.
      pretrained_model_dir: str, directory with pretrained model checkpoints and
        operative config.
      pretrained_checkpoint_step: int, checkpoint to initialize weights from. If
        -1 (default), use the latest checkpoint from the pretrained model
        directory.
      split: str, the mixture/task split to finetune on.
    """
    if pretrained_checkpoint_step == -1:
      checkpoint_step = _get_latest_checkpoint_from_dir(pretrained_model_dir)
    else:
      checkpoint_step = pretrained_checkpoint_step
    with gin.unlock_config():
      gin.parse_config_file(_operative_config_path(pretrained_model_dir))

    model_ckpt = "model.ckpt-" + str(checkpoint_step)
    self.train(mixture_or_task_name, checkpoint_step + finetune_steps,
               init_checkpoint=os.path.join(pretrained_model_dir, model_ckpt),
               split=split)

  def predict(self, input_file, output_file, checkpoint_steps=-1,
              beam_size=1, temperature=1.0,
              sentencepiece_model_path=t5.data.DEFAULT_SPM_PATH):
    """Predicts targets from the given inputs.

    Args:
      input_file: str, path to a text file containing newline-separated input
        prompts to predict from.
      output_file: str, path prefix of output file to write predictions to. Note
        the checkpoint step will be appended to the given filename.
      checkpoint_steps: int, list of ints, or None. If an int or list of ints,
        inference will be run on the checkpoint files in `model_dir` whose
        global steps are closest to the global steps provided. If None, run
        inference continuously waiting for new checkpoints. If -1, get the
        latest checkpoint from the model directory.
      beam_size: int, a number >= 1 specifying the number of beams to use for
        beam search.
      temperature: float, a value between 0 and 1 (must be 0 if beam_size > 1)
        0.0 means argmax, 1.0 means sample according to predicted distribution.
      sentencepiece_model_path: str, path to the SentencePiece model file to use
        for decoding. Must match the one used during training.
    """
    # TODO(sharannarang) : It would be nice to have a function like
    # load_checkpoint that loads the model once and then call decode_from_file
    # multiple times without having to restore the checkpoint weights again.
    # This would be particularly useful in colab demo.

    if checkpoint_steps == -1:
      checkpoint_steps = _get_latest_checkpoint_from_dir(self._model_dir)

    with gin.unlock_config():
      gin.parse_config_file(_operative_config_path(self._model_dir))
      gin.bind_parameter("Bitransformer.decode.beam_size", beam_size)
      gin.bind_parameter("Bitransformer.decode.temperature", temperature)

    vocabulary = t5.data.SentencePieceVocabulary(sentencepiece_model_path)
    utils.infer_model(
        self.estimator(vocabulary), vocabulary, self._sequence_length,
        self.batch_size, self._model_type, self._model_dir, checkpoint_steps,
        input_file, output_file)

  def export(self, export_dir=None, checkpoint_step=-1, beam_size=1,
             temperature=1.0,
             sentencepiece_model_path=t5.data.DEFAULT_SPM_PATH):
    """Exports a TensorFlow SavedModel.

    Args:
      export_dir: str, a directory in which to export SavedModels. Will use
        `model_dir` if unspecified.
      checkpoint_step: int, checkpoint to export. If -1 (default), use the
        latest checkpoint from the pretrained model directory.
      beam_size: int, a number >= 1 specifying the number of beams to use for
        beam search.
      temperature: float, a value between 0 and 1 (must be 0 if beam_size > 1)
        0.0 means argmax, 1.0 means sample according to predicted distribution.
      sentencepiece_model_path: str, path to the SentencePiece model file to use
        for decoding. Must match the one used during training.

    Returns:
      The string path to the exported directory.
    """
    if checkpoint_step == -1:
      checkpoint_step = _get_latest_checkpoint_from_dir(self._model_dir)
    with gin.unlock_config():
      gin.parse_config_file(_operative_config_path(self._model_dir))
      gin.bind_parameter("Bitransformer.decode.beam_size", beam_size)
      gin.bind_parameter("Bitransformer.decode.temperature", temperature)

    vocabulary = t5.data.SentencePieceVocabulary(sentencepiece_model_path)
    model_ckpt = "model.ckpt-" + str(checkpoint_step)
    export_dir = export_dir or self._model_dir
    return utils.export_model(
        self.estimator(vocabulary, disable_tpu=True), export_dir, vocabulary,
        self._sequence_length, batch_size=self.batch_size,
        checkpoint_path=os.path.join(self._model_dir, model_ckpt))

