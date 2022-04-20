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

"""Mesh Tensorflow T5 Model."""

import functools
import os

import gin
import gin.tf
import mesh_tensorflow as mtf
from mesh_tensorflow import optimize
from mesh_tensorflow.transformer import dataset as transformer_dataset
from mesh_tensorflow.transformer import learning_rate_schedules
from mesh_tensorflow.transformer import utils as mtf_utils
from t5.models import mesh_transformer
from t5.models import utils
from t5.models.t5_model import T5Model
import tensorflow.compat.v1 as tf


def _parse_operative_config(model_dir):
  with gin.unlock_config():
    gin.parse_config_file(
        os.path.join(model_dir, "operative_config.gin"),
        skip_unknown=mesh_transformer.DEPRECATED_GIN_REFERENCES)


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
      tpu_topology="v2-8",
      model_parallelism=8,
      batch_size=("sequences_per_batch", 1),
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
      iterations_per_loop=100,
      extra_gin_bindings=None):
    """Constructor for MtfModel class.

    Args:
      model_dir: str, directory to save the model.
      tpu: str, the TPU address to use.
      tpu_job_name: str, name of the TPU worker binary.
      tpu_zone: str, GCE zone where the Cloud TPU is located
      gcp_project: str, project name for the Cloud TPU-enabled project.
      tpu_topology: str, e.g. "2x2" or "v2-8".
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
      extra_gin_bindings: an optional list of strings, extra gin bindings to
        pass to `gin.parse_config` after loading the operative config.
    """
    mesh_shape = mesh_shape or (
        mtf_utils.tpu_mesh_shape(
            tpu_topology, model_parallelism) if tpu else "")

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
        tpu, zone=tpu_zone, project=gcp_project) if tpu else None
    self._tpu = tpu
    self._tpu_job_name = tpu_job_name
    self._estimator = None

    # Must be called after _sequence_length, _mesh_shape, and _layout_rules are
    # set.
    self.batch_size = batch_size

    self._gin_bindings = extra_gin_bindings

  @property
  def batch_size(self):
    return self._batch_size

  @batch_size.setter
  def batch_size(self, batch_size):
    if not isinstance(batch_size, int):
      self._batch_size = mtf_utils.compute_batch_size(
          self._sequence_length, self._mesh_shape, self._layout_rules,
          batch_size)
    else:
      self._batch_size = batch_size

  def estimator(self, vocabulary, init_checkpoint=None, disable_tpu=False,
                score_in_predict_mode=False, sequence_length=None):

    if not self._tpu or disable_tpu:
      with gin.unlock_config():
        gin.bind_parameter("utils.get_variable_dtype.slice_dtype", "float32")
        gin.bind_parameter(
            "utils.get_variable_dtype.activation_dtype", "float32")
    with gin.unlock_config():
      gin.parse_config(self._gin_bindings)

    return mtf_utils.get_estimator(
        model_type=self._model_type,
        vocabulary=vocabulary,
        layout_rules=self._layout_rules,
        mesh_shape=mtf.Shape([]) if disable_tpu else self._mesh_shape,
        mesh_devices=None if disable_tpu else self._mesh_devices,
        model_dir=self._model_dir,
        batch_size=self.batch_size,
        sequence_length=sequence_length or self._sequence_length,
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
        init_checkpoint=init_checkpoint,
        score_in_predict_mode=score_in_predict_mode)

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
    vocabulary = mesh_transformer.get_vocabulary(mixture_or_task_name)
    dataset_fn = functools.partial(
        mesh_transformer.mesh_train_dataset_fn,
        mixture_or_task_name=mixture_or_task_name,
    )
    mtf_utils.train_model(
        self.estimator(vocabulary, init_checkpoint),
        vocabulary,
        self._sequence_length,
        self.batch_size,
        dataset_fn,
        steps,
        self._ensemble_inputs,
        dataset_split=split)

  def _predict_or_score_fn(self,
                           tasks,
                           vocabulary,
                           checkpoint_step,
                           sequence_length,
                           split,
                           eval_with_score=False,
                           **unused_kwargs):
    """Helper function used by eval method to generate predictions or scores.

    Args:
      tasks: list, list of valid tasks to generate predictions or scores.
      vocabulary: a t5.data.vocabulary object or a tuple with separate
        vocabularies for inputs and targets,
      checkpoint_step: integer, step to evaluate the tasks.
      sequence_length: a dict, dictionary with sequence length for inputs and
        targets.
      split: string, split to run the evaluation on.
      eval_with_score: bool, whether to compute log likelihood of targets
        instead of predictions.
    Returns:
      list of decoded predictions or scores depending on eval_with_score flag.
    """
    estimator = self.estimator(
        vocabulary, score_in_predict_mode=eval_with_score,
        sequence_length=sequence_length)

    def estimator_input_fn(params):
      """Eval input function for estimator."""
      del params
      # Concatenate all dataset inputs to only have to do one decode loop
      combined_ds = None
      for task in tasks:
        ds = mesh_transformer.mesh_eval_dataset_fn(
            mixture_or_task_name=task.name,
            sequence_length=sequence_length,
            dataset_split=split)[0].dataset_fn()
        ds = ds.map(
            utils.filter_features,
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        combined_ds = ds if not combined_ds else combined_ds.concatenate(ds)
      combined_ds = combined_ds.batch(self.batch_size, drop_remainder=False)  # pytype:disable=attribute-error
      # Pad the final batch.
      combined_ds = transformer_dataset.trim_and_pad_dataset(
          combined_ds, length=self.batch_size)
      combined_ds = combined_ds.prefetch(tf.data.experimental.AUTOTUNE)
      return combined_ds

    checkpoint_path = os.path.join(self._model_dir,
                                   "model.ckpt-{}".format(checkpoint_step))
    if eval_with_score:
      outputs, _ = mtf_utils.score_with_estimator(
          estimator,
          estimator_input_fn,
          checkpoint_step,
          self._model_dir,
          vocabulary)
    else:
      outputs = [
          tf.compat.as_text(d) for d in mtf_utils.decode(
              estimator, estimator_input_fn, vocabulary, checkpoint_path)
      ]

    return outputs

  def eval(self,
           mixture_or_task_name,
           checkpoint_steps=None,
           summary_dir=None,
           split="validation",
           eval_with_score=False,
           compute_sequence_length=True):
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
      eval_with_score: bool, whether to evaluate using log likelihood scores of
        targets instead of decoded predictions.
      compute_sequence_length: bool, automatically compute maximum sequence
        length to use during eval mode.
    """
    _parse_operative_config(self._model_dir)

    summary_dir = summary_dir or os.path.join(self._model_dir,
                                              "{}_eval".format(split))

    checkpoint_steps = utils.get_checkpoints_iterator(checkpoint_steps,
                                                      self._model_dir)

    def _get_task_eval_dataset(task, sequence_length, split):
      eval_datasets = mesh_transformer.mesh_eval_dataset_fn(
          sequence_length=sequence_length,
          dataset_split=split,
          mixture_or_task_name=task.name,
      )

      return eval_datasets[0].dataset_fn()

    utils.run_eval(
        mixture_or_task_name=mixture_or_task_name,
        predict_or_score_fn=functools.partial(self._predict_or_score_fn,
                                              eval_with_score=eval_with_score,
                                              split=split),
        checkpoint_steps=checkpoint_steps,
        dataset_fn=_get_task_eval_dataset,
        summary_dir=summary_dir,
        split=split,
        sequence_length=(None
                         if compute_sequence_length else self._sequence_length),
        batch_size=self._batch_size)

  def finetune(self,
               mixture_or_task_name,
               finetune_steps,
               pretrained_model_dir,
               pretrained_checkpoint_step=-1,
               split="train"):
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
      checkpoint_step = utils.get_latest_checkpoint_from_dir(
          pretrained_model_dir)
    else:
      checkpoint_step = pretrained_checkpoint_step
    _parse_operative_config(pretrained_model_dir)

    model_ckpt = "model.ckpt-" + str(checkpoint_step)
    self.train(mixture_or_task_name, checkpoint_step + finetune_steps,
               init_checkpoint=os.path.join(pretrained_model_dir, model_ckpt),
               split=split)

  def predict(self, input_file, output_file, checkpoint_steps=-1,
              beam_size=1, temperature=1.0, keep_top_k=-1, vocabulary=None):
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
      keep_top_k: integer, a value between 1 and the vocabulary size. When
        sampling, only pick tokens that are in the k most likely.
      vocabulary: vocabularies.Vocabulary object to use for tokenization, or
        None to use the default SentencePieceVocabulary.
    """
    # TODO(sharannarang) : It would be nice to have a function like
    # load_checkpoint that loads the model once and then call decode_from_file
    # multiple times without having to restore the checkpoint weights again.
    # This would be particularly useful in colab demo.

    if checkpoint_steps == -1:
      checkpoint_steps = utils.get_latest_checkpoint_from_dir(self._model_dir)

    _parse_operative_config(self._model_dir)
    with gin.unlock_config():
      gin.bind_parameter("Bitransformer.decode.beam_size", beam_size)
      gin.bind_parameter("Bitransformer.decode.temperature", temperature)
      gin.bind_parameter("Bitransformer.decode.sampling_keep_top_k", keep_top_k)
      gin.bind_parameter("utils.decode_from_file.input_filename", input_file)
      gin.bind_parameter("utils.decode_from_file.output_filename", output_file)

    if vocabulary is None:
      vocabulary = utils.get_vocabulary()
    mtf_utils.infer_model(
        self.estimator(vocabulary), vocabulary, self._sequence_length,
        self.batch_size, self._model_type, self._model_dir, checkpoint_steps)

  def score(self,
            inputs=None,
            targets=None,
            mixture_or_task_name=None,
            mixture_or_task_split=None,
            scores_file=None,
            checkpoint_steps=-1,
            vocabulary=None):
    """Computes log-likelihood of target per example in targets.

    Args:
      inputs: optional - a string (filename), or a list of strings (inputs)
      targets: optional - a string (filename), or a list of strings (targets)
      mixture_or_task_name: optional - a string, the name of the Mixture or Task
        to score on. Must be pre-registered in the global `TaskRegistry` or
        `MixtureRegistry.` Cannot be supplied in addition to `inputs` and
        `targets`.
      mixture_or_task_split: optional - a string, the split of the Mixture or
        Task to score on. Must be provided if scoring on a Mixture or Task.
      scores_file: optional - a string (filename), to write example scores to,
        one per line.
      checkpoint_steps: int, list of ints, or None. If an int or list of ints,
        inference will be run on the checkpoint files in `model_dir` whose
        global steps are closest to the global steps provided. If None, run
        inference continuously waiting for new checkpoints. If -1, get the
        latest checkpoint from the model directory.
      vocabulary: vocabularies.Vocabulary object to use for tokenization, or
        None to use the default SentencePieceVocabulary.

    Returns:
      scores: a list of floating point scores matching the dataset order.
      targets: a list of scored strings matching the dataset order.
    """
    if bool(inputs or targets) == bool(
        mixture_or_task_name or mixture_or_task_split):
      raise ValueError(
          "Either 'inputs' and 'targets' or "
          "'mixture_or_task_name' and 'mixture_or_task_split' must be "
          "specified, but not both.")

    if checkpoint_steps == -1:
      checkpoint_steps = utils.get_latest_checkpoint_from_dir(self._model_dir)

    _parse_operative_config(self._model_dir)
    with gin.unlock_config():
      gin.parse_config(self._gin_bindings)

    if vocabulary is None:
      vocabulary = utils.get_vocabulary(mixture_or_task_name)

    estimator = self.estimator(vocabulary, score_in_predict_mode=True)
    score_postprocess_fn = functools.partial(
        mtf_utils.save_scores, scores_filename=scores_file)

    if mixture_or_task_name:
      score_dataset_fn = functools.partial(
          mesh_transformer.mesh_eval_dataset_fn,
          mixture_or_task_name=mixture_or_task_name,
      )
      return mtf_utils.score_from_dataset(
          estimator=estimator, vocabulary=vocabulary,
          batch_size=self.batch_size, sequence_length=self._sequence_length,
          model_dir=self._model_dir, eval_checkpoint_step=checkpoint_steps,
          dataset_split=mixture_or_task_split,
          score_dataset_fn=score_dataset_fn,
          score_postprocess_fn=score_postprocess_fn)
    else:
      return mtf_utils.score_from_strings(
          estimator=estimator, vocabulary=vocabulary,
          model_type=self._model_type, batch_size=self.batch_size,
          sequence_length=self._sequence_length, model_dir=self._model_dir,
          eval_checkpoint_step=checkpoint_steps, inputs=inputs, targets=targets,
          score_postprocess_fn=score_postprocess_fn)

  def export(self, export_dir=None, checkpoint_step=-1, beam_size=1,
             temperature=1.0, keep_top_k=-1, vocabulary=None,
             eval_with_score=False):
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
      keep_top_k: integer, a value between 1 and the vocabulary size. When
        sampling, only pick tokens that are in the k most likely.
      vocabulary: vocabularies.Vocabulary object to use for tokenization, or
        None to use the default SentencePieceVocabulary.
      eval_with_score: If True, compute log-likelihood scores of targets.
        If False, do inference to generate outputs.

    Returns:
      The string path to the exported directory.
    """
    if checkpoint_step == -1:
      checkpoint_step = utils.get_latest_checkpoint_from_dir(self._model_dir)
    _parse_operative_config(self._model_dir)
    with gin.unlock_config():
      gin.bind_parameter("Bitransformer.decode.beam_size", beam_size)
      gin.bind_parameter("Bitransformer.decode.temperature", temperature)
      gin.bind_parameter("Bitransformer.decode.sampling_keep_top_k", keep_top_k)

    if vocabulary is None:
      vocabulary = utils.get_vocabulary()
    model_ckpt = "model.ckpt-" + str(checkpoint_step)
    export_dir = export_dir or self._model_dir
    estimator = self.estimator(
        vocabulary, disable_tpu=True, score_in_predict_mode=eval_with_score)
    return mtf_utils.export_model(
        estimator, export_dir, vocabulary,
        self._sequence_length, self._model_type, batch_size=self.batch_size,
        checkpoint_path=os.path.join(self._model_dir, model_ckpt),
        eval_with_score=eval_with_score)
