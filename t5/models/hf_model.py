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

"""Hugging Face Transformers T5 Model.

This model API is fully functional but should be treated as experimental and
subject to change. Due to implementation details, if you are interested in
exactly replicating the results in ``Exploring the Limits of Transfer Learning
with a Unified Text-to-Text Transformer'' you should use the MtfModel API
instead.

Usage example for fine-tuning and evaluating on CoLA:

```Python
import functools

import t5
import t5.data.mixtures
import t5.models
import torch
import transformers

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = t5.models.HfPyTorchModel("t5-base", "/tmp/hft5/", device)

# Evaluate the pre-trained checkpoint, before further fine-tuning
model.eval(
    "glue_cola_v002",
    sequence_length={"inputs": 64, "targets": 4},
    batch_size=128,
)

# Run 1000 steps of fine-tuning
model.train(
    mixture_or_task_name="glue_cola_v002",
    steps=1000,
    save_steps=100,
    sequence_length={"inputs": 64, "targets": 4},
    split="train",
    batch_size=32,
    optimizer=functools.partial(transformers.AdamW, lr=1e-4),
)

# Evaluate after fine-tuning
model.eval(
    "glue_cola_v002",
    checkpoint_steps="all",
    sequence_length={"inputs": 64, "targets": 4},
    batch_size=128,
)

# Generate some predictions
inputs = [
    "cola sentence: This is a totally valid sentence.",
    "cola sentence: A doggy detail was walking famously.",
]
model.predict(
    inputs,
    sequence_length={"inputs": 32},
    batch_size=2,
    output_file="/tmp/hft5/example_predictions.txt",
)
```

"""

import functools
import itertools
import os
import re
import time

from absl import logging
import mesh_tensorflow.transformer.dataset as transformer_dataset
import t5.data
from t5.models import utils
from t5.models.t5_model import T5Model
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds
import torch
import torch.utils.tensorboard

CHECKPOINT_FILE_FORMAT = "model-{}.checkpoint"


def tokens_to_batches(dataset,
                      sequence_length,
                      batch_size,
                      output_features,
                      mixture_or_task=None):
  """Convert a dataset of token sequences to batches of padded/masked examples.

  Args:
    dataset: tf.data.Dataset containing examples with token sequences.
    sequence_length: dict of int, a dict mapping feature name to length.
    batch_size: int, the number of padded sequences in each batch.
    output_features: list of str, features to include in the dataset.
    mixture_or_task: a Task or Mixture object, used to correctly specify eos if
      provided. If none, eos is always added at the end of the sequence.

  Returns:
    A generator that produces batches of numpy examples.
  """

  if mixture_or_task:
    eos_keys = set(
        k for k, f in mixture_or_task.output_features.items() if f.add_eos)
  else:
    eos_keys = True

  dataset = transformer_dataset.pack_or_pad(
      dataset,
      sequence_length,
      pack=False,
      feature_keys=output_features,
      ensure_eos=eos_keys,
  )

  def _map_fn(ex):
    for key in output_features:
      tensor = ex[key]
      mask = tf.cast(tf.greater(tensor, 0), tensor.dtype)
      ex[key + "_mask"] = mask
    return ex

  dataset = dataset.map(
      _map_fn,
      num_parallel_calls=tf.data.experimental.AUTOTUNE,
  )

  dataset = dataset.batch(batch_size, drop_remainder=False)
  return tfds.as_numpy(dataset)


def _get_dataset(mixture_or_task_or_name,
                 sequence_length,
                 split,
                 shuffle=True):
  """Get a tf.data.Dataset for a given Task or Mixture.

  Args:
    mixture_or_task_or_name: Task or Mixture or str, the name of the Mixture or
      Task to train on or the Tasks or Mixture object itself.
      Must be pre-registered in the global `t5.data.TaskRegistry` or
      `t5.data.MixtureRegistry.`
    sequence_length: dict of int, a dict mapping feature name to length.
    split: str or `tensorflow_datasets.Split`, the data split to load.
    shuffle: boolean, whether to shuffle the dataset.

  Returns:
    A generator that produces batches of numpy examples.
  """
  if isinstance(mixture_or_task_or_name, str):
    task = t5.data.get_mixture_or_task(mixture_or_task_or_name)
  else:
    task = mixture_or_task_or_name

  return task.get_dataset(sequence_length, split, shuffle=shuffle)


class HfPyTorchModel(T5Model):
  """Wrapper class for Hugging Face Transformers PyTorch T5 model."""

  def __init__(self, model_spec, model_dir, device):
    """Constructor for HfModel class.

    Args:
      model_spec: A str to pass into the `pretrained_model_name_or_path`
        argument of `transformers.T5ForConditionalGeneration.from_pretrained`
        (e.g. `"t5-base"` or a path to a previously trained model) or an
        instance of the `transformers.configuration_t5.T5Config` class to use
        to directly construct the `transformers.T5ForConditionalGeneration`
        object.
      model_dir: str, directory to save and load model checkpoints.
      device: `torch.device` on which the model should be run.
    """
    # We have to import transformers here because it has a side effect of
    # creating a TensorFlow graph, which prevents eager execution from being
    # enabled in files that import hf_model.py
    import transformers  # pylint: disable=import-outside-toplevel,g-import-not-at-top
    if isinstance(model_spec, str):
      self._model = transformers.T5ForConditionalGeneration.from_pretrained(
          model_spec
      )
    elif isinstance(model_spec, transformers.T5Config):
      self._model = transformers.T5ForConditionalGeneration(model_spec)
    else:
      raise ValueError("model_spec should be a string or T5Config.")

    tf.io.gfile.makedirs(model_dir)
    self._writer = torch.utils.tensorboard.writer.SummaryWriter(model_dir)
    self._model_dir = model_dir
    self._device = device
    if self._device.type == "cuda":
      self._model.cuda()
    self._step = 0
    self.load_latest_checkpoint()
    self.to_tensor = functools.partial(
        torch.as_tensor, device=self._device, dtype=torch.long)

  @property
  def model(self):
    return self._model

  @property
  def step(self):
    return self._step

  def save_checkpoint(self, step):
    """Save the current model parameters to the `model_dir`.

    Args:
      step: int, the current training step.
    """
    path = os.path.join(self._model_dir, CHECKPOINT_FILE_FORMAT.format(step))
    torch.save(self._model.state_dict(), path)

  def load_checkpoint(self, step, model_dir=None):
    """Load the model parameters from a checkpoint at a given step.

    Args:
      step: int, load the checkpoint from this training step.
      model_dir: str, the directory of the checkpoint to load or None to use
        this model's directory.
    """
    model_dir = model_dir or self._model_dir
    path = os.path.join(model_dir, CHECKPOINT_FILE_FORMAT.format(step))
    logging.info("Loading from %s", path)
    self._model.load_state_dict(torch.load(path))
    self._step = step

  def get_all_checkpoint_steps(self, model_dir=None):
    """Retrieve the steps corresponding to all checkpoints in `model_dir`.

    Args:
      model_dir: str, the directory of the checkpoints or None to use this
        model's directory.

    Returns:
      A list of ints corresponding to all checkpoint steps, or None if there
        are no checkpoints in the model directory.
    """
    model_dir = model_dir or self._model_dir
    checkpoint_files = tf.io.gfile.glob(
        os.path.join(model_dir, CHECKPOINT_FILE_FORMAT.format("*"))
    )
    if not checkpoint_files:
      return
    step_regex = re.compile(".*" + CHECKPOINT_FILE_FORMAT.format(r"(\d+)"))
    steps = [int(step_regex.match(path).group(1)) for path in checkpoint_files]
    return sorted(steps)

  def get_latest_checkpoint_step(self, model_dir=None):
    """Retrieve the step corresponding to the most recent checkpoint.

    Args:
      model_dir: str, the directory of the checkpoints or None to use this
        model's directory.

    Returns:
      An integer corresponding to the most recent step, or None if there are no
      checkpoints in the model directory.
    """
    steps = self.get_all_checkpoint_steps(model_dir)
    if steps is not None:
      return max(steps)

  def load_latest_checkpoint(self):
    """Load the most recent checkpoint and update the model's current step."""
    latest_step = self.get_latest_checkpoint_step()
    if latest_step is not None:
      self.load_checkpoint(latest_step)

  def train(
      self,
      mixture_or_task_name,
      steps,
      save_steps,
      sequence_length,
      split,
      batch_size,
      optimizer,
      learning_rate_scheduler=None,
  ):
    """Train the model on the given Mixture or Task.

    Args:
      mixture_or_task_name: str, the name of the Mixture or Task to train on.
        Must be pre-registered in the global `t5.data.TaskRegistry` or
        `t5.data.MixtureRegistry.`
      steps: int, the total number of steps to train for.
      save_steps: int, the number of steps between checkpoint saves.
      sequence_length: dict of int, a dict mapping feature name to length.
      split: str or `tensorflow_datasets.Split`, the data split to load.
      batch_size: int, the number of padded sequences in each batch.
      optimizer: function that takes the model parameters as its sole argument.
        For example, to use an AdamW optimizer with a learning rate of 1e-4,
        you could pass in `functools.partial(transformers.AdamW, lr=1e-4)`.
      learning_rate_scheduler: optional function that takes in an optimizer as
        its sole argument. For example, to use a schedule that warms up the
        optimizer's learning rate after 100 steps, you could pass in
        `functools.partial(transformers.get_constant_schedule_with_warmup,
       num_warmup_steps=100)`.
    """
    self._model.train()
    ds = _get_dataset(mixture_or_task_name, sequence_length, split)
    task = t5.data.get_mixture_or_task(mixture_or_task_name)
    ds = tokens_to_batches(ds, sequence_length, batch_size,
                           tuple(task.output_features), task)
    # Repeat dataset forever
    ds = itertools.cycle(ds)
    optimizer = optimizer(self._model.parameters())
    if learning_rate_scheduler:
      learning_rate_scheduler = learning_rate_scheduler(optimizer)

    now = time.time()
    for train_step, batch in enumerate(itertools.islice(ds, steps)):

      if not train_step % save_steps:
        # TODO(craffel): Consider saving optimizer and scheduler state.
        logging.info("Saving checkpoint for step %s", self._step)
        self.save_checkpoint(self._step)

      self._model.zero_grad()
      outputs = self._model(
          input_ids=self.to_tensor(batch["inputs"]),
          attention_mask=self.to_tensor(batch["inputs_mask"]),
          decoder_attention_mask=self.to_tensor(batch["targets_mask"]),
          labels=self.to_tensor(batch["targets"]),
      )
      loss = outputs[0]
      loss.backward()
      optimizer.step()
      if learning_rate_scheduler:
        learning_rate_scheduler.step()

      self._writer.add_scalar(
          "loss", loss.detach().cpu().numpy(), self._step
      )
      self._writer.add_scalar("step/s", 1 / (time.time() - now), self._step)
      now = time.time()
      self._step += 1

    logging.info("Saving final checkpoint for step %s", self._step)
    self.save_checkpoint(self._step)

  def eval(
      self,
      mixture_or_task_name,
      sequence_length,
      batch_size,
      checkpoint_steps=None,
      summary_dir=None,
      split="validation",
      compute_sequence_length=False,
      **generate_kwargs,
  ):
    """Evaluate the model on the given Mixture or Task.

    *Note*: If a checkpoint step is provided (i.e. `checkpoint_steps is not
    None`), the model's state will be replaced by the state in those
    checkpoints. If you have not saved your model before calling `eval`, you
    should call `save_checkpoint` before `eval` to avoid losing its parameter
    values and state.

    Args:
      mixture_or_task_name: str, the name of the Mixture or Task to evaluate
        on.  Must be pre-registered in the global `t5.data.TaskRegistry` or
        `t5.data.MixtureRegistry.`
      sequence_length: dict of int, a dict mapping feature name to length.
      batch_size: int, the number of padded sequences in each batch.
      checkpoint_steps: int, list of ints, "all", or None. If None, eval in the
        model in its current state without loading any checkpoints. If an int
        or list of ints, evaluation will be run on the checkpoint files in
        `model_dir` whose global steps are those provided. If -1, eval on the
        latest checkpoint from the model directory. If "all", evaluate all
        checkpoints in the model directory.
      summary_dir: str, path to write TensorBoard events file summaries for
        eval. If None, use model_dir/{split}_eval.
      split: str, the mixture/task split to evaluate on.
      compute_sequence_length: bool, automatically compute sequence length
        during eval mode.
      **generate_kwargs: Additional keyword arguments to pass to
        `transformers.PretrainedModel.generate()`, for example to change the
        decoding strategy. See the documentation for
        `transformers.PretrainedModel.generate()` for options.
    """

    def _predict_from_tasks(tasks, vocabulary, checkpoint_step, sequence_length,
                            datasets, **unused_kwargs):

      if isinstance(vocabulary, tuple):
        vocab = vocabulary[1]

      if checkpoint_step != self._step:
        self.load_checkpoint(checkpoint_step)
      self._model.eval()
      outputs = []
      for task in tasks:
        if compute_sequence_length:
          ds = _get_dataset(task.name, sequence_length, split, shuffle=False)
        else:
          ds = datasets[task.name]

        ds = list(tokens_to_batches(
            ds, sequence_length, batch_size, tuple(task.output_features), task))
        for batch in ds:
          predicted_tokens = self._model.generate(
              input_ids=self.to_tensor(batch["inputs"]), **generate_kwargs
          )
          predicted_tokens = predicted_tokens.cpu().numpy().tolist()
          predictions = [vocab.decode(p) for p in predicted_tokens]

          outputs.extend(predictions)

      return outputs

    if checkpoint_steps is None:
      checkpoint_steps = [self._step]
    elif isinstance(checkpoint_steps, int):
      checkpoint_steps = [checkpoint_steps]
    elif checkpoint_steps == "all":
      checkpoint_steps = self.get_all_checkpoint_steps()
    elif not isinstance(checkpoint_steps, (list, tuple)):
      raise ValueError(
          f"checkpoint_steps must be None, int or list; got {checkpoint_steps}"
      )

    summary_dir = summary_dir or os.path.join(self._model_dir, f"{split}_eval")
    tf.io.gfile.makedirs(summary_dir)

    utils.run_eval(
        mixture_or_task_name=mixture_or_task_name,
        predict_or_score_fn=_predict_from_tasks,
        checkpoint_steps=checkpoint_steps,
        dataset_fn=functools.partial(_get_dataset, shuffle=False),
        summary_dir=summary_dir,
        split=split,
        sequence_length=None if compute_sequence_length else sequence_length,
        batch_size=batch_size)

  def predict(
      self,
      inputs,
      sequence_length,
      batch_size,
      output_file=None,
      vocabulary=None,
      **generate_kwargs,
  ):
    """Evaluate the model on the given Mixture or Task.

    *Note*: If a checkpoint step is provided (i.e. `checkpoint_steps is not
    None`), the model's state will be replaced by the state in those
    checkpoints. If you have not saved your model before calling `eval`, you
    should call `save_checkpoint` before `eval` to avoid losing its parameter
    values and state.

    Args:
      inputs: list of str or str, either a list of inputs to feed into the
        model or the path to a text file that contains a single input on each
        line.
      sequence_length: dict of int, a dict mapping feature name to length.
      batch_size: int, the number of padded sequences in each batch.
      output_file: str or None, path to write out predictions or None to skip
        writing.
      vocabulary: t5.data.vocabularies.Vocabulary or dict or None. Either the
        Vocabulary to use for processing inputs and targets, a dict mapping
        "inputs" to a Vocabulary for encoding the inputs and "targets" for
        decoding the predictions, or None (default) to use a
        t5.data.SentencePieceVocabulary with the provided
        sentencepiece_model_path (as was used in all pre-trained T5 models).
      **generate_kwargs: Additional keyword arguments to pass to
        `transformers.PretrainedModel.generate()`, for example to change the
        decoding strategy. See the documentation for
        `transformers.PretrainedModel.generate()` for options.
    """
    if isinstance(inputs, str):
      if not tf.io.gfile.exists(inputs):
        raise ValueError(
            f"A str was provided for `inputs`, but the path {inputs} does not "
            "exist. If you want the model's output for {inputs}, you should "
            "feed in inputs=['{inputs}']"
        )
      with tf.io.gfile.GFile(inputs) as f:
        inputs = [l.strip() for l in f]

    if vocabulary is None:
      vocab = t5.data.get_default_vocabulary()
      vocabs = {"inputs": vocab, "targets": vocab}
    elif isinstance(vocabulary, t5.data.vocabularies.Vocabulary):
      vocabs = {"inputs": vocabulary, "targets": vocabulary}
    elif isinstance(vocabulary, dict):
      vocabs = vocabulary
    else:
      raise ValueError("vocabulary must be a dict, a Vocabulary, or None")

    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    dataset = dataset.map(
        lambda x: {"inputs": tf.cast(vocabs["inputs"].encode_tf(x), tf.int64)},
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    dataset = tokens_to_batches(
        dataset, sequence_length, batch_size, ["inputs"]
    )

    predictions = []
    for batch in dataset:
      predicted_tokens = self._model.generate(
          input_ids=self.to_tensor(batch["inputs"]), **generate_kwargs
      )
      predicted_tokens = predicted_tokens.cpu().numpy().tolist()
      predictions.extend(
          [vocabs["targets"].decode(p) for p in predicted_tokens]
      )

    for inp, pred in zip(inputs, predictions):
      logging.info("%s\n  -> %s", inp, pred)

    if output_file is not None:
      utils.write_lines_to_file(predictions, output_file)

  def finetune(
      self,
      mixture_or_task_name,
      finetune_steps,
      pretrained_model_dir,
      pretrained_checkpoint_step=-1,
      **train_kwargs,
  ):
    """Trains model after loading from any existing checkpoint.

    Note that if you have initialized the model using a pre-trained model
    specification (e.g. by passing "t5-base" for `model_spec`) then you can
    just call `train` directly. This function is only provided for convenience
    for loading a pre-trained model checkpoint from an arbitrary model
    directory before calling `train`.

    Args:
      mixture_or_task_name: str, the name of the Mixture or Task to evaluate
        on.  Must be pre-registered in the global `t5.data.TaskRegistry` or
        `t5.data.MixtureRegistry.`
      finetune_steps: int, the number of additional steps to train for.
      pretrained_model_dir: str, directory with pretrained model checkpoints.
      pretrained_checkpoint_step: int, checkpoint to initialize weights from.
        If -1 (default), use the latest checkpoint from the pretrained model
        directory.
      **train_kwargs: Additional keyword arguments to pass to `train`. See the
        docstring for `train` for more details.
    """
    if pretrained_checkpoint_step == -1:
      pretrained_checkpoint_step = self.get_latest_checkpoint_step(
          pretrained_model_dir
      )
    self.load_checkpoint(pretrained_checkpoint_step, pretrained_model_dir)
    self.train(mixture_or_task_name, finetune_steps, **train_kwargs)
