# Copyright 2026 The T5 Authors.
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

"""Focused no-network tests for the Hugging Face PyTorch model backend."""

import copy
import functools
import json
import os
import random
import tempfile
from types import SimpleNamespace
from unittest import mock

from absl.testing import absltest
import numpy as np
from t5.models import hf_model
import torch


class _Writer:

  def add_scalar(self, *unused_args, **unused_kwargs):
    pass


class _ToyModel(torch.nn.Module):

  def __init__(self, dropout=False):
    super().__init__()
    self.trainable = torch.nn.Parameter(torch.tensor([0.25]))
    self.frozen = torch.nn.Parameter(
        torch.tensor([7.0]), requires_grad=False
    )
    self.dropout = dropout
    self.last_labels = None
    self.last_decoder_attention_mask = None

  def forward(
      self,
      input_ids,
      attention_mask,
      decoder_attention_mask,
      labels,
  ):
    del attention_mask
    self.last_labels = labels.detach().clone()
    self.last_decoder_attention_mask = decoder_attention_mask.detach().clone()
    valid_targets = labels.masked_select(labels.ne(-100)).float().mean() / 10.0
    multiplier = torch.ones_like(self.trainable)
    if self.dropout:
      multiplier = torch.nn.functional.dropout(
          multiplier, p=0.5, training=self.training
      )
    prediction = self.trainable * multiplier + input_ids.float().mean() / 100.0
    return SimpleNamespace(loss=(prediction - valid_targets).pow(2).sum())


def _batch(batch_id=1, targets=None, targets_mask=None):
  if targets is None:
    targets = [[batch_id + 1, batch_id + 2]]
  if targets_mask is None:
    targets_mask = [[1, 1]]
  return {
      "inputs": np.asarray([[batch_id, batch_id + 1]], dtype=np.int64),
      "inputs_mask": np.asarray([[1, 1]], dtype=np.int64),
      "targets": np.asarray(targets, dtype=np.int64),
      "targets_mask": np.asarray(targets_mask, dtype=np.int64),
  }


def _task():
  return SimpleNamespace(
      name="task", output_features={"inputs": object(), "targets": object()}
  )


def _make_wrapper(model_dir, model=None, hook=None):
  wrapper = object.__new__(hf_model.HfPyTorchModel)
  wrapper._model = model or _ToyModel()
  wrapper._model_dir = os.path.abspath(model_dir)
  os.makedirs(wrapper._model_dir, exist_ok=True)
  wrapper._device = torch.device("cpu")
  wrapper._writer = _Writer()
  wrapper._step = 0
  wrapper._pending_training_state = None
  wrapper._checkpoint_state_usable = True
  wrapper._checkpoint_manifest_hook = hook
  wrapper.to_tensor = functools.partial(
      torch.as_tensor, device=wrapper._device, dtype=torch.long
  )
  return wrapper


def _patch_data(epoch_size=2, batch_offset=0):
  task = _task()

  def get_dataset(unused_task, unused_lengths, unused_split, **kwargs):
    return kwargs["seed"]

  def tokens_to_batches(
      epoch_seed,
      unused_lengths,
      unused_batch_size,
      unused_features,
      unused_task,
  ):
    return [
        _batch(batch_offset + epoch_seed * 10 + index + 1)
        for index in range(epoch_size)
    ]

  return (
      mock.patch.object(
          hf_model.seqio, "get_mixture_or_task", return_value=task
      ),
      mock.patch.object(hf_model, "_get_dataset", side_effect=get_dataset),
      mock.patch.object(
          hf_model, "tokens_to_batches", side_effect=tokens_to_batches
      ),
  )


def _valid_resumable_envelope(wrapper, step):
  optimizer = torch.optim.SGD([wrapper.model.trainable], lr=0.01)
  training_config = hf_model._training_config(
      "task",
      {"inputs": 2, "targets": 2},
      "train",
      1,
      0,
      ("inputs", "targets"),
      ("trainable",),
      optimizer,
      None,
      torch.device("cpu"),
      "sha256:fixture-data",
      "requirements:fixture",
  )
  return hf_model._make_checkpoint_envelope(
      step,
      copy.deepcopy(wrapper.model.state_dict()),
      optimizer_state_dict=optimizer.state_dict(),
      scheduler_state_dict=None,
      training_config=training_config,
      iterator_state={
          "version": hf_model.ITERATOR_STATE_VERSION,
          "base_seed": 0,
          "epoch": 0,
          "batches_consumed_in_epoch": 0,
      },
      rng_state=hf_model._capture_rng_state(),
  )


class HfModelTrainingTest(absltest.TestCase):

  def test_train_derives_labels_only_from_targets_mask_and_filters_parameters(self):
    wrapper = _make_wrapper(tempfile.mkdtemp())
    task = _task()
    batch = _batch(
        targets=[[5, 0, 7, 9]], targets_mask=[[1, 0, 1, 0]]
    )
    received_parameters = []

    def optimizer_factory(parameters):
      received_parameters.extend(parameters)
      return torch.optim.SGD(received_parameters, lr=0.01)

    with mock.patch.object(
        hf_model.seqio, "get_mixture_or_task", return_value=task
    ), mock.patch.object(hf_model, "_get_dataset", return_value=object()), \
         mock.patch.object(hf_model, "tokens_to_batches", return_value=[batch]):
      frozen_before = wrapper.model.frozen.detach().clone()
      wrapper.train(
          "task",
          steps=1,
          save_steps=10,
          sequence_length={"inputs": 4, "targets": 4},
          split="train",
          batch_size=1,
          optimizer=optimizer_factory,
      )

    self.assertLen(received_parameters, 1)
    self.assertIs(received_parameters[0], wrapper.model.trainable)
    self.assertTrue(torch.equal(wrapper.model.frozen, frozen_before))
    self.assertIsNone(wrapper.model.frozen.grad)
    self.assertTrue(torch.equal(
        wrapper.model.last_labels, torch.tensor([[5, -100, 7, -100]])
    ))
    self.assertTrue(torch.equal(
        wrapper.model.last_decoder_attention_mask, torch.tensor([[1, 0, 1, 0]])
    ))

  def test_all_frozen_parameters_are_rejected(self):
    wrapper = _make_wrapper(tempfile.mkdtemp())
    wrapper.model.trainable.requires_grad_(False)
    with mock.patch.object(
        hf_model.seqio, "get_mixture_or_task", return_value=_task()
    ):
      with self.assertRaisesRegex(ValueError, "No model parameters"):
        wrapper.train(
            "task",
            steps=1,
            save_steps=1,
            sequence_length={"inputs": 2, "targets": 2},
            split="train",
            batch_size=1,
            optimizer=lambda parameters: torch.optim.SGD(parameters, lr=0.1),
        )

  def test_masked_padding_does_not_affect_t5_loss(self):
    try:
      import transformers  # pylint: disable=g-import-not-at-top
    except ImportError:
      self.skipTest("transformers is not installed")
    config = transformers.T5Config(
        vocab_size=16,
        d_model=8,
        d_ff=16,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        dropout_rate=0.0,
        decoder_start_token_id=0,
        pad_token_id=0,
    )
    model = transformers.T5ForConditionalGeneration(config).eval()
    input_ids = torch.tensor([[2, 3]])
    target_mask = torch.tensor([[1, 0]])
    labels_a = torch.tensor([[4, 5]]).masked_fill(~target_mask.bool(), -100)
    labels_b = torch.tensor([[4, 9]]).masked_fill(~target_mask.bool(), -100)
    with torch.no_grad():
      loss_a = model(
          input_ids=input_ids,
          decoder_attention_mask=target_mask,
          labels=labels_a,
      ).loss
      loss_b = model(
          input_ids=input_ids,
          decoder_attention_mask=target_mask,
          labels=labels_b,
      ).loss
    torch.testing.assert_close(loss_a, loss_b)

  def test_split_resume_matches_uninterrupted_training(self):
    sequence_length = {"inputs": 2, "targets": 2}
    optimizer = lambda parameters: torch.optim.Adam(parameters, lr=0.02)
    scheduler = lambda opt: torch.optim.lr_scheduler.StepLR(
        opt, step_size=1, gamma=0.8
    )

    random.seed(12)
    np.random.seed(12)
    torch.manual_seed(12)
    uninterrupted = _make_wrapper(
        tempfile.mkdtemp(), _ToyModel(dropout=True)
    )
    patches = _patch_data(epoch_size=3)
    with patches[0], patches[1], patches[2]:
      uninterrupted.train(
          "task", 4, 2, sequence_length, "train", 1, optimizer, scheduler, seed=4
      )

    random.seed(12)
    np.random.seed(12)
    torch.manual_seed(12)
    split_dir = tempfile.mkdtemp()
    first = _make_wrapper(split_dir, _ToyModel(dropout=True))
    patches = _patch_data(epoch_size=3)
    with patches[0], patches[1], patches[2]:
      first.train(
          "task", 2, 2, sequence_length, "train", 1, optimizer, scheduler, seed=4
      )

    random.random()
    np.random.random()
    torch.rand(3)
    resumed = _make_wrapper(split_dir, _ToyModel(dropout=True))
    resumed.load_checkpoint(2, resume_training=True)
    patches = _patch_data(epoch_size=3)
    with patches[0], patches[1], patches[2]:
      resumed.train(
          "task", 2, 2, sequence_length, "train", 1, optimizer, scheduler, seed=4
      )

    self.assertEqual(uninterrupted.step, resumed.step)
    for name, value in uninterrupted.model.state_dict().items():
      torch.testing.assert_close(value, resumed.model.state_dict()[name])
    self.assertEqual(
        uninterrupted._pending_training_state["iterator_state"],
        resumed._pending_training_state["iterator_state"],
    )
    self.assertEqual(
        uninterrupted._pending_training_state["scheduler_state_dict"],
        resumed._pending_training_state["scheduler_state_dict"],
    )
    self.assertEqual(
        uninterrupted._pending_training_state["optimizer_state_dict"][
            "param_groups"
        ],
        resumed._pending_training_state["optimizer_state_dict"]["param_groups"],
    )
    uninterrupted_optimizer_state = uninterrupted._pending_training_state[
        "optimizer_state_dict"
    ]["state"]
    resumed_optimizer_state = resumed._pending_training_state[
        "optimizer_state_dict"
    ]["state"]
    self.assertEqual(
        uninterrupted_optimizer_state.keys(), resumed_optimizer_state.keys()
    )
    for parameter_id in uninterrupted_optimizer_state:
      for key, value in uninterrupted_optimizer_state[parameter_id].items():
        other = resumed_optimizer_state[parameter_id][key]
        if isinstance(value, torch.Tensor):
          torch.testing.assert_close(value, other)
        else:
          self.assertEqual(value, other)

  def test_resume_rejects_configuration_change_before_update(self):
    model_dir = tempfile.mkdtemp()
    wrapper = _make_wrapper(model_dir)
    patches = _patch_data()
    with patches[0], patches[1], patches[2]:
      wrapper.train(
          "task",
          1,
          10,
          {"inputs": 2, "targets": 2},
          "train",
          1,
          lambda parameters: torch.optim.Adam(parameters, lr=0.01),
          seed=3,
      )
    resumed = _make_wrapper(model_dir)
    resumed.load_checkpoint(1, resume_training=True)
    loaded_weight = resumed.model.trainable.detach().clone()
    patches = _patch_data()
    with patches[0], patches[1], patches[2]:
      with self.assertRaisesRegex(ValueError, "base_seed"):
        resumed.train(
            "task",
            5,
            10,
            {"inputs": 2, "targets": 2},
            "train",
            1,
            lambda parameters: torch.optim.Adam(parameters, lr=0.01),
            seed=4,
        )
    torch.testing.assert_close(loaded_weight, resumed.model.trainable)
    self.assertEqual(1, resumed.step)

  def test_pending_global_step_resets_live_step(self):
    model_dir = tempfile.mkdtemp()
    wrapper = _make_wrapper(model_dir)
    patches = _patch_data()
    with patches[0], patches[1], patches[2]:
      wrapper.train(
          "task",
          1,
          10,
          {"inputs": 2, "targets": 2},
          "train",
          1,
          lambda parameters: torch.optim.SGD(parameters, lr=0.01),
          seed=3,
      )
    wrapper._step = 99
    patches = _patch_data()
    with patches[0], patches[1], patches[2]:
      wrapper.train(
          "task",
          0,
          10,
          {"inputs": 2, "targets": 2},
          "train",
          1,
          lambda parameters: torch.optim.SGD(parameters, lr=0.01),
          seed=3,
      )
    self.assertEqual(1, wrapper.step)

  def test_same_length_changed_data_with_fingerprint_rejects_before_update(self):
    model_dir = tempfile.mkdtemp()
    wrapper = _make_wrapper(model_dir)
    patches = _patch_data(epoch_size=2, batch_offset=0)
    with patches[0], patches[1], patches[2]:
      wrapper.train(
          "task",
          1,
          10,
          {"inputs": 2, "targets": 2},
          "train",
          1,
          lambda parameters: torch.optim.SGD(parameters, lr=0.01),
          data_fingerprint="sha256:old-data",
      )
    resumed = _make_wrapper(model_dir)
    resumed.load_checkpoint(1, resume_training=True)
    weight_before = resumed.model.trainable.detach().clone()
    patches = _patch_data(epoch_size=2, batch_offset=1000)
    with patches[0], patches[1], patches[2]:
      with self.assertRaisesRegex(ValueError, "data_fingerprint"):
        resumed.train(
            "task",
            1,
            10,
            {"inputs": 2, "targets": 2},
            "train",
            1,
            lambda parameters: torch.optim.SGD(parameters, lr=0.01),
            data_fingerprint="sha256:new-data",
        )
    torch.testing.assert_close(weight_before, resumed.model.trainable)
    self.assertEqual(1, resumed.step)


class EpochBatchIteratorTest(absltest.TestCase):

  def test_fresh_epochs_and_mid_epoch_restore(self):
    dataset_calls = []

    def get_dataset(unused_task, unused_lengths, unused_split, **kwargs):
      dataset_calls.append((kwargs["seed"], kwargs["num_epochs"]))
      return kwargs["seed"]

    def batches(seed, *unused_args):
      return [seed * 10, seed * 10 + 1]

    with mock.patch.object(hf_model, "_get_dataset", side_effect=get_dataset), \
         mock.patch.object(hf_model, "tokens_to_batches", side_effect=batches):
      iterator = hf_model._EpochBatchIterator(
          "task", {"inputs": 2}, "train", 1, ("inputs",), _task(), 7
      )
      self.assertEqual(70, next(iterator))
      state = iterator.state_dict()
      self.assertEqual(71, next(iterator))
      self.assertEqual(80, next(iterator))
      restored = hf_model._EpochBatchIterator.from_state_dict(
          "task", {"inputs": 2}, "train", 1, ("inputs",), _task(), state
      )
      self.assertEqual(71, next(restored))
      self.assertEqual(80, next(restored))

    self.assertIn((7, 1), dataset_calls)
    self.assertIn((8, 1), dataset_calls)

  def test_empty_epoch_raises(self):
    with mock.patch.object(hf_model, "_get_dataset", return_value=object()), \
         mock.patch.object(hf_model, "tokens_to_batches", return_value=[]):
      iterator = hf_model._EpochBatchIterator(
          "task", {"inputs": 2}, "train", 1, ("inputs",), _task(), 0
      )
      with self.assertRaisesRegex(ValueError, "produced no batches"):
        next(iterator)


class CheckpointPersistenceTest(absltest.TestCase):

  def test_legacy_load_is_strict_warm_start_and_cpu_safe(self):
    model_dir = tempfile.mkdtemp()
    source = _ToyModel()
    source.trainable.data.fill_(3.5)
    path = os.path.join(model_dir, "model-7.checkpoint")
    torch.save(source.state_dict(), path)
    wrapper = _make_wrapper(model_dir)
    with mock.patch.object(hf_model.torch, "load", wraps=torch.load) as load_spy:
      wrapper.load_checkpoint(7, resume_training=True)
    self.assertEqual("cpu", load_spy.call_args.kwargs["map_location"])
    self.assertEqual(7, wrapper.step)
    self.assertIsNone(wrapper._pending_training_state)
    torch.testing.assert_close(source.trainable, wrapper.model.trainable)

  def test_unknown_or_malformed_envelope_does_not_set_pending_state(self):
    model_dir = tempfile.mkdtemp()
    wrapper = _make_wrapper(model_dir)
    sentinel = {"old": True}
    wrapper._pending_training_state = sentinel
    malformed = hf_model._make_checkpoint_envelope(
        3, copy.deepcopy(wrapper.model.state_dict())
    )
    malformed[hf_model.CHECKPOINT_ENVELOPE_KEY] = 99
    torch.save(malformed, os.path.join(model_dir, "model-3.checkpoint"))
    with self.assertRaisesRegex(ValueError, "Unsupported"):
      wrapper.load_checkpoint(3, resume_training=True)
    self.assertIs(sentinel, wrapper._pending_training_state)

    malformed = hf_model._make_checkpoint_envelope(
        4, copy.deepcopy(wrapper.model.state_dict())
    )
    del malformed["rng_state"]
    torch.save(malformed, os.path.join(model_dir, "model-4.checkpoint"))
    with self.assertRaisesRegex(ValueError, "missing required fields"):
      wrapper.load_checkpoint(4, resume_training=True)
    self.assertIs(sentinel, wrapper._pending_training_state)

  def test_malformed_nested_state_does_not_mutate_live_or_process_state(self):
    for field, corrupt, expected_message in (
        (
            "rng_state",
            lambda envelope: envelope["rng_state"]["numpy"].__setitem__(
                "position", -1
            ),
            "position",
        ),
        (
            "iterator_state",
            lambda envelope: envelope["iterator_state"].__setitem__(
                "epoch", -1
            ),
            "epoch",
        ),
        (
            "training_config",
            lambda envelope: envelope["training_config"].__setitem__(
                "unsupported", "value"
            ),
            "training_config",
        ),
    ):
      with self.subTest(field=field):
        model_dir = tempfile.mkdtemp()
        wrapper = _make_wrapper(model_dir)
        wrapper._step = 41
        sentinel = {"existing": True}
        wrapper._pending_training_state = sentinel
        model_before = copy.deepcopy(wrapper.model.state_dict())
        random.seed(101)
        np.random.seed(101)
        torch.manual_seed(101)
        rng_before = hf_model._capture_rng_state()

        envelope = _valid_resumable_envelope(wrapper, 3)
        corrupt(envelope)
        checkpoint_path = os.path.join(model_dir, "model-3.checkpoint")
        torch.save(envelope, checkpoint_path)

        with self.assertRaisesRegex(ValueError, expected_message):
          wrapper.load_checkpoint(3, resume_training=True)

        self.assertEqual(41, wrapper.step)
        self.assertIs(sentinel, wrapper._pending_training_state)
        for name, value in model_before.items():
          torch.testing.assert_close(value, wrapper.model.state_dict()[name])
        rng_after = hf_model._capture_rng_state()
        self.assertEqual(rng_before["python"], rng_after["python"])
        self.assertEqual(
            rng_before["numpy"]["bit_generator"],
            rng_after["numpy"]["bit_generator"],
        )
        self.assertEqual(
            rng_before["numpy"]["position"], rng_after["numpy"]["position"]
        )
        self.assertTrue(torch.equal(
            rng_before["numpy"]["keys"], rng_after["numpy"]["keys"]
        ))
        self.assertTrue(torch.equal(
            rng_before["torch_cpu"], rng_after["torch_cpu"]
        ))

  def test_duplicate_committed_step_cannot_be_overwritten(self):
    model_dir = tempfile.mkdtemp()
    wrapper = _make_wrapper(model_dir)
    wrapper.save_checkpoint(5)
    checkpoint_path = os.path.join(model_dir, "model-5.checkpoint")
    manifest_path = os.path.join(
        model_dir, hf_model.CHECKPOINT_MANIFEST_FILENAME
    )
    with open(checkpoint_path, "rb") as checkpoint_file:
      checkpoint_before = checkpoint_file.read()
    with open(manifest_path, "rb") as manifest_file:
      manifest_before = manifest_file.read()

    wrapper.model.trainable.data.fill_(9.0)
    with mock.patch.object(hf_model.torch, "save", wraps=torch.save) as save_spy:
      with self.assertRaisesRegex(ValueError, "already committed"):
        wrapper.save_checkpoint(5)
    save_spy.assert_not_called()

    with open(checkpoint_path, "rb") as checkpoint_file:
      self.assertEqual(checkpoint_before, checkpoint_file.read())
    with open(manifest_path, "rb") as manifest_file:
      self.assertEqual(manifest_before, manifest_file.read())
    self.assertFalse(any(
        filename.endswith(".tmp") for filename in os.listdir(model_dir)
    ))

  def test_manifest_failure_poisons_instance_and_payload_stays_unlisted(self):
    model_dir = tempfile.mkdtemp()
    wrapper = _make_wrapper(model_dir)
    wrapper.save_checkpoint(1)
    with mock.patch.object(
        hf_model,
        "_publish_manifest_entry",
        side_effect=OSError("manifest unavailable"),
    ):
      with self.assertRaisesRegex(OSError, "manifest unavailable"):
        wrapper.save_checkpoint(2)

    self.assertTrue(os.path.isfile(os.path.join(model_dir, "model-2.checkpoint")))
    self.assertEqual([1], wrapper.get_all_checkpoint_steps())
    self.assertIsNone(wrapper._pending_training_state)
    self.assertFalse(wrapper._checkpoint_state_usable)
    with self.assertRaisesRegex(RuntimeError, "cannot train or save"):
      wrapper.save_checkpoint(3)

    wrapper.load_checkpoint(1)
    self.assertTrue(wrapper._checkpoint_state_usable)
    wrapper.save_checkpoint(3)
    self.assertEqual([1, 3], wrapper.get_all_checkpoint_steps())

  def test_manifest_rejects_unlisted_and_tampered_before_torch_load(self):
    model_dir = tempfile.mkdtemp()
    wrapper = _make_wrapper(model_dir)
    wrapper.save_checkpoint(1)
    torch.save(_ToyModel().state_dict(), os.path.join(model_dir, "model-2.checkpoint"))
    with mock.patch.object(hf_model.torch, "load", wraps=torch.load) as load_spy:
      with self.assertRaisesRegex(ValueError, "not listed"):
        wrapper.load_checkpoint(2)
    load_spy.assert_not_called()

    checkpoint_path = os.path.join(model_dir, "model-1.checkpoint")
    with open(checkpoint_path, "ab") as checkpoint_file:
      checkpoint_file.write(b"tampered-size")
    with mock.patch.object(hf_model.torch, "load", wraps=torch.load) as load_spy:
      with self.assertRaisesRegex(ValueError, "size"):
        wrapper.load_checkpoint(1)
    load_spy.assert_not_called()

    hash_dir = tempfile.mkdtemp()
    hash_wrapper = _make_wrapper(hash_dir)
    hash_wrapper.save_checkpoint(1)
    checkpoint_path = os.path.join(hash_dir, "model-1.checkpoint")
    with open(checkpoint_path, "r+b") as checkpoint_file:
      first_byte = checkpoint_file.read(1)
      checkpoint_file.seek(0)
      checkpoint_file.write(bytes([first_byte[0] ^ 1]))
    with mock.patch.object(hf_model.torch, "load", wraps=torch.load) as load_spy:
      with self.assertRaisesRegex(ValueError, "SHA-256"):
        hash_wrapper.load_checkpoint(1)
    load_spy.assert_not_called()

  def test_manifest_is_authoritative_migrates_legacy_and_hook_is_best_effort(self):
    model_dir = tempfile.mkdtemp()
    torch.save(_ToyModel().state_dict(), os.path.join(model_dir, "model-1.checkpoint"))
    hook_calls = []

    def hook(entry):
      self.assertTrue(os.path.exists(entry["path"]))
      self.assertTrue(os.path.exists(os.path.join(
          model_dir, hf_model.CHECKPOINT_MANIFEST_FILENAME
      )))
      hook_calls.append(entry)
      raise RuntimeError("observability outage")

    wrapper = _make_wrapper(model_dir, hook=hook)
    wrapper.save_checkpoint(2)
    torch.save(_ToyModel().state_dict(), os.path.join(model_dir, "model-99.checkpoint"))
    with open(os.path.join(model_dir, ".model-100.checkpoint.tmp"), "wb") as temp_file:
      temp_file.write(b"incomplete")

    self.assertEqual([1, 2], wrapper.get_all_checkpoint_steps())
    with open(
        os.path.join(model_dir, hf_model.CHECKPOINT_MANIFEST_FILENAME),
        encoding="utf-8",
    ) as manifest_file:
      manifest = json.load(manifest_file)
    self.assertTrue(manifest["entries"]["1"]["legacy"])
    self.assertEqual(1, manifest["entries"]["2"]["checkpoint_version"])
    self.assertRegex(manifest["entries"]["2"]["sha256"], r"^[0-9a-f]{64}$")
    self.assertLen(hook_calls, 1)
    self.assertEqual("checkpoint", hook_calls[0]["artifact_type"])
    self.assertEqual(hf_model.CHECKPOINT_BACKEND, hook_calls[0]["backend"])

  def test_rng_round_trip(self):
    random.seed(23)
    np.random.seed(23)
    torch.manual_seed(23)
    state = hf_model._capture_rng_state()
    expected = (random.random(), np.random.random(), torch.rand(3))
    random.random()
    np.random.random()
    torch.rand(3)
    hf_model._restore_rng_state(state)
    actual = (random.random(), np.random.random(), torch.rand(3))
    self.assertEqual(expected[0], actual[0])
    self.assertEqual(expected[1], actual[1])
    torch.testing.assert_close(expected[2], actual[2])

  def test_rejects_nonlocal_checkpoint_directory(self):
    with self.assertRaisesRegex(ValueError, "local filesystem"):
      hf_model._require_local_path("gs://bucket/model", "checkpoint writes")


if __name__ == "__main__":
  absltest.main()
