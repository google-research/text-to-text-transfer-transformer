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

import copy
import functools
import hashlib
import inspect
import json
import math
import os
import random
import re
import tempfile
import time
from collections.abc import Mapping

from absl import logging
import mesh_tensorflow.transformer.dataset as transformer_dataset
import seqio
import t5.data
from t5.models import utils
from t5.models.t5_model import T5Model
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds
import numpy as np
import torch
import torch.utils.tensorboard

CHECKPOINT_FILE_FORMAT = "model-{}.checkpoint"
CHECKPOINT_ENVELOPE_KEY = "t5_hf_checkpoint_version"
CHECKPOINT_ENVELOPE_VERSION = 1
CHECKPOINT_MANIFEST_FILENAME = "hf-checkpoints.manifest.json"
CHECKPOINT_MANIFEST_VERSION = 1
CHECKPOINT_BACKEND = "t5_hf_pytorch"
ITERATOR_STATE_VERSION = 1
_MAX_DATASET_SEED = 2**31 - 1
_CHECKPOINT_BASENAME_RE = re.compile(r"^model-(\d+)\.checkpoint$")


def _is_int(value):
  return isinstance(value, int) and not isinstance(value, bool)


def _require_local_path(path, operation="checkpoint I/O"):
  """Rejects URI-style paths: atomic publication is local-filesystem only."""
  path = os.fspath(path)
  if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", path):
    raise ValueError(
        f"HfPyTorchModel {operation} only supports local filesystem paths; "
        f"got {path!r}. Use one writer per model_dir."
    )
  return os.path.abspath(path)


def _atomic_write(path, write_fn):
  """Writes and atomically replaces a local file on a POSIX filesystem."""
  directory = os.path.dirname(path)
  os.makedirs(directory, exist_ok=True)
  fd, temporary_path = tempfile.mkstemp(
      prefix="." + os.path.basename(path) + ".", suffix=".tmp", dir=directory
  )
  try:
    with os.fdopen(fd, "wb") as temporary_file:
      write_fn(temporary_file)
      temporary_file.flush()
      os.fsync(temporary_file.fileno())
    os.replace(temporary_path, path)
    temporary_path = None
    try:
      directory_fd = os.open(directory, os.O_RDONLY)
      try:
        os.fsync(directory_fd)
      finally:
        os.close(directory_fd)
    except OSError:
      # Some local filesystems do not support directory fsync.
      pass
  finally:
    if temporary_path is not None:
      try:
        os.unlink(temporary_path)
      except FileNotFoundError:
        pass


def _sha256_file(path):
  digest = hashlib.sha256()
  with open(path, "rb") as checkpoint_file:
    for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _checkpoint_entry(path, step, checkpoint_version, legacy=False):
  entry = {
      "filename": os.path.basename(path),
      "global_step": step,
      "checkpoint_version": checkpoint_version,
      "sha256": _sha256_file(path),
      "bytes": os.path.getsize(path),
  }
  if legacy:
    entry["legacy"] = True
  return entry


def _validate_manifest(manifest):
  if not isinstance(manifest, Mapping):
    raise ValueError("HF checkpoint manifest must be a mapping.")
  if manifest.get("manifest_version") != CHECKPOINT_MANIFEST_VERSION:
    raise ValueError("Unsupported or missing HF checkpoint manifest version.")
  if manifest.get("backend") != CHECKPOINT_BACKEND:
    raise ValueError("HF checkpoint manifest has an unexpected backend.")
  entries = manifest.get("entries")
  if not isinstance(entries, Mapping):
    raise ValueError("HF checkpoint manifest entries must be a mapping.")
  for key, entry in entries.items():
    if not isinstance(key, str) or not key.isdigit() or not isinstance(
        entry, Mapping
    ):
      raise ValueError("Malformed HF checkpoint manifest entry.")
    step = int(key)
    if key != str(step):
      raise ValueError(f"Manifest checkpoint key {key!r} is not canonical.")
    expected_filename = CHECKPOINT_FILE_FORMAT.format(step)
    if entry.get("filename") != expected_filename:
      raise ValueError(f"Malformed filename for manifest checkpoint {step}.")
    if entry.get("global_step") != step:
      raise ValueError(f"Malformed global step for manifest checkpoint {step}.")
    if not _is_int(entry.get("checkpoint_version")):
      raise ValueError(f"Malformed version for manifest checkpoint {step}.")
    allowed_fields = {
        "filename",
        "global_step",
        "checkpoint_version",
        "sha256",
        "bytes",
    }
    if entry.get("legacy") is True:
      allowed_fields.add("legacy")
    if set(entry) != allowed_fields:
      raise ValueError(f"Unsupported fields for manifest checkpoint {step}.")
    if not _is_int(entry.get("bytes")) or entry["bytes"] < 0:
      raise ValueError(f"Malformed size for manifest checkpoint {step}.")
    checksum = entry.get("sha256")
    if not isinstance(checksum, str) or not re.fullmatch(
        r"[0-9a-f]{64}", checksum
    ):
      raise ValueError(f"Malformed checksum for manifest checkpoint {step}.")
  return manifest


def _read_manifest(model_dir):
  manifest_path = os.path.join(model_dir, CHECKPOINT_MANIFEST_FILENAME)
  if not os.path.exists(manifest_path):
    return None
  try:
    with open(manifest_path, "r", encoding="utf-8") as manifest_file:
      return _validate_manifest(json.load(manifest_file))
  except (OSError, json.JSONDecodeError) as exc:
    raise ValueError(f"Unable to read HF checkpoint manifest: {exc}") from exc


def _legacy_manifest(model_dir):
  entries = {}
  for filename in os.listdir(model_dir):
    match = _CHECKPOINT_BASENAME_RE.fullmatch(filename)
    if match is None:
      continue
    step = int(match.group(1))
    path = os.path.join(model_dir, filename)
    if os.path.isfile(path):
      entries[str(step)] = _checkpoint_entry(path, step, 0, legacy=True)
  return {
      "manifest_version": CHECKPOINT_MANIFEST_VERSION,
      "backend": CHECKPOINT_BACKEND,
      "entries": entries,
  }


def _write_manifest(model_dir, manifest):
  manifest_path = os.path.join(model_dir, CHECKPOINT_MANIFEST_FILENAME)

  def _write_manifest(manifest_file):
    manifest_file.write(
        json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
    )
    manifest_file.write(b"\n")

  _atomic_write(manifest_path, _write_manifest)


def _ensure_manifest_exists(model_dir):
  """Creates an authoritative legacy snapshot before publishing new bytes."""
  manifest = _read_manifest(model_dir)
  if manifest is None:
    manifest = _legacy_manifest(model_dir)
    _write_manifest(model_dir, manifest)
  return manifest


def _publish_manifest_entry(model_dir, entry):
  manifest = _read_manifest(model_dir)
  if manifest is None:
    raise RuntimeError("HF checkpoint manifest was not initialized.")
  manifest = copy.deepcopy(manifest)
  step_key = str(entry["global_step"])
  if step_key in manifest["entries"]:
    raise ValueError(
        f"Checkpoint step {entry['global_step']} is already committed and "
        "cannot be overwritten."
    )
  manifest["entries"][step_key] = copy.deepcopy(entry)
  _write_manifest(model_dir, manifest)


def _verify_manifest_entry(model_dir, step, entry):
  """Resolves and verifies one canonical committed checkpoint artifact."""
  path = os.path.join(model_dir, CHECKPOINT_FILE_FORMAT.format(step))
  if entry["filename"] != os.path.basename(path):
    raise ValueError(f"Manifest checkpoint {step} is not canonical.")
  if not os.path.isfile(path):
    raise ValueError(f"Committed checkpoint payload is missing: {path}")
  actual_bytes = os.path.getsize(path)
  if actual_bytes != entry["bytes"]:
    raise ValueError(
        f"Committed checkpoint {step} has size {actual_bytes}, expected "
        f"{entry['bytes']}; refusing to deserialize it."
    )
  actual_sha256 = _sha256_file(path)
  if actual_sha256 != entry["sha256"]:
    raise ValueError(
        f"Committed checkpoint {step} failed SHA-256 verification; refusing "
        "to deserialize it."
    )
  return path


def _resolve_committed_checkpoint(model_dir, step, manifest=None):
  """Returns (path, entry) under manifest or exact legacy semantics."""
  model_dir = _require_local_path(model_dir, "checkpoint reads")
  if not _is_int(step) or step < 0:
    raise ValueError("Checkpoint step must be a nonnegative integer.")
  if manifest is None:
    manifest = _read_manifest(model_dir)
  path = os.path.join(model_dir, CHECKPOINT_FILE_FORMAT.format(step))
  if manifest is None:
    if not os.path.isfile(path):
      raise FileNotFoundError(path)
    return path, None
  entry = manifest["entries"].get(str(step))
  if entry is None:
    raise ValueError(
        f"Checkpoint step {step} is not listed in the authoritative HF "
        "checkpoint manifest."
    )
  return _verify_manifest_entry(model_dir, step, entry), entry


def _torch_load_cpu(path):
  """Loads tensors onto CPU and requests restricted loading when supported."""
  load_kwargs = {"map_location": "cpu"}
  try:
    if "weights_only" in inspect.signature(torch.load).parameters:
      load_kwargs["weights_only"] = True
  except (TypeError, ValueError):
    pass
  return torch.load(path, **load_kwargs)


def _capture_rng_state():
  numpy_state = np.random.get_state()
  return {
      "python": random.getstate(),
      "numpy": {
          "bit_generator": numpy_state[0],
          "keys": torch.as_tensor(
              numpy_state[1].astype(np.int64, copy=True), device="cpu"
          ),
          "position": int(numpy_state[2]),
          "has_gauss": int(numpy_state[3]),
          "cached_gaussian": float(numpy_state[4]),
      },
      "torch_cpu": torch.get_rng_state().cpu(),
      "torch_cuda": (
          [state.cpu() for state in torch.cuda.get_rng_state_all()]
          if torch.cuda.is_available()
          else None
      ),
  }


def _validate_rng_state(state, require_runtime=False):
  """Validates every RNG component without changing process-global RNGs."""
  if not isinstance(state, Mapping):
    raise ValueError("Checkpoint RNG state must be a mapping.")
  if set(state) != {"python", "numpy", "torch_cpu", "torch_cuda"}:
    raise ValueError("Checkpoint RNG state has missing or unexpected fields.")
  python_state = state["python"]
  try:
    random.Random().setstate(python_state)
  except (TypeError, ValueError) as exc:
    raise ValueError("Checkpoint Python RNG state is malformed.") from exc
  numpy_state = state["numpy"]
  if not isinstance(numpy_state, Mapping) or set(numpy_state) != {
      "bit_generator",
      "keys",
      "position",
      "has_gauss",
      "cached_gaussian",
  }:
    raise ValueError("Checkpoint NumPy RNG state is malformed.")
  keys = numpy_state["keys"]
  if (
      not isinstance(numpy_state["bit_generator"], str)
      or numpy_state["bit_generator"] != "MT19937"
      or not isinstance(keys, torch.Tensor)
      or keys.device.type != "cpu"
      or keys.ndim != 1
      or keys.numel() != 624
      or keys.dtype != torch.int64
  ):
    raise ValueError("Checkpoint NumPy RNG state is malformed.")
  if bool(torch.any(keys < 0)) or bool(torch.any(keys > 2**32 - 1)):
    raise ValueError("Checkpoint NumPy RNG keys are outside uint32 range.")
  position = numpy_state["position"]
  has_gauss = numpy_state["has_gauss"]
  cached_gaussian = numpy_state["cached_gaussian"]
  if not _is_int(position) or not 0 <= position <= 624:
    raise ValueError("Checkpoint NumPy RNG position is malformed.")
  if not _is_int(has_gauss) or has_gauss not in (0, 1):
    raise ValueError("Checkpoint NumPy RNG has_gauss is malformed.")
  if (
      isinstance(cached_gaussian, bool)
      or not isinstance(cached_gaussian, (int, float))
      or not math.isfinite(cached_gaussian)
  ):
    raise ValueError("Checkpoint NumPy cached Gaussian is malformed.")
  normalized_numpy_state = (
      numpy_state["bit_generator"],
      keys.numpy().astype(np.uint32, copy=True),
      position,
      has_gauss,
      float(cached_gaussian),
  )
  try:
    np.random.RandomState().set_state(normalized_numpy_state)
  except (TypeError, ValueError) as exc:
    raise ValueError("Checkpoint NumPy RNG state is malformed.") from exc
  torch_cpu_state = state["torch_cpu"]
  if (
      not isinstance(torch_cpu_state, torch.Tensor)
      or torch_cpu_state.device.type != "cpu"
      or torch_cpu_state.dtype != torch.uint8
      or torch_cpu_state.ndim != 1
  ):
    raise ValueError("Checkpoint Torch CPU RNG state must be a CPU byte tensor.")
  try:
    torch.Generator(device="cpu").set_state(torch_cpu_state)
  except RuntimeError as exc:
    raise ValueError("Checkpoint Torch CPU RNG state is malformed.") from exc
  cuda_states = state["torch_cuda"]
  if cuda_states is not None:
    if not isinstance(cuda_states, (list, tuple)) or not all(
        isinstance(cuda_state, torch.Tensor)
        and cuda_state.device.type == "cpu"
        and cuda_state.dtype == torch.uint8
        and cuda_state.ndim == 1
        for cuda_state in cuda_states
    ):
      raise ValueError("Checkpoint CUDA RNG states are malformed.")
    if require_runtime and not torch.cuda.is_available():
      raise ValueError(
          "Exact resume requires CUDA, but this runtime has no CUDA device."
      )
    if require_runtime and len(cuda_states) != torch.cuda.device_count():
      raise ValueError(
          "Exact resume requires the same number of CUDA devices as the "
          "checkpoint."
      )
    if require_runtime:
      try:
        for device_index, cuda_state in enumerate(cuda_states):
          torch.Generator(device=f"cuda:{device_index}").set_state(cuda_state)
      except (RuntimeError, TypeError) as exc:
        raise ValueError("Checkpoint CUDA RNG states are malformed.") from exc
  return python_state, normalized_numpy_state, torch_cpu_state, cuda_states


def _restore_rng_state(state):
  validated = _validate_rng_state(state, require_runtime=True)
  python_state, numpy_state, torch_cpu_state, cuda_states = validated
  try:
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_cpu_state)
    if cuda_states is not None:
      torch.cuda.set_rng_state_all(list(cuda_states))
  except (TypeError, ValueError, RuntimeError) as exc:
    raise ValueError(f"Unable to restore checkpoint RNG state: {exc}") from exc


def _validate_optional_identifier(value, field_name):
  if value is None:
    return None
  if not isinstance(value, str):
    raise ValueError(f"{field_name} must be a string or None.")
  if not value.strip() or len(value) > 4096 or not value.isprintable():
    raise ValueError(
        f"{field_name} must be a nonempty string without control characters."
    )
  return value


def _validate_serialized_tree(value, field_name):
  """Rejects callable/custom nested values from optimizer/scheduler state."""
  if value is None or isinstance(value, (str, bool, int)):
    return
  if isinstance(value, float):
    if not math.isfinite(value):
      raise ValueError(f"Checkpoint {field_name} contains a non-finite float.")
    return
  if isinstance(value, torch.Tensor):
    return
  if isinstance(value, (list, tuple)):
    for item in value:
      _validate_serialized_tree(item, field_name)
    return
  if isinstance(value, Mapping):
    for key, item in value.items():
      if not isinstance(key, (str, int)) or isinstance(key, bool):
        raise ValueError(f"Checkpoint {field_name} has an unsupported key.")
      _validate_serialized_tree(item, field_name)
    return
  raise ValueError(
      f"Checkpoint {field_name} contains unsupported value type "
      f"{type(value).__name__}."
  )


def _validate_optimizer_state_dict(state):
  if not isinstance(state, Mapping) or set(state) != {"state", "param_groups"}:
    raise ValueError("Checkpoint optimizer_state_dict is malformed.")
  if not isinstance(state["state"], Mapping):
    raise ValueError("Checkpoint optimizer state must be a mapping.")
  groups = state["param_groups"]
  if not isinstance(groups, (list, tuple)) or not groups:
    raise ValueError("Checkpoint optimizer param_groups must be a nonempty list.")
  for group in groups:
    if not isinstance(group, Mapping) or not isinstance(
        group.get("params"), (list, tuple)
    ):
      raise ValueError("Checkpoint optimizer parameter group is malformed.")
    if not all(_is_int(parameter_id) for parameter_id in group["params"]):
      raise ValueError("Checkpoint optimizer parameter IDs must be integers.")
  _validate_serialized_tree(state, "optimizer_state_dict")


def _validate_iterator_state(state):
  expected_fields = {
      "version",
      "base_seed",
      "epoch",
      "batches_consumed_in_epoch",
  }
  if not isinstance(state, Mapping) or set(state) != expected_fields:
    raise ValueError("Iterator state has missing or unexpected fields.")
  if state["version"] != ITERATOR_STATE_VERSION:
    raise ValueError(
        f"Unsupported iterator state version: {state['version']!r}."
    )
  if (
      not _is_int(state["base_seed"])
      or not 0 <= state["base_seed"] <= _MAX_DATASET_SEED
  ):
    raise ValueError("Iterator state base_seed is outside the supported range.")
  for field in ("epoch", "batches_consumed_in_epoch"):
    if not _is_int(state[field]) or state[field] < 0:
      raise ValueError(f"Iterator state {field} must be nonnegative integer.")
  if state["base_seed"] + state["epoch"] > _MAX_DATASET_SEED:
    raise ValueError("Iterator epoch exceeds the supported dataset seed range.")


_TRAINING_CONFIG_FIELDS = {
    "mixture_or_task_name",
    "split",
    "sequence_length",
    "batch_size",
    "base_seed",
    "output_feature_names",
    "trainable_parameter_names",
    "optimizer_class",
    "optimizer_param_groups",
    "scheduler_class",
    "scheduler_initial_state",
    "device",
    "data_fingerprint",
    "environment_lock_id",
}


def _validate_training_config(config):
  if not isinstance(config, Mapping) or set(config) != _TRAINING_CONFIG_FIELDS:
    raise ValueError(
        "Checkpoint training_config has missing or unexpected fields."
    )
  for field in ("mixture_or_task_name", "split", "optimizer_class", "device"):
    if not isinstance(config[field], str) or not config[field]:
      raise ValueError(f"Checkpoint training_config {field} is malformed.")
  sequence_length = config["sequence_length"]
  if not isinstance(sequence_length, Mapping) or not sequence_length:
    raise ValueError("Checkpoint training_config sequence_length is malformed.")
  if not all(
      isinstance(key, str) and key and _is_int(value) and value > 0
      for key, value in sequence_length.items()
  ):
    raise ValueError("Checkpoint sequence lengths must be positive integers.")
  if not _is_int(config["batch_size"]) or config["batch_size"] <= 0:
    raise ValueError("Checkpoint training_config batch_size is malformed.")
  if (
      not _is_int(config["base_seed"])
      or not 0 <= config["base_seed"] <= _MAX_DATASET_SEED
  ):
    raise ValueError("Checkpoint training_config base_seed is malformed.")
  for field in ("output_feature_names", "trainable_parameter_names"):
    values = config[field]
    if (
        not isinstance(values, (list, tuple))
        or not values
        or not all(isinstance(value, str) and value for value in values)
        or len(set(values)) != len(values)
    ):
      raise ValueError(f"Checkpoint training_config {field} is malformed.")
  groups = config["optimizer_param_groups"]
  if not isinstance(groups, (list, tuple)) or not groups or not all(
      isinstance(group, Mapping) for group in groups
  ):
    raise ValueError("Checkpoint optimizer_param_groups is malformed.")
  scheduler_class = config["scheduler_class"]
  scheduler_initial_state = config["scheduler_initial_state"]
  if scheduler_class is not None and (
      not isinstance(scheduler_class, str) or not scheduler_class
  ):
    raise ValueError("Checkpoint scheduler_class is malformed.")
  if (scheduler_class is None) != (scheduler_initial_state is None):
    raise ValueError("Checkpoint scheduler configuration is inconsistent.")
  _validate_optional_identifier(config["data_fingerprint"], "data_fingerprint")
  _validate_optional_identifier(
      config["environment_lock_id"], "environment_lock_id"
  )
  _validate_serialized_tree(groups, "training_config optimizer_param_groups")
  _validate_serialized_tree(
      scheduler_initial_state, "training_config scheduler_initial_state"
  )


def _canonical_json_value(value, field_name):
  if value is None or isinstance(value, (str, bool, int, float)):
    return value
  if isinstance(value, torch.Tensor) and value.numel() == 1:
    return value.detach().cpu().item()
  if isinstance(value, (list, tuple)):
    return [_canonical_json_value(item, field_name) for item in value]
  if isinstance(value, Mapping):
    return {
        str(key): _canonical_json_value(item, field_name)
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
    }
  raise ValueError(
      f"Training configuration field {field_name!r} is not JSON-safe: "
      f"{type(value).__name__}."
  )


def _training_config(
    mixture_or_task_name,
    sequence_length,
    split,
    batch_size,
    base_seed,
    output_feature_names,
    trainable_parameter_names,
    optimizer_instance,
    scheduler_instance,
    device,
    data_fingerprint,
    environment_lock_id,
):
  optimizer_groups = []
  for group in optimizer_instance.param_groups:
    optimizer_groups.append(_canonical_json_value(
        {key: value for key, value in group.items() if key != "params"},
        "optimizer_param_groups",
    ))
  scheduler_state = (
      _canonical_json_value(
          scheduler_instance.state_dict(), "scheduler_initial_state"
      )
      if scheduler_instance is not None
      else None
  )
  task_name = (
      mixture_or_task_name
      if isinstance(mixture_or_task_name, str)
      else getattr(mixture_or_task_name, "name", None)
  )
  if not isinstance(task_name, str):
    raise ValueError("Training requires a stable string Task or Mixture name.")
  config = {
      "mixture_or_task_name": task_name,
      "split": str(split),
      "sequence_length": _canonical_json_value(
          dict(sequence_length), "sequence_length"
      ),
      "batch_size": batch_size,
      "base_seed": base_seed,
      "output_feature_names": list(output_feature_names),
      "trainable_parameter_names": list(trainable_parameter_names),
      "optimizer_class": (
          optimizer_instance.__class__.__module__
          + "."
          + optimizer_instance.__class__.__qualname__
      ),
      "optimizer_param_groups": optimizer_groups,
      "scheduler_class": (
          scheduler_instance.__class__.__module__
          + "."
          + scheduler_instance.__class__.__qualname__
          if scheduler_instance is not None
          else None
      ),
      "scheduler_initial_state": scheduler_state,
      "device": str(device),
      "data_fingerprint": data_fingerprint,
      "environment_lock_id": environment_lock_id,
  }
  _validate_training_config(config)
  return config


def _make_checkpoint_envelope(
    global_step,
    model_state_dict,
    optimizer_state_dict=None,
    scheduler_state_dict=None,
    training_config=None,
    iterator_state=None,
    rng_state=None,
):
  if not _is_int(global_step) or global_step < 0:
    raise ValueError("Checkpoint global_step must be a nonnegative integer.")
  training_parts = (
      optimizer_state_dict,
      training_config,
      iterator_state,
      rng_state,
  )
  if any(part is None for part in training_parts) and not all(
      part is None for part in training_parts
  ):
    raise ValueError(
        "A resumable checkpoint requires optimizer, training configuration, "
        "iterator, and RNG state."
    )
  envelope = {
      CHECKPOINT_ENVELOPE_KEY: CHECKPOINT_ENVELOPE_VERSION,
      "global_step": global_step,
      "model_state_dict": model_state_dict,
      "optimizer_state_dict": optimizer_state_dict,
      "scheduler_state_dict": scheduler_state_dict,
      "training_config": training_config,
      "iterator_state": iterator_state,
      "rng_state": rng_state,
  }
  _parse_checkpoint_payload(envelope, global_step)
  return envelope


def _parse_checkpoint_payload(payload, filename_step):
  """Validates a payload and returns (model state, resumable envelope)."""
  if not isinstance(payload, Mapping):
    raise ValueError("Checkpoint payload must be a mapping.")
  if CHECKPOINT_ENVELOPE_KEY not in payload:
    return payload, None
  required_fields = {
      CHECKPOINT_ENVELOPE_KEY,
      "global_step",
      "model_state_dict",
      "optimizer_state_dict",
      "scheduler_state_dict",
      "training_config",
      "iterator_state",
      "rng_state",
  }
  missing_fields = required_fields.difference(payload)
  unexpected_fields = set(payload).difference(required_fields)
  if missing_fields:
    raise ValueError(
        "Checkpoint envelope is missing required fields: "
        + ", ".join(sorted(missing_fields))
    )
  if unexpected_fields:
    raise ValueError(
        "Checkpoint envelope has unsupported fields: "
        + ", ".join(sorted(unexpected_fields))
    )
  version = payload[CHECKPOINT_ENVELOPE_KEY]
  if not _is_int(version) or version != CHECKPOINT_ENVELOPE_VERSION:
    raise ValueError(f"Unsupported HF checkpoint version: {version!r}.")
  global_step = payload["global_step"]
  if not _is_int(global_step) or global_step < 0:
    raise ValueError("Checkpoint global_step must be a nonnegative integer.")
  if global_step != filename_step:
    raise ValueError(
        f"Checkpoint filename step {filename_step} does not match envelope "
        f"global_step {global_step}."
    )
  if not isinstance(payload["model_state_dict"], Mapping):
    raise ValueError("Checkpoint model_state_dict must be a mapping.")
  training_parts = (
      payload["optimizer_state_dict"],
      payload["training_config"],
      payload["iterator_state"],
      payload["rng_state"],
  )
  if any(part is None for part in training_parts) and not all(
      part is None for part in training_parts
  ):
    raise ValueError("Checkpoint has incomplete resumable training state.")
  if all(part is None for part in training_parts):
    if payload["scheduler_state_dict"] is not None:
      raise ValueError("Weights-only checkpoint has unexpected scheduler state.")
    return payload["model_state_dict"], None
  _validate_optimizer_state_dict(payload["optimizer_state_dict"])
  _validate_training_config(payload["training_config"])
  _validate_iterator_state(payload["iterator_state"])
  _validate_rng_state(payload["rng_state"])
  if payload["iterator_state"]["base_seed"] != payload["training_config"][
      "base_seed"
  ]:
    raise ValueError(
        "Checkpoint iterator base_seed does not match training_config."
    )
  if payload["scheduler_state_dict"] is not None and not isinstance(
      payload["scheduler_state_dict"], Mapping
  ):
    raise ValueError("Checkpoint scheduler_state_dict must be a mapping or None.")
  if payload["scheduler_state_dict"] is not None:
    _validate_serialized_tree(
        payload["scheduler_state_dict"], "scheduler_state_dict"
    )
  if (payload["scheduler_state_dict"] is None) != (
      payload["training_config"]["scheduler_class"] is None
  ):
    raise ValueError(
        "Checkpoint scheduler state and training configuration disagree."
    )
  return payload["model_state_dict"], payload


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
                 shuffle=True,
                 seed=None,
                 num_epochs=1):
  """Get a tf.data.Dataset for a given Task or Mixture.

  Args:
    mixture_or_task_or_name: Task or Mixture or str, the name of the Mixture or
      Task to train on or the Tasks or Mixture object itself.
      Must be pre-registered in the global `t5.data.TaskRegistry` or
      `t5.data.MixtureRegistry.`
    sequence_length: dict of int, a dict mapping feature name to length.
    split: str or `tensorflow_datasets.Split`, the data split to load.
    shuffle: boolean, whether to shuffle the dataset.
    seed: optional int, the deterministic dataset seed.
    num_epochs: optional int, the finite number of dataset epochs.

  Returns:
    A generator that produces batches of numpy examples.
  """
  if isinstance(mixture_or_task_or_name, str):
    task = seqio.get_mixture_or_task(mixture_or_task_or_name)
  else:
    task = mixture_or_task_or_name

  return task.get_dataset(
      sequence_length,
      split=split,
      shuffle=shuffle,
      seed=seed,
      num_epochs=num_epochs,
  )


class _EpochBatchIterator:
  """Finite, seed-aware epochs with a serializable next-batch cursor."""

  def __init__(
      self,
      mixture_or_task_name,
      sequence_length,
      split,
      batch_size,
      output_feature_names,
      task,
      base_seed,
      state=None,
  ):
    if not _is_int(base_seed) or not 0 <= base_seed <= _MAX_DATASET_SEED:
      raise ValueError(
          f"seed must be a non-boolean integer in [0, {_MAX_DATASET_SEED}]."
      )
    self._mixture_or_task_name = mixture_or_task_name
    self._sequence_length = sequence_length
    self._split = split
    self._batch_size = batch_size
    self._output_feature_names = tuple(output_feature_names)
    self._task = task
    self._base_seed = base_seed
    self._epoch = 0
    self._batches_consumed_in_epoch = 0
    if state is not None:
      self._restore_cursor(state)
    self._build_epoch()
    self._advance_to_cursor()

  @classmethod
  def from_state_dict(
      cls,
      mixture_or_task_name,
      sequence_length,
      split,
      batch_size,
      output_feature_names,
      task,
      state,
  ):
    if not isinstance(state, Mapping):
      raise ValueError("Iterator state must be a mapping.")
    return cls(
        mixture_or_task_name,
        sequence_length,
        split,
        batch_size,
        output_feature_names,
        task,
        state.get("base_seed"),
        state=state,
    )

  def _restore_cursor(self, state):
    expected_fields = {
        "version",
        "base_seed",
        "epoch",
        "batches_consumed_in_epoch",
    }
    if not isinstance(state, Mapping) or set(state) != expected_fields:
      raise ValueError("Iterator state has missing or unexpected fields.")
    if state["version"] != ITERATOR_STATE_VERSION:
      raise ValueError(
          f"Unsupported iterator state version: {state['version']!r}."
      )
    if state["base_seed"] != self._base_seed:
      raise ValueError("Iterator state base seed does not match training seed.")
    for field in ("epoch", "batches_consumed_in_epoch"):
      if not _is_int(state[field]) or state[field] < 0:
        raise ValueError(f"Iterator state {field} must be nonnegative integer.")
    if self._base_seed + state["epoch"] > _MAX_DATASET_SEED:
      raise ValueError("Iterator epoch exceeds the supported dataset seed range.")
    self._epoch = state["epoch"]
    self._batches_consumed_in_epoch = state["batches_consumed_in_epoch"]

  def _build_epoch(self):
    epoch_seed = self._base_seed + self._epoch
    if epoch_seed > _MAX_DATASET_SEED:
      raise ValueError("Training epoch exceeds the supported dataset seed range.")
    dataset = _get_dataset(
        self._task,
        self._sequence_length,
        self._split,
        shuffle=True,
        seed=epoch_seed,
        num_epochs=1,
    )
    batches = tokens_to_batches(
        dataset,
        self._sequence_length,
        self._batch_size,
        self._output_feature_names,
        self._task,
    )
    self._iterator = iter(batches)

  def _advance_to_cursor(self):
    for _ in range(self._batches_consumed_in_epoch):
      try:
        next(self._iterator)
      except StopIteration as exc:
        raise ValueError(
            "The restored dataset exhausted before its saved iterator cursor; "
            "the data snapshot or preprocessing configuration changed."
        ) from exc

  def __iter__(self):
    return self

  def __next__(self):
    try:
      batch = next(self._iterator)
    except StopIteration:
      if self._batches_consumed_in_epoch == 0:
        raise ValueError(
            f"Training dataset produced no batches for epoch {self._epoch}."
        )
      self._epoch += 1
      self._batches_consumed_in_epoch = 0
      self._build_epoch()
      try:
        batch = next(self._iterator)
      except StopIteration as exc:
        raise ValueError(
            f"Training dataset produced no batches for epoch {self._epoch}."
        ) from exc
    self._batches_consumed_in_epoch += 1
    return batch

  def state_dict(self):
    return {
        "version": ITERATOR_STATE_VERSION,
        "base_seed": self._base_seed,
        "epoch": self._epoch,
        "batches_consumed_in_epoch": self._batches_consumed_in_epoch,
    }


class HfPyTorchModel(T5Model):
  """Wrapper class for Hugging Face Transformers PyTorch T5 model."""

  def __init__(
      self, model_spec, model_dir, device, checkpoint_manifest_hook=None
  ):
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
      checkpoint_manifest_hook: optional best-effort callback invoked after a
        checkpoint and its local manifest entry have both been committed.
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

    self._model_dir = _require_local_path(model_dir, "checkpoint storage")
    os.makedirs(self._model_dir, exist_ok=True)
    self._writer = torch.utils.tensorboard.writer.SummaryWriter(self._model_dir)
    self._device = device
    self._model.to(self._device)
    self._step = 0
    self._pending_training_state = None
    self._checkpoint_state_usable = True
    self._checkpoint_manifest_hook = checkpoint_manifest_hook
    self.load_latest_checkpoint()
    self.to_tensor = functools.partial(
        torch.as_tensor, device=self._device, dtype=torch.long)

  @property
  def model(self):
    return self._model

  @property
  def step(self):
    return self._step

  def save_checkpoint(
      self,
      step,
      *,
      optimizer_instance=None,
      scheduler_instance=None,
      training_config=None,
      epoch_iterator=None,
  ):
    """Atomically save a versioned checkpoint to the local `model_dir`.

    Direct calls create a weights-only snapshot. Training passes all resumable
    components. This P0 implementation supports one writer per local model
    directory; it does not provide a distributed-writer lock.

    Args:
      step: int, the current training step.
      optimizer_instance: optional live optimizer used for a training save.
      scheduler_instance: optional live scheduler used for a training save.
      training_config: optional exact-resume configuration mapping.
      epoch_iterator: optional serializable training batch iterator.
    """
    if not getattr(self, "_checkpoint_state_usable", True):
      raise RuntimeError(
          "This model instance cannot train or save after checkpoint payload "
          "publication succeeded but manifest publication failed. Load a "
          "committed checkpoint successfully before continuing."
      )
    if not _is_int(step) or step < 0:
      raise ValueError("Checkpoint step must be a nonnegative integer.")
    resumable_parts = (optimizer_instance, training_config, epoch_iterator)
    if any(part is None for part in resumable_parts) and not all(
        part is None for part in resumable_parts
    ):
      raise ValueError(
          "Training checkpoint saves require optimizer, configuration, and "
          "iterator state together."
      )
    if optimizer_instance is None and scheduler_instance is not None:
      raise ValueError("A scheduler cannot be saved without an optimizer.")
    if optimizer_instance is None:
      envelope = _make_checkpoint_envelope(
          step, self._model.state_dict()
      )
    else:
      envelope = _make_checkpoint_envelope(
          step,
          self._model.state_dict(),
          optimizer_state_dict=optimizer_instance.state_dict(),
          scheduler_state_dict=(
              scheduler_instance.state_dict()
              if scheduler_instance is not None
              else None
          ),
          training_config=training_config,
          iterator_state=epoch_iterator.state_dict(),
          rng_state=_capture_rng_state(),
      )
    path = os.path.join(self._model_dir, CHECKPOINT_FILE_FORMAT.format(step))
    path = _require_local_path(path, "checkpoint writes")
    manifest = _ensure_manifest_exists(self._model_dir)
    if str(step) in manifest["entries"]:
      raise ValueError(
          f"Checkpoint step {step} is already committed and cannot be "
          "overwritten."
      )

    def _write_checkpoint(checkpoint_file):
      torch.save(envelope, checkpoint_file)

    _atomic_write(path, _write_checkpoint)
    try:
      entry = _checkpoint_entry(path, step, CHECKPOINT_ENVELOPE_VERSION)
      _publish_manifest_entry(self._model_dir, entry)
    except Exception:
      # The payload remains unlisted and therefore invisible. The live object
      # can no longer claim to represent a durable committed training state.
      self._pending_training_state = None
      self._checkpoint_state_usable = False
      raise
    if optimizer_instance is not None:
      self._pending_training_state = copy.deepcopy(envelope)
    else:
      self._pending_training_state = None
    if self._checkpoint_manifest_hook is not None:
      hook_entry = copy.deepcopy(entry)
      hook_entry.update({
          "artifact_type": "checkpoint",
          "backend": CHECKPOINT_BACKEND,
          "path": path,
      })
      try:
        self._checkpoint_manifest_hook(hook_entry)
      except Exception:  # pylint: disable=broad-except
        logging.warning(
            "HF checkpoint manifest hook failed after committing %s.",
            path,
            exc_info=True,
        )

  def load_checkpoint(self, step, model_dir=None, *, resume_training=False):
    """Load the model parameters from a checkpoint at a given step.

    Args:
      step: int, load the checkpoint from this training step.
      model_dir: str, the directory of the checkpoint to load or None to use
        this model's directory.
      resume_training: whether to retain complete optimizer, scheduler, RNG,
        and iterator state for exact restoration by the next `train` call.
    """
    if not _is_int(step) or step < 0:
      raise ValueError("Checkpoint step must be a nonnegative integer.")
    model_dir = _require_local_path(
        model_dir or self._model_dir, "checkpoint reads"
    )
    path, unused_entry = _resolve_committed_checkpoint(model_dir, step)
    logging.info("Loading from %s", path)
    payload = _torch_load_cpu(path)
    model_state_dict, training_state = _parse_checkpoint_payload(payload, step)
    self._validate_model_state_dict(model_state_dict)
    self._model.load_state_dict(model_state_dict, strict=True)
    self._step = (
        training_state["global_step"] if training_state is not None else step
    )
    if resume_training and training_state is not None:
      self._pending_training_state = copy.deepcopy(training_state)
    else:
      self._pending_training_state = None
      if resume_training and training_state is None:
        logging.info(
            "Checkpoint %s is a strict weights-only warm start; optimizer, "
            "scheduler, RNG, and iterator state will start fresh.",
            path,
        )
    self._checkpoint_state_usable = True

  def _validate_model_state_dict(self, model_state_dict):
    """Checks strict model-state compatibility without changing live tensors."""
    current_state = self._model.state_dict()
    if set(model_state_dict) != set(current_state):
      missing = sorted(set(current_state).difference(model_state_dict))
      unexpected = sorted(set(model_state_dict).difference(current_state))
      raise ValueError(
          "Checkpoint model_state_dict keys do not strictly match the model; "
          f"missing={missing}, unexpected={unexpected}."
      )
    for name, current_value in current_state.items():
      saved_value = model_state_dict[name]
      if isinstance(current_value, torch.Tensor):
        if not isinstance(saved_value, torch.Tensor):
          raise ValueError(f"Checkpoint model value {name!r} is not a tensor.")
        if saved_value.shape != current_value.shape:
          raise ValueError(
              f"Checkpoint model value {name!r} has shape "
              f"{tuple(saved_value.shape)}, expected {tuple(current_value.shape)}."
          )
        if saved_value.dtype != current_value.dtype:
          raise ValueError(
              f"Checkpoint model value {name!r} has dtype "
              f"{saved_value.dtype}, expected {current_value.dtype}."
          )

  def _is_exact_pending_checkpoint(self, step, training_config):
    pending_state = self._pending_training_state
    if (
        pending_state is None
        or pending_state.get("global_step") != step
        or pending_state.get("training_config") != training_config
    ):
      return False
    current_state = self._model.state_dict()
    saved_state = pending_state.get("model_state_dict")
    if not isinstance(saved_state, Mapping) or set(saved_state) != set(current_state):
      return False
    return all(
        isinstance(saved_state[name], torch.Tensor)
        and torch.equal(saved_state[name].detach().cpu(), value.detach().cpu())
        for name, value in current_state.items()
        if isinstance(value, torch.Tensor)
    )

  def _save_training_checkpoint(
      self,
      optimizer_instance,
      scheduler_instance,
      training_config,
      epoch_iterator,
  ):
    manifest = _read_manifest(self._model_dir)
    if manifest is not None and str(self._step) in manifest["entries"]:
      if self._is_exact_pending_checkpoint(self._step, training_config):
        _verify_manifest_entry(
            self._model_dir, self._step, manifest["entries"][str(self._step)]
        )
        logging.info(
            "Skipping checkpoint step %s because training is continuing from "
            "that exact committed state.",
            self._step,
        )
        return
      raise ValueError(
          f"Checkpoint step {self._step} is already committed, but the live "
          "model is not continuing from that exact committed state."
      )
    self.save_checkpoint(
        self._step,
        optimizer_instance=optimizer_instance,
        scheduler_instance=scheduler_instance,
        training_config=training_config,
        epoch_iterator=epoch_iterator,
    )

  def get_all_checkpoint_steps(self, model_dir=None):
    """Retrieve the steps corresponding to all checkpoints in `model_dir`.

    Args:
      model_dir: str, the directory of the checkpoints or None to use this
        model's directory.

    Returns:
      A list of ints corresponding to all checkpoint steps, or None if there
        are no checkpoints in the model directory.
    """
    model_dir = _require_local_path(
        model_dir or self._model_dir, "checkpoint discovery"
    )
    manifest = _read_manifest(model_dir)
    if manifest is not None:
      steps = []
      for step_string, entry in manifest["entries"].items():
        step = int(step_string)
        _verify_manifest_entry(model_dir, step, entry)
        steps.append(step)
      return sorted(steps) or None
    if not os.path.isdir(model_dir):
      return None
    steps = []
    for filename in os.listdir(model_dir):
      match = _CHECKPOINT_BASENAME_RE.fullmatch(filename)
      if match is not None and os.path.isfile(os.path.join(model_dir, filename)):
        steps.append(int(match.group(1)))
    return sorted(steps) or None

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
      self.load_checkpoint(latest_step, resume_training=True)

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
      seed=0,
      data_fingerprint=None,
      environment_lock_id=None,
  ):
    """Train the model on the given Mixture or Task.

    Args:
      mixture_or_task_name: str, the name of the Mixture or Task to train on.
        Must be pre-registered in the global `t5.data.TaskRegistry` or
        `t5.data.MixtureRegistry.`
      steps: int, the number of additional optimizer updates to perform.
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
      seed: deterministic base seed for finite, freshly constructed epochs.
      data_fingerprint: optional immutable external-data identifier included in
        the exact-resume contract. Supply this when data can change without the
        local task configuration changing (for example, a remote data source).
      environment_lock_id: optional immutable identifier for the exact runtime,
        dependency, and deterministic-device configuration used for training.
    """
    if not getattr(self, "_checkpoint_state_usable", True):
      raise RuntimeError(
          "This model instance cannot train or save after checkpoint payload "
          "publication succeeded but manifest publication failed. Load a "
          "committed checkpoint successfully before continuing."
      )
    if not _is_int(steps) or steps < 0:
      raise ValueError("steps must be a nonnegative integer.")
    if not _is_int(save_steps) or save_steps <= 0:
      raise ValueError("save_steps must be a positive integer.")
    if not _is_int(seed) or not 0 <= seed <= _MAX_DATASET_SEED:
      raise ValueError(
          f"seed must be a non-boolean integer in [0, {_MAX_DATASET_SEED}]."
      )
    if not _is_int(batch_size) or batch_size <= 0:
      raise ValueError("batch_size must be a positive integer.")
    data_fingerprint = _validate_optional_identifier(
        data_fingerprint, "data_fingerprint"
    )
    environment_lock_id = _validate_optional_identifier(
        environment_lock_id, "environment_lock_id"
    )
    if data_fingerprint is None:
      logging.warning(
          "No data_fingerprint was supplied. Exact resume assumes the "
          "external dataset is immutable; data drift cannot be detected."
      )
    pending_state = self._pending_training_state
    if pending_state is not None:
      if not isinstance(pending_state, Mapping):
        raise ValueError("Pending checkpoint training state must be a mapping.")
      _, pending_state = _parse_checkpoint_payload(
          pending_state, pending_state.get("global_step")
      )
    task = seqio.get_mixture_or_task(mixture_or_task_name)
    output_feature_names = tuple(task.output_features)
    trainable_named_parameters = [
        (name, parameter)
        for name, parameter in self._model.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable_named_parameters:
      raise ValueError("No model parameters have requires_grad=True.")
    optimizer_instance = optimizer([
        parameter for _, parameter in trainable_named_parameters
    ])
    scheduler_instance = (
        learning_rate_scheduler(optimizer_instance)
        if learning_rate_scheduler is not None
        else None
    )
    current_training_config = _training_config(
        mixture_or_task_name,
        sequence_length,
        split,
        batch_size,
        seed,
        output_feature_names,
        [name for name, _ in trainable_named_parameters],
        optimizer_instance,
        scheduler_instance,
        self._device,
        data_fingerprint,
        environment_lock_id,
    )
    if pending_state is not None:
      saved_training_config = pending_state["training_config"]
      if saved_training_config != current_training_config:
        differing_fields = sorted(
            key
            for key in set(saved_training_config).union(current_training_config)
            if saved_training_config.get(key) != current_training_config.get(key)
        )
        raise ValueError(
            "Cannot exactly resume because the training configuration changed: "
            + ", ".join(differing_fields)
        )
      saved_scheduler_state = pending_state["scheduler_state_dict"]
      if (saved_scheduler_state is None) != (scheduler_instance is None):
        raise ValueError(
            "Cannot exactly resume because scheduler presence changed."
        )
      optimizer_instance.load_state_dict(pending_state["optimizer_state_dict"])
      if scheduler_instance is not None:
        scheduler_instance.load_state_dict(saved_scheduler_state)
      epoch_iterator = _EpochBatchIterator.from_state_dict(
          mixture_or_task_name,
          sequence_length,
          split,
          batch_size,
          output_feature_names,
          task,
          pending_state["iterator_state"],
      )
      self._validate_model_state_dict(pending_state["model_state_dict"])
      self._model.load_state_dict(
          pending_state["model_state_dict"], strict=True
      )
      self._step = pending_state["global_step"]
      # Dataset construction and cursor replay may touch global RNGs. Restore
      # them only after those operations and immediately before training.
      _restore_rng_state(pending_state["rng_state"])
    else:
      epoch_iterator = _EpochBatchIterator(
          mixture_or_task_name,
          sequence_length,
          split,
          batch_size,
          output_feature_names,
          task,
          seed,
      )

    self._model.train()
    now = time.time()
    for _ in range(steps):
      if self._step % save_steps == 0:
        logging.info("Saving checkpoint for step %s", self._step)
        self._save_training_checkpoint(
            optimizer_instance,
            scheduler_instance,
            current_training_config,
            epoch_iterator,
        )

      batch = next(epoch_iterator)
      optimizer_instance.zero_grad(set_to_none=True)
      targets = self.to_tensor(batch["targets"])
      targets_mask = self.to_tensor(batch["targets_mask"])
      labels = targets.masked_fill(~targets_mask.to(torch.bool), -100)
      outputs = self._model(
          input_ids=self.to_tensor(batch["inputs"]),
          attention_mask=self.to_tensor(batch["inputs_mask"]),
          decoder_attention_mask=targets_mask,
          labels=labels,
      )
      loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
      loss.backward()
      optimizer_instance.step()
      if scheduler_instance is not None:
        scheduler_instance.step()
      self._step += 1

      self._writer.add_scalar(
          "loss", loss.detach().cpu().numpy(), self._step
      )
      self._writer.add_scalar("step/s", 1 / (time.time() - now), self._step)
      now = time.time()

    logging.info("Saving final checkpoint for step %s", self._step)
    self._save_training_checkpoint(
        optimizer_instance,
        scheduler_instance,
        current_training_config,
        epoch_iterator,
    )

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
    load_selected_checkpoints = checkpoint_steps is not None

    def _predict_from_tasks(tasks, vocabulary, checkpoint_step, sequence_length,
                            datasets, **unused_kwargs):

      if isinstance(vocabulary, tuple):
        vocab = vocabulary[1]

      if load_selected_checkpoints or checkpoint_step != self._step:
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
      checkpoint_steps = [
          self.get_latest_checkpoint_step()
          if checkpoint_steps == -1
          else checkpoint_steps
      ]
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
    elif isinstance(vocabulary, seqio.Vocabulary):
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
