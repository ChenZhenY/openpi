from collections.abc import Sequence
import enum
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
from openpi_client.messages import InferRequest
from openpi_client.messages import InferResponse
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.policies.aloha_policy import make_aloha_example
from openpi.policies.droid_policy import make_droid_example
from openpi.policies.libero_policy import make_libero_example
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"
    LIBERO_REALTIME = "libero_realtime"


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            assert isinstance(self._model, torch.nn.Module), "Model must be a PyTorch model"
            self._model = self._model.to(pytorch_device)
            self._model.eval()
        else:
            self._rng = rng or jax.random.key(0)
            self._model.sample_actions = nnx_utils.module_jit(self._model.sample_actions)
        self._sample_actions = model.sample_actions

    @override
    def infer(
        self,
        obs: dict,
        *,
        prev_action: np.ndarray | None = None,
        use_rtc: bool = False,
        noise: np.ndarray | None = None,
        s_param: int = 5,
        d_param: int = 4,
    ) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(
                lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...],
                inputs,
            )
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        times: dict[str, float] = {}

        actions, times = self._sample_actions(
            sample_rng_or_pytorch_device,
            observation,
            prev_action=prev_action,
            use_rtc=use_rtc,
            s=s_param,
            d=d_param,
            **self._sample_kwargs,
        )
        outputs = {
            "state": observation.state,
            "actions": actions,
        }

        # Ensure we always record a total inference time.
        times.setdefault("infer_total", time.monotonic() - start_time)

        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = times
        return outputs

    def create_batch_obs(self, observations: list[dict]) -> _model.Observation:
        # Stack observations into batch format
        batched_obs = {}

        # FIXME: don't hardcode these values
        keys = (
            "observation/state",
            "observation/image",
            "observation/wrist_image",
            "prompt",
        )
        for key in keys:
            # Stack all values for this key
            values = [obs[key] for obs in observations]
            if isinstance(values[0], np.ndarray):
                batched_obs[key] = np.stack(values, axis=0)
            elif isinstance(values[0], dict):
                # Handle nested dictionaries (like images)
                batched_obs[key] = {}
                for subkey in values[0]:
                    subvalues = [obs[key][subkey] for obs in observations]
                    if isinstance(subvalues[0], np.ndarray):
                        batched_obs[key][subkey] = np.stack(subvalues, axis=0)
                    else:
                        batched_obs[key][subkey] = subvalues
            else:
                batched_obs[key] = values

        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, batched_obs)
        # Apply transforms to batched observation
        inputs = self._input_transform(inputs)

        if not self._is_pytorch_model:
            # Convert to jax.Array (already batched)
            inputs = jax.tree.map(lambda x: jnp.asarray(x), inputs)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device), inputs)

        return _model.Observation.from_dict(inputs)

    def infer_batch(self, requests: list[InferRequest], *, noise: np.ndarray | None = None) -> list[InferResponse]:
        """Run inference on a batch of request.

        Args:
            obs_batch: List of InferRequest objects of the same infer_type.
            noise: Optional noise tensor for batch (shape: batch_size, action_horizon, action_dim)

        Returns:
            List of InferResponse objects, one for each input request.
        """
        if not requests:
            return []

        # Fast batched path for plain observation dicts (non-RTC).
        observation = self.create_batch_obs([request.observation for request in requests])
        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)
            sample_kwargs["noise"] = noise

        # TODO: separate logic for jax and pytorch?
        if not self._is_pytorch_model:
            # Convert to jax.Array (already batched)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            sample_rng_or_pytorch_device = self._pytorch_device

        # TODO: delete timing
        start_time = time.monotonic()
        actions, times = self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)
        times["infer_total"] = time.monotonic() - start_time
        outputs = {
            "state": observation.state,
            "actions": actions,
        }

        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x.detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = times

        # Split batch results back into individual results
        results = []
        for i in range(len(requests)):
            result = {}
            for key, value in outputs.items():
                if key == "policy_timing":
                    result[key] = value  # Timing is shared
                elif isinstance(value, np.ndarray) and len(value.shape) > 0:
                    result[key] = value[i]
                else:
                    result[key] = value
            results.append(result)

        return results

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def make_example(self) -> dict:
        assert "env" in self._metadata, "Environment not set in metadata"
        env = EnvMode(self._metadata["env"])
        if env == EnvMode.ALOHA:
            return make_aloha_example()
        if env == EnvMode.DROID:
            return make_droid_example()
        if env in [EnvMode.LIBERO, EnvMode.LIBERO_REALTIME]:
            return make_libero_example()

        raise ValueError(f"Unknown environment: {env}")

    def make_example_actions(self) -> np.ndarray:
        return self._model.make_example_actions()


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
