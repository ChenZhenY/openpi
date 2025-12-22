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
from openpi_client import messages as _messages
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
    LIBERO_PYTORCH = "libero_pytorch"
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
            self._model = self._model.to(pytorch_device)
            self._model.eval()
        else:
            # JAX model setup

            self._rng = rng or jax.random.key(0)
            self._model.embed_prefix = nnx_utils.module_jit(self._model.embed_prefix)
            self._model.prefill = nnx_utils.module_jit(self._model.prefill)
            self._model.flow_matching = nnx_utils.module_jit(self._model.flow_matching)
            self._guided_inference = nnx_utils.module_jit(model.guided_flow_matching)
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
        return_debug_data: bool = False,
    ) -> dict:  # type: ignore[misc]

        raw_obs = None
        if return_debug_data:
            raw_obs = jax.tree.map(lambda x: np.array(x) if hasattr(x, '__array__') else x, obs)
        
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
        # TODO: Only pass RTC parameters (s, d) to JAX, PyTorch doesn't support yet
        if not self._is_pytorch_model:
            sample_kwargs["s"] = s_param
            sample_kwargs["d"] = d_param
        sample_kwargs["return_debug_data"] = return_debug_data
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        times: dict[str, float] = {}
        debug_data: dict | None = None

        if use_rtc:
            if prev_action is None:
                # First RTC call: fall back to normal sampling (but with RTC parameters).
                origin_actions, times, debug_data = self._sample_actions(
                    sample_rng_or_pytorch_device,
                    observation,
                    **self._sample_kwargs,
                )
            else:
                # Subsequent RTC call: use guided_inference. This API returns only actions,
                # so we construct a simple timing dict here.
                prev_action = jnp.asarray(prev_action)[np.newaxis, ...]  # Add batch dimension
                guided_start = time.monotonic()
                origin_actions = self._guided_inference(
                    sample_rng_or_pytorch_device,
                    prev_action,
                    observation,
                    **self._sample_kwargs,
                )
                times["guided_inference"] = time.monotonic() - guided_start

            outputs = {
                "state": inputs["state"],
                "actions": origin_actions,
                "origin_actions": origin_actions,
            }
        else:
            # Non-RTC path: standard sampling with full sample_kwargs (including s/d/noise).
            origin_actions, times, debug_data = self._sample_actions(
                sample_rng_or_pytorch_device,
                observation,
                **sample_kwargs,
            )
            outputs = {
                "state": inputs["state"],
                "actions": origin_actions,
                "origin_actions": origin_actions,
            }

        # Ensure we always record a total inference time.
        times.setdefault("infer_total", time.monotonic() - start_time)

        # Collect data for JAX models (after JIT execution)
        if not self._is_pytorch_model and hasattr(self._model, "output_actions_save"):
            self._model.output_actions_save.append(origin_actions)

        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = times
        
        # Add debug data if requested
        if debug_data is not None:
            # Convert JAX arrays to numpy for serialization
            def to_numpy(x):
                if hasattr(x, "numpy"):
                    return x.numpy()
                elif hasattr(x, "__array__"):
                    return np.asarray(x)
                return x
            debug_data_numpy = jax.tree.map(to_numpy, debug_data)
            # Also save the final post-processed actions (after unnormalization)
            # These are the actual actions sent to the robot
            debug_data_numpy["final_actions"] = outputs["actions"]
            # Save the raw observation (before any transforms) for exact replay
            if raw_obs is not None:
                debug_data_numpy["raw_obs"] = raw_obs
            outputs["debug_data"] = debug_data_numpy
        
        return outputs

    def infer_batch(self, obs_batch: list[dict | _messages.InferRequest], *, noise: np.ndarray | None = None, return_debug_data: bool = False) -> list[dict]:
        """Run inference on a batch of observations.

        This supports three input formats:
        1. Raw observation dicts expected by the model.
        2. Websocket envelopes containing:
           {"observation": obs, "prev_action": ..., "use_rtc": ..., "s_param": ..., "d_param": ...}
        3. InferRequest dataclass objects from the websocket server.

        In case (2) and (3) we delegate to `infer` per-example so that RTC / guided_inference
        behavior matches the single-sample path.

        Args:
            obs_batch: List of observation dictionaries, websocket-style envelopes, or InferRequest objects.
            noise: Optional noise tensor for batch (shape: batch_size, action_horizon, action_dim)
            return_debug_data: Whether to return debug data (obs before/after preprocessing, noise, output)

        Returns:
            List of result dictionaries, one for each input observation.
        """
        if not obs_batch:
            return []

        # Check if inputs are InferRequest dataclass objects
        has_infer_request = any(isinstance(obs, _messages.InferRequest) for obs in obs_batch)
        
        if has_infer_request:
            results: list[dict] = []
            for req in obs_batch:
                if isinstance(req, _messages.InferRequest):
                    inner_obs = req.observation
                    use_rtc = req.infer_type == _messages.InferType.INFERENCE_TIME_RTC
                    prev_action = None
                    s_param = 5
                    d_param = 4
                    if req.params is not None and isinstance(req.params, _messages.RTCParams):
                        prev_action = req.params.prev_action
                        s_param = req.params.s_param
                        d_param = req.params.d_param
                    req_debug_data = req.return_debug_data
                    req_noise = req.noise
                    res = self.infer(
                        inner_obs,
                        prev_action=prev_action,
                        use_rtc=use_rtc,
                        noise=req_noise,
                        s_param=s_param,
                        d_param=d_param,
                        return_debug_data=req_debug_data,
                    )
                else:
                    # Fallback for mixed batches (shouldn't happen normally)
                    res = self.infer(req, return_debug_data=return_debug_data)
                results.append(res)
            return results

        # If inputs look like websocket envelopes (with prev_action / use_rtc / s_param / d_param),
        # use the per-example infer() path so that guided_inference and RTC semantics are respected.
        envelope_keys = {"observation", "prev_action", "use_rtc", "s_param", "d_param", "return_debug_data"}
        has_envelope_like = any(isinstance(obs, dict) and any(k in obs for k in envelope_keys) for obs in obs_batch)

        if has_envelope_like:
            results: list[dict] = []
            for obs in obs_batch:
                # Handle both pure observation dicts and websocket-style envelopes.
                if isinstance(obs, dict) and "observation" in obs:
                    inner_obs = obs["observation"]
                    prev_action = obs.get("prev_action", None)
                    use_rtc = obs.get("use_rtc", False)
                    s_param = obs.get("s_param", 5)
                    d_param = obs.get("d_param", 4)
                    req_debug_data = obs.get("return_debug_data", return_debug_data)
                    res = self.infer(
                        inner_obs,
                        prev_action=prev_action,
                        use_rtc=use_rtc,
                        noise=None,
                        s_param=s_param,
                        d_param=d_param,
                        return_debug_data=req_debug_data,
                    )
                else:
                    # Fallback: treat as raw observation dict with default non-RTC settings.
                    res = self.infer(obs, return_debug_data=return_debug_data)
                results.append(res)
            return results

        # Fast batched path for plain observation dicts (non-RTC).
        first_obs = obs_batch[0]
        batch_size = len(obs_batch)

        # Stack observations into batch format
        batched_obs = {}
        for key in first_obs:
            if key in first_obs:
                # Stack all values for this key
                values = [obs[key] for obs in obs_batch]
                if isinstance(values[0], np.ndarray):
                    batched_obs[key] = np.stack(values, axis=0)
                elif isinstance(values[0], dict):
                    # Handle nested dictionaries (like images)
                    batched_obs[key] = {}
                    for subkey in values[0]:
                        subvalues = [obs[key][subkey] for obs in obs_batch]
                        if isinstance(subvalues[0], np.ndarray):
                            batched_obs[key][subkey] = np.stack(subvalues, axis=0)
                        else:
                            batched_obs[key][subkey] = subvalues
                else:
                    batched_obs[key] = values
            else:
                batched_obs[key] = [obs.get(key, None) for obs in obs_batch]

        # Apply transforms to batched observation
        inputs = jax.tree.map(lambda x: x, batched_obs)
        inputs = self._input_transform(inputs)

        if not self._is_pytorch_model:
            # Convert to jax.Array (already batched)
            inputs = jax.tree.map(lambda x: jnp.asarray(x), inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device), inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        sample_kwargs["return_debug_data"] = return_debug_data
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        actions, times, debug_data = self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)
        times["infer_total"] = time.monotonic() - start_time
        outputs = {
            "state": inputs["state"],
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
        for i in range(batch_size):
            result = {}
            for key, value in outputs.items():
                if key == "policy_timing":
                    result[key] = value  # Timing is shared
                elif isinstance(value, np.ndarray) and len(value.shape) > 0:
                    result[key] = value[i]
                else:
                    result[key] = value
            
            # Add debug data for this batch element if available
            if debug_data is not None:
                def to_numpy(x):
                    if hasattr(x, "numpy"):
                        return x.numpy()
                    elif hasattr(x, "__array__"):
                        return np.asarray(x)
                    return x
                
                # Extract debug data for this batch element
                result_debug = {}
                for debug_key, debug_value in debug_data.items():
                    if isinstance(debug_value, dict):
                        result_debug[debug_key] = {}
                        for subkey, subvalue in debug_value.items():
                            arr = to_numpy(subvalue)
                            if isinstance(arr, np.ndarray) and len(arr.shape) > 0:
                                result_debug[debug_key][subkey] = arr[i]
                            else:
                                result_debug[debug_key][subkey] = arr
                    else:
                        arr = to_numpy(debug_value)
                        if isinstance(arr, np.ndarray) and len(arr.shape) > 0:
                            result_debug[debug_key] = arr[i]
                        else:
                            result_debug[debug_key] = arr
                result["debug_data"] = result_debug
            
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
        if env in [EnvMode.LIBERO, EnvMode.LIBERO_REALTIME, EnvMode.LIBERO_PYTORCH]:
            return make_libero_example()

        raise ValueError(f"Unknown environment: {env}")


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
