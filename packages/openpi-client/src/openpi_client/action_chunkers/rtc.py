from typing import Dict

import threading
import time
import numpy as np
import tree
from typing_extensions import override

from openpi_client import base_policy as _base_policy
from examples.libero.schemas import ActionChunk
from openpi_client.action_chunkers.action_chunk_broker import ActionChunkBroker


class InferenceTimeRTCBroker(ActionChunkBroker):
    """Wraps a policy to return action chunks one-at-a-time.

    Assumes that the first dimension of all action fields is the chunk size.

    A new inference call to the inner policy is only made when the current
    list of chunks is exhausted.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        action_horizon: int,
        s_min: int = 5,
        d_init: int = 3,
        is_rtc: bool = True,
    ):
        ActionChunkBroker.__init__(self)
        raise NotImplementedError("InferenceTimeRTCBroker is broken, needs to be fixed")
        self._policy = policy

        self._action_horizon = action_horizon
        self._cur_step: int = 0

        self._last_results: Dict[str, np.ndarray] | None = None
        self._last_origin_actions: np.ndarray | None = None
        self._background_results: Dict[str, np.ndarray] | None = None
        self._background_running: bool = False

        self._obs: Dict[str, np.ndarray] | None = None
        self._s_min = s_min
        self._d_init = d_init
        self._is_rtc = is_rtc

        self._infer_thread = threading.Thread(target=self._background_infer)
        self._infer_thread.start()

    def _background_infer(self):
        while True:
            if self._cur_step == self._s_min:
                self._background_running = True
                request_timestamp = time.time()
                self._background_results = self._policy.infer(
                    self._obs,
                    self._last_origin_actions,
                    self._is_rtc,
                    s_param=self._s_min,
                    d_param=self._d_init,
                )
                response_timestamp = time.time()
                self._action_chunks.append(
                    ActionChunk(
                        chunk_length=len(self._last_results["actions"]),
                        request_timestamp=request_timestamp,
                        response_timestamp=response_timestamp,
                    )
                )
                self._background_running = False
            else:
                time.sleep(0.01)

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        env_step = obs["step"]
        if self._last_results is None:
            # TODO: refactor
            request_timestamp = time.time()
            self._last_results = self._policy.infer(obs, None, self._is_rtc, s_param=self._s_min, d_param=self._d_init)
            response_timestamp = time.time()
            self._action_chunks.append(
                ActionChunk(
                    chunk_length=len(self._last_results["actions"]),
                    request_timestamp=request_timestamp,
                    response_timestamp=response_timestamp,
                )
            )
            assert isinstance(self._last_results, dict), "last_results must be a dict"
            self._last_origin_actions = self._last_results["origin_actions"]
            self._last_state = self._last_results["state"]
            self._last_results = {"actions": self._last_results["actions"]}
            self._cur_step = 0
            self._action_chunks[-1].set_start_step(env_step - self._cur_step)

        results = tree.map_structure(lambda x: x[self._cur_step, ...], self._last_results)
        results["action_chunk_index"] = len(self._action_chunks) - 1
        results["action_chunk_current_step"] = self._cur_step
        self._obs = obs
        self._cur_step += 1

        # if current step equals s+d, wait for background inference to complete
        if self._cur_step == self._s_min + self._d_init:
            while self._background_running:
                time.sleep(0.01)
            self._last_origin_actions = self._background_results["origin_actions"]
            self._last_state = self._background_results["state"]
            self._last_results = {"actions": self._background_results["actions"]}
            self._cur_step -= self._s_min
            self._action_chunks[-1].set_start_step(env_step - self._cur_step)  # TODO: double-check

        return results

    @override
    def reset(self) -> None:
        self._policy.reset()
        self._action_chunks = []
        self._last_results = None
        self._last_origin_actions = None
        self._background_results = None
        self._cur_step = 0
