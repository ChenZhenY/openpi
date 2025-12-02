from typing import Dict, List

import threading
import time
import numpy as np
import tree
from typing_extensions import override

from openpi_client import base_policy as _base_policy
from dataclasses import dataclass


@dataclass
class ActionChunk:
    chunk_index: int
    request_timestamp: float
    response_timestamp: float
    actions: np.ndarray


class ActionChunkBroker(_base_policy.BasePolicy):
    """Wraps a policy to return action chunks one-at-a-time.

    Assumes that the first dimension of all action fields is the chunk size.

    A new inference call to the inner policy is only made when the current
    list of chunks is exhausted.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        action_horizon: int,
        is_rtc: bool = False,
        s: int = 12,
        d: int = 5,
    ):
        self._policy = policy

        self._action_horizon = action_horizon
        self._cur_step: int = 0

        self._last_results: Dict[str, np.ndarray] | None = None
        self._last_origin_actions: np.ndarray | None = None
        self._background_results: Dict[str, np.ndarray] | None = None
        self._background_running: bool = False
        self._action_chunks: List[ActionChunk] = []

        self._obs: Dict[str, np.ndarray] | None = None
        self._s = s  # 25
        self._d = d  # 10
        self._is_rtc = is_rtc
        print(f"initialized with s: {s}, d: {d}")
        if self._is_rtc:
            self._infer_thread = threading.Thread(target=self._background_infer)
            self._infer_thread.start()

    def _background_infer(self):
        while True:
            if self._cur_step == self._s:
                self._background_running = True
                request_timestamp = time.time()
                self._background_results = self._policy.infer(
                    self._obs,
                    self._last_origin_actions,
                    self._is_rtc,
                    s_param=self._s,
                    d_param=self._d,
                )
                response_timestamp = time.time()
                self._action_chunks.append(
                    ActionChunk(
                        chunk_index=len(self._action_chunks),
                        request_timestamp=request_timestamp,
                        response_timestamp=response_timestamp,
                        actions=self._background_results["actions"],
                    )
                )
                self._background_running = False
            else:
                time.sleep(0.01)

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        if self._is_rtc:
            # init
            if self._last_results is None:
                self._last_results = self._policy.infer(obs, None, self._is_rtc, s_param=self._s, d_param=self._d)
                assert isinstance(self._last_results, dict), "last_results must be a dict"
                self._last_origin_actions = self._last_results["origin_actions"]
                self._last_state = self._last_results["state"]
                self._last_results = {"actions": self._last_results["actions"]}
                self._cur_step = 0

            results = tree.map_structure(lambda x: x[self._cur_step, ...], self._last_results)
            self._obs = obs
            self._cur_step += 1

            # if current step equals s+d, wait for background inference to complete
            if self._cur_step == self._s + self._d:
                while self._background_running:
                    time.sleep(0.01)
                self._last_origin_actions = self._background_results["origin_actions"]
                self._last_state = self._background_results["state"]
                self._last_results = {"actions": self._background_results["actions"]}
                self._cur_step -= self._s

            return results

        else:
            if self._last_results is None:
                self._last_results = self._policy.infer(obs, use_rtc=self._is_rtc, s_param=self._s, d_param=self._d)
                self._cur_step = 0

            self._last_results = {"actions": self._last_results["actions"]}

            results = tree.map_structure(lambda x: x[self._cur_step, ...], self._last_results)
            self._cur_step += 1

            if self._cur_step >= self._action_horizon:
                self._last_results = None

            return results

    @override
    def infer_batch(self, obs_batch: List[Dict]) -> List[Dict]:
        return [self.infer(obs) for obs in obs_batch]

    @override
    def make_example(self) -> Dict:
        return None

    @override
    def reset(self) -> None:
        self._policy.reset()
        self._action_chunks = []
        self._last_results = None
        self._last_origin_actions = None
        self._background_results = None
        self._cur_step = 0

    @property
    def action_chunks(self) -> List[ActionChunk]:
        return self._action_chunks

    @property
    def current_action_chunk(self) -> ActionChunk:
        return self._action_chunks[-1]
