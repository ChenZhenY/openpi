from typing import Dict, List

import threading
import time
from typing_extensions import override

from openpi_client import base_policy as _base_policy
from examples.libero.schemas import ActionChunk


class SyncBroker(_base_policy.BasePolicy):
    """Wraps a policy to return action chunks asynchronously.

    The policy is called synchronously in the background thread whenever the current action chunk is exhausted.
    """

    def __init__(self, policy: _base_policy.BasePolicy, action_horizon: int):
        self._policy = policy
        self._action_horizon = action_horizon
        self._cur_step: int = -1

        self._action_chunks: List[ActionChunk] = []

        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._infer_thread = threading.Thread(target=self._background_infer)
        self._infer_thread.start()

    def _background_infer(self):
        # TODO: figure out best way, not sure if sleeping makes sense
        while True:
            with self._condition:
                self._condition.wait()

                obs = self._obs
                infer_step = self._cur_step

            # TODO: might get pre-empted here
            request_timestamp = time.time()
            response = self._policy.infer(obs)
            actions = response["actions"]
            response_timestamp = time.time()

            action_chunk = ActionChunk(
                actions=actions,
                request_timestamp=request_timestamp,
                response_timestamp=response_timestamp,
                start_step=infer_step,
            )
            with self._condition:
                self._action_chunks.append(action_chunk)
                self._condition.notify()

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        with self._condition:
            self._obs = obs
            self._cur_step = obs["step"]

            # Assume no latency for step 0, so we wait until we have an action chunk
            if len(self._action_chunks) == 0:
                assert self._cur_step == 0, "First inference should be for step 0"
                self._condition.notify()
                self._condition.wait()

        with self._condition:
            if self._cur_step < self.current_action_chunk.start_step + self.current_action_chunk.chunk_length:
                current_action_index = self._cur_step - self.current_action_chunk.start_step
            else:
                self._condition.notify()
                # return the last action in the last action chunk if we have run out of actions
                current_action_index = self.current_action_chunk.chunk_length - 1

        results = {
            "actions": self.current_action_chunk.get_action(current_action_index),
            "action_chunk_index": len(self._action_chunks) - 1,
            "action_chunk_current_step": current_action_index,
        }

        return results

    @override
    def reset(self) -> None:
        with self._lock:
            self._policy.reset()
            self._action_chunks = []
            self._cur_step = -1

    @property
    def action_chunks(self) -> List[ActionChunk]:
        assert all(chunk.start_step >= 0 for chunk in self._action_chunks), (
            "An action chunk did not have a start step set"
        )
        return self._action_chunks

    @property
    def current_action_chunk(self) -> ActionChunk:
        return self._action_chunks[-1]
