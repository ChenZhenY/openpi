from typing import List
from examples.libero.schemas import ActionChunk
from abc import ABC
from openpi_client import base_policy as _base_policy


class ActionChunkBroker(_base_policy.BasePolicy, ABC):
    """Wraps a policy to return action chunks one-at-a-time.

    Assumes that the first dimension of all action fields is the chunk size.

    A new inference call to the inner policy is only made when the current
    list of chunks is exhausted.
    """

    def __init__(self) -> None:
        self._action_chunks: List[ActionChunk] = []

    @property
    def action_chunks(self) -> List[ActionChunk]:
        assert all(chunk.start_step >= 0 for chunk in self._action_chunks), (
            "An action chunk did not have a start step set"
        )
        return self._action_chunks

    @property
    def current_action_chunk(self) -> ActionChunk:
        return self._action_chunks[-1]
