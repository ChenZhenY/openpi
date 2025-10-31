import abc
from typing import Dict


class BasePolicy(abc.ABC):
    @abc.abstractmethod
    def infer(self, obs: Dict) -> Dict:
        """Infer actions from observations."""

    @abc.abstractmethod
    def infer_batch(self, obs_batch: list[Dict]) -> list[Dict]:
        """Infer actions from a batch of observations."""

    def reset(self) -> None:
        """Reset the policy to its initial state."""
        pass

    @abc.abstractmethod
    def make_example(self) -> Dict:
        """Make an example observation for the policy."""
