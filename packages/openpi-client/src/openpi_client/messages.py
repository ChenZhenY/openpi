from dataclasses import dataclass
from enum import Enum
import jax
import numpy as np


# TODO: merge with broker types
class InferType(Enum):
    SYNC = "sync"
    INFERENCE_TIME_RTC = "inference_time_rtc"
    TRAIN_TIME_RTC = "train_time_rtc"
    VLASH = "vlash"


@dataclass
class RTCParams:
    prev_action: np.ndarray | jax.Array
    s_param: int
    d_param: int


@dataclass
class VlashParams:
    # TODO:
    pass


@dataclass
class TrainTimeRTCParams:
    # TODO:
    pass


@dataclass
class InferRequest:
    observation: dict
    infer_type: InferType
    params: RTCParams | VlashParams | TrainTimeRTCParams | None = None


@dataclass
class InferResponse:
    actions: np.ndarray
    times: dict
