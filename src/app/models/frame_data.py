from dataclasses import dataclass

import numpy as np

@dataclass
class FrameDataDescriptor:
    frame_id: str
    shm_idx: int
    timestamp: float
    metadata: dict


@dataclass
class FrameData:
    frame_id: str
    frame: np.ndarray
    timestamp: float
    metadata: dict
    descriptor: FrameDataDescriptor

    def __init__(self, frame_id: str, frame: np.ndarray, timestamp: float | None, metadata: dict, descriptor: FrameDataDescriptor | None = None):
        self.frame_id = frame_id
        self.frame = frame
        self.timestamp = timestamp
        self.metadata = metadata
        self.descriptor = descriptor