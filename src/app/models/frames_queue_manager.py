import logging
import multiprocessing as mp

import numpy as np
from dataclasses import dataclass

from src.app.models.shared_frame import SharedFramePool

logger = logging.getLogger(__name__)


@dataclass
class FrameDescriptor:
    frame_id: str
    shm_idx: int
    metadata: dict


class FrameQueueManager:
    def __init__(self, pool: SharedFramePool):
        self.pool = pool
        self.queues = {}

        self.ctx = mp.get_context()

    def get_pool(self):
        return self.pool

    def create_queue(self, name, maxsize=10):
        self.queues[name] = self.ctx.Queue(maxsize=maxsize)
        logger.debug(f"Created multiprocessing queue '{name}' with maxsize {maxsize}")
        return self.queues[name]

    def register_queues(self, queue_name, queue):
        self.queues[queue_name] = queue
        return self.queues[queue_name]

    def put_frame(self, frame_id, frame_array, metadata=None):
        metadata = metadata or {}
        idx, shm_arr = self.pool.get_buffer()
        np.copyto(shm_arr, frame_array)
        self.pool.acquire(idx)

        descriptor = FrameDescriptor(
            frame_id=frame_id,
            shm_idx=idx,
            metadata=metadata
        )

        for q in self.queues.values():
            self.pool.acquire(idx)
            q.put(descriptor)

        self.pool.release(idx)

    def put_frame_in_queue(self, frame_id, frame_array, queue_name, metadata=None):
        metadata = metadata or {}
        idx, shm_arr = self.pool.get_buffer()
        np.copyto(shm_arr, frame_array)
        self.pool.acquire(idx)

        descriptor = FrameDescriptor(
            frame_id=frame_id,
            shm_idx=idx,
            metadata=metadata
        )

        self.queues[queue_name].put_nowait(descriptor)


    def get_queue(self, name):
        return self.queues[name]

    def get_queues(self):
        return self.queues
