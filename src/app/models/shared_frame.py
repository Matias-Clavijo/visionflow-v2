import logging

import numpy as np
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory

logger = logging.getLogger(__name__)

class SharedFramePool:
    def __init__(self, n_buffers, shape, dtype=np.uint8, name="Processor"):
        self.shape = shape
        self.dtype = dtype
        self.n_buffers = n_buffers
        self.name = name

        # Internal structures for shared memory management
        self._local_arrays = {}  # idx -> np.ndarray (per-process cache)
        self._local_shms = {}    # idx -> SharedMemory (per-process handles)

        # Get multiprocessing context
        ctx = mp.get_context()
        
        # Always use SharedMemory buffers and multiprocessing primitives
        buffer_bytes = int(np.prod(shape) * np.dtype(dtype).itemsize)

        # Create shared memory blocks
        self.shms = []
        self.shm_names = []

        # Compatibility: expose a list-like so len(self.buffers) works in monitors
        self.buffers = [None] * n_buffers
        for _ in range(n_buffers):
            shm = SharedMemory(create=True, size=buffer_bytes)
            self.shms.append(shm)
            self.shm_names.append(shm.name)
            # Warm local cache for the creating process
            arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
            self._local_shms[len(self._local_shms)] = shm
            self._local_arrays[len(self._local_arrays)] = arr

        # Process-safe primitives
        self.locks = [ctx.Lock() for _ in range(n_buffers)]
        self.ref_counts_lock = ctx.Lock()

        # Use a shared array for ref counts; protect with per-buffer lock + global for sums
        self.ref_counts = ctx.Array('i', [0] * n_buffers, lock=False)
        self.free = ctx.Queue()
        for i in range(n_buffers):
            self.free.put(i)

        buffer_size_mb = (np.prod(shape) * np.dtype(dtype).itemsize * n_buffers) / (1024 * 1024)
        logger.info(f"SharedFramePool[MP-Shared] initialized: {n_buffers} buffers, {buffer_size_mb:.2f}MB total")

    def get_buffer(self):
        idx = self.free.get()
        arr = self._get_array(idx)

        return idx, arr

    def acquire(self, idx):
        with self.locks[idx]:
            with self.ref_counts_lock:
                self.ref_counts[idx] = int(self.ref_counts[idx]) + 1

    def release(self, idx):
        should_cleanup = False

        with self.locks[idx]:
            with self.ref_counts_lock:
                current = int(self.ref_counts[idx]) - 1
                self.ref_counts[idx] = current
                new_ref_count = current

                if new_ref_count <= 0:
                    should_cleanup = True
                    self.free.put(idx)

                if new_ref_count < 0:
                    self.ref_counts[idx] = 0

        if should_cleanup:
            self._cleanup_buffer_content(idx)

    def to_numpy(self, idx):
        return self._get_array(idx)

    def _get_array(self, idx):
        arr = self._local_arrays.get(idx)
        if arr is not None:
            return arr

        shm_name = self.shm_names[idx]
        shm = SharedMemory(name=shm_name, create=False)
        arr = np.ndarray(self.shape, dtype=self.dtype, buffer=shm.buf)
        self._local_shms[idx] = shm
        self._local_arrays[idx] = arr
        return arr

    def _cleanup_buffer_content(self, idx):
        try:
            arr = self._get_array(idx)
            arr.fill(0)

        except Exception as e:
            logger.error(f"Error cleaning buffer[{idx}]: {e}")

    def cleanup(self):
        with self.ref_counts_lock:
            ref_counts_copy = list(int(v) for v in self.ref_counts[:])
            active_refs = sum(ref_counts_copy)

        if active_refs > 0:
            logger.warning(f"Cleaning up SharedFramePool with {active_refs} active references still remaining")

            for idx, ref_count in enumerate(ref_counts_copy):
                if ref_count > 0:
                    logger.warning(f"  Buffer[{idx}] has {ref_count} active references")

        logger.info(f"SharedFramePool cleanup - Active refs: {active_refs}")

        try:
            for idx in range(self.n_buffers):
                try:
                    arr = self._get_array(idx)
                    arr.fill(0)
                except Exception:
                    pass
        finally:
            for shm in list(self._local_shms.values()):
                try:
                    shm.close()
                except Exception:
                    pass
            if hasattr(self, 'shms'):
                for shm in self.shms:
                    try:
                        shm.unlink()
                    except FileNotFoundError:
                        pass
                    except Exception:
                        pass

    def __del__(self):
        self.cleanup()