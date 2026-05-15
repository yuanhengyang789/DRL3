
import random
from collections import deque

import numpy as np


class ReplayBuffer(object):
    def __init__(self, buffer_size, random_seed=123):
        """
        The right side of the deque contains the most recent experiences.
        """
        self.buffer_size = int(buffer_size)
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, s, a, r, t, s2):
        experience = (s, a, r, t, s2)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def add_global(self, s_global, a_joint, r, t, s2_global):
        self.add(s_global, a_joint, r, t, s2_global)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        if self.count == 0:
            return None

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.stack([np.asarray(sample[0], dtype=np.float32) for sample in batch], axis=0)
        a_batch = np.stack([np.asarray(sample[1], dtype=np.float32) for sample in batch], axis=0)

        r_elems = [np.asarray(sample[2], dtype=np.float32) for sample in batch]
        t_elems = [np.asarray(sample[3], dtype=np.float32) for sample in batch]

        if r_elems[0].ndim == 0:
            r_batch = np.array(r_elems, dtype=np.float32).reshape(-1, 1)
        else:
            r_batch = np.stack(r_elems, axis=0)

        if t_elems[0].ndim == 0:
            t_batch = np.array(t_elems, dtype=np.float32).reshape(-1, 1)
        else:
            t_batch = np.stack(t_elems, axis=0)

        s2_batch = np.stack([np.asarray(sample[4], dtype=np.float32) for sample in batch], axis=0)

        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0