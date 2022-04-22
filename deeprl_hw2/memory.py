from __future__ import annotations

from collections import Counter
import random

import numpy as np
import torch

from deeprl_hw2.core import Memory, Sample

class ReplaySample(Sample):
    def __init__(self, state, action, reward, state_prime, terminate):
        super(ReplaySample, self).__init__()
        self.content = [state, action, reward, state_prime, terminate]

    def get_content(self):
        return self.content


class ReplayMemory(Memory):
    """Interface for replay memories.

    We have found this to be a useful interface for the replay
    memory. Feel free to add, modify or delete methods/attributes to
    this class.

    It is expected that the replay memory has implemented the
    __iter__, __getitem__, and __len__ methods.

    If you are storing raw Sample objects in your memory, then you may
    not need the end_episode method, and you may want to tweak the
    append method. This will make the sample method easy to implement
    (just randomly draw samples saved in your memory).

    However, the above approach will waste a lot of memory (as states
    will be stored multiple times in s as next state and then s' as
    state, etc.). Depending on your machine resources you may want to
    implement a version that stores samples in a more memory efficient
    manner.

    Methods
    -------
    append(state, action, reward, debug_info=None)
      Add a sample to the replay memory. The sample can be any python
      object, but it is suggested that tensorflow_rl.core.Sample be
      used.
    end_episode(final_state, is_terminal, debug_info=None)
      Set the final state of an episode and mark whether it was a true
      terminal state (i.e. the env returned is_terminal=True), or it
      is an artificial terminal state (i.e. agent quit the episode
      early, but agent could have kept running episode).
    sample(batch_size, indexes=None)
      Return list of samples from the memory. Each class will
      implement a different method of choosing the
      samples. Optionally, specify the sample indexes manually.
    clear()
      Reset the memory. Deletes all references to the samples.
    """
    def __init__(self, max_size: int, img_size: int, history_len: int, device: torch.device | str, warmup_size: int = 50000):
        """Setup memory.

        You should specify the maximum size of the memory. Once the
        memory fills up oldest values should be removed. You can try
        the collections.deque class as the underlying storage, but
        your sample method will be very slow.

        We recommend using a list as a ring buffer. Just track the
        index where the next sample should be inserted in the list.
        """
        super().__init__(max_size)
        self.cap = max_size
        self.replace_idx = self.size = 0
        self.state = torch.zeros(max_size, history_len, img_size, img_size, dtype=torch.uint8)
        self.next_state = torch.zeros_like(self.state)
        self.reward = torch.zeros(max_size, dtype=torch.float)
        self.action = torch.zeros(max_size, dtype=torch.long)
        self.terminate = torch.zeros(max_size, dtype=torch.bool)
        self.device = device
        self.reward_counter = Counter()
        self.warmup_size = warmup_size

    def append(
        self,
        state: torch.ByteTensor,
        action: int,
        reward: int,
        next_state: torch.ByteTensor,
        terminate: int
    ):
        if self.size >= self.warmup_size:
            reward_weights = np.sqrt(list(self.reward_counter.values()))
            if np.random.choice(list(self.reward_counter), p=reward_weights / reward_weights.sum()) != reward:
                return

        idx = self.replace_idx
        if self.size == self.cap:
            self.reward_counter[self.reward[idx]] -= 1
        self.state[idx] = state
        self.action[idx] = action
        self.reward[idx] = reward
        self.reward_counter[reward] += 1
        self.next_state[idx] = next_state
        if self.size < self.cap:
            self.size += 1

        self.replace_idx = (self.replace_idx + 1) % self.cap
        if self.replace_idx % self.warmup_size == 0:
            print(self.reward_counter)

    def sample(self, batch_size, indexes=None) -> dict[str, torch.Tensor]:
        idx = np.random.randint(self.size, size=batch_size)
        # 状态使用 uint8 存储的唯一意义是节省空间，所以当从 memory 取出时，就除以 255
        ret = {
            'state': self.state[idx] / 255,
            'action': self.action[idx],
            'reward': self.reward[idx],
            'next_state': self.next_state[idx] / 255,
            'terminate': self.terminate[idx],
        }
        for k, v in ret.items():
            ret[k] = v.to(self.device)
        return ret

    def clear(self):
        self.size = self.replace_idx = 0
        self.reward_counter.clear()
