import torchvision.transforms as transforms

from deeprl_hw2.core import Memory
from deeprl_hw2.core import Sample
import numpy as np
from torch.utils.data import DataLoader


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
    def __init__(self, max_size: int, img_size: int, history_len: int):
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
        self.replace_idx = 0
        self.size = 0
        self.state = np.empty((max_size, img_size, img_size, history_len), dtype=np.uint8)
        self.state_prime = np.empty((max_size, img_size, img_size, history_len), dtype=np.uint8)
        self.reward = np.empty(max_size)
        self.action = np.empty(max_size)
        self.terminate = np.empty(max_size)

    def append(self, state: np.ndarray, action: int, reward: int, state_prime: np.ndarray, terminate: int):
        # sample = ReplaySample(state, action, reward, state_prime, terminate)
        # sample = [state.astype(np.uint8), action, reward, state_prime.astype(np.uint8), terminate]
        idx = self.replace_idx
        self.state[idx] = state
        self.action[idx] = action
        self.reward[idx] = reward
        self.state_prime[idx] = state_prime
        self.replace_idx = (self.replace_idx + 1) % self.cap
        if self.size < self.cap:
            self.size += 1

    def sample(self, batch_size, indexes=None):
        idx = np.random.choice(range(self.size), batch_size)
        return self.state[idx], self.action[idx], self.reward[idx], self.state_prime[idx], self.terminate[idx]

    def clear(self):
        self.size = 0
        self.replace_idx = 0
