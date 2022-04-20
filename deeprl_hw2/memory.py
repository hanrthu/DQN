import torchvision.transforms as transforms

from deeprl_hw2.core import Memory
from deeprl_hw2.core import Sample
import numpy as np


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
    def __init__(self, max_size):
        """Setup memory.

        You should specify the maximum size of the memory. Once the
        memory fills up oldest values should be removed. You can try
        the collections.deque class as the underlying storage, but
        your sample method will be very slow.

        We recommend using a list as a ring buffer. Just track the
        index where the next sample should be inserted in the list.
        """
        super().__init__(max_size)
        self.max_size = max_size
        self.memory = []
        self.indices = 0
        self.replace_idx = 0
        # self.device = device

    def append(self, state, action, reward, state_prime, terminate):
        # sample = ReplaySample(state, action, reward, state_prime, terminate)
        to_tensor = transforms.ToTensor()
        sample = [to_tensor(state), action, reward, to_tensor(state_prime), terminate]
        if self.indices < self.max_size:
            self.memory.append(sample)
            self.indices += 1
        else:
            self.memory[self.replace_idx] = sample
            self.replace_idx = (self.replace_idx + 1) % self.max_size

    def sample(self, batch_size, indexes=None):
        idx = np.random.choice(self.indices, batch_size)
        # print(self.indices, idx)
        # print(len(self.memory))
        states = [self.memory[i][0] for i in idx]
        actions = [self.memory[i][1] for i in idx]
        rewards = [self.memory[i][2] for i in idx]
        next_states = [self.memory[i][3] for i in idx]
        terminates = [self.memory[i][4] for i in idx]
        # print("Memory:", np.concatenate(states).shape)
        return states, actions, rewards, next_states, terminates

    def clear(self):
        self.memory = []
        self.indices = 0
        self.replace_idx = 0
