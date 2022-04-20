import collections
import numpy as np
import os
"""Core classes."""



Sample = collections.namedtuple('Sample', ['s', 'a', 'r', 'ns', 'ter'])


class Preprocessor:
    """Preprocessor base class.

    This is a suggested interface for the preprocessing steps. You may
    implement any of these functions. Feel free to add or change the
    interface to suit your needs.

    Preprocessor can be used to perform some fixed operations on the
    raw state from an environment. For example, in ConvNet based
    networks which use image as the raw state, it is often useful to
    convert the image to greyscale or downsample the image.

    Preprocessors are implemented as class so that they can have
    internal state. This can be useful for things like the
    AtariPreproccessor which maxes over k frames.

    If you're using internal states, such as for keeping a sequence of
    inputs like in Atari, you should probably call reset when a new
    episode begins so that state doesn't leak in from episode to
    episode.
    """

    def process_state_for_network(self, state):
        """Preprocess the given state before giving it to the network.

        Should be called just before the action is selected.

        This is a different method from the process_state_for_memory
        because the replay memory may require a different storage
        format to reduce memory usage. For example, storing images as
        uint8 in memory is a lot more efficient thant float32, but the
        networks work better with floating point images.

        Parameters
        ----------
        state: np.ndarray
          Generally a numpy array. A single state from an environment.

        Returns
        -------
        processed_state: np.ndarray
          Generally a numpy array. The state after processing. Can be
          modified in anyway.

        """
        return state

    def process_state_for_memory(self, state):
        """Preprocess the given state before giving it to the replay memory.

        Should be called just before appending this to the replay memory.

        This is a different method from the process_state_for_network
        because the replay memory may require a different storage
        format to reduce memory usage. For example, storing images as
        uint8 in memory and the network expecting images in floating
        point.

        Parameters
        ----------
        state: np.ndarray
          A single state from an environmnet. Generally a numpy array.

        Returns
        -------
        processed_state: np.ndarray
          Generally a numpy array. The state after processing. Can be
          modified in any manner.

        """
        return state

    def process_batch(self, samples):
        """Process batch of samples.

        If your replay memory storage format is different than your
        network input, you may want to apply this function to your
        sampled batch before running it through your update function.

        Parameters
        ----------
        samples: list(tensorflow_rl.core.Sample)
          List of samples to process

        Returns
        -------
        processed_samples: list(tensorflow_rl.core.Sample)
          Samples after processing. Can be modified in anyways, but
          the list length will generally stay the same.
        """
        return samples

    def process_reward(self, reward):
        """Process the reward.

        Useful for things like reward clipping. The Atari environments
        from DQN paper do this. Instead of taking real score, they
        take the sign of the delta of the score.

        Parameters
        ----------
        reward: float
          Reward to process

        Returns
        -------
        processed_reward: float
          The processed reward
        """
        return reward

    def reset(self):
        """Reset any internal state.

        Will be called at the start of every new episode. Makes it
        possible to do history snapshots.
        """
        pass


class Memory:
    """Interface for replay memories.

    We have found this to be a useful interface for the replay
    memory. Feel free to add, modify or delete methods/attributes to
    this class.

    It is expected that the replay memory has implemented the
    __iter__, __getitem__, and __len__ methods.

    If you are storing raw Sample objects in your memory, then you may
    not need the end_episode method, and you may want to tweak the
    append method. This will make the sample method easy to implement
    (just ranomly draw saamples saved in your memory).

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
      terminal state (i.e. the env returned is_terminal=True), of it
      is is an artificial terminal state (i.e. agent quit the episode
      early, but agent could have kept running episode).
    sample(batch_size, indexes=None)
      Return list of samples from the memory. Each class will
      implement a different method of choosing the
      samples. Optionally, specify the sample indexes manually.
    clear()
      Reset the memory. Deletes all references to the samples.
    """
    def __init__(self):
        """Setup memory.

        You should specify the maximum size o the memory. Once the
        memory fills up oldest values should be removed. You can try
        the collections.deque class as the underlying storage, but
        your sample method will be very slow.

        We recommend using a list as a ring buffer. Just track the
        index where the next sample should be inserted in the list.
        """
        pass

    def append(self, sample):
        raise NotImplementedError('This method should be overridden')

    def end_episode(self, final_state, is_terminal):
        raise NotImplementedError('This method should be overridden')

    def sample(self, batch_size, indexes=None):
        raise NotImplementedError('This method should be overridden')

    def clear(self):
        raise NotImplementedError('This method should be overridden')


class NoMemory(Memory):
    def __init__(self):
        super(NoMemory, self).__init__()
        self.buffer = []

    def append(self, sample):
        self.buffer.append(sample)

    def end_episode(self, final_state, is_terminal):
        pass

    def sample(self, batch_size, indexes=None):
        assert len(self.buffer) == batch_size
        buffer = self.buffer
        self.buffer = []
        return buffer

    def clear(self):
        assert len(self.buffer) == 0


class ReplayMemory(Memory):
    def __init__(self, max_length):
        super(ReplayMemory, self).__init__()
        self.buffer = []
        self.buffer_len = 0
        self.max_length = max_length
        self.insert_id = 0

    def append(self, sample):
        if self.buffer_len != self.max_length:
            self.buffer.append(sample)
            self.buffer_len += 1
        else:
            self.buffer[self.insert_id] = sample
            self.insert_id = (self.insert_id + 1) % self.max_length

    def end_episode(self, final_state, is_terminal):
        pass

    @staticmethod
    def encode(sample):
        return Sample(s=sample.s.astype(np.int8), ns=sample.ns.astype(np.int8), r=sample.r, ter=sample.ter, a=sample.a)

    @staticmethod
    def decode(sample):
        return Sample(s=sample.s.astype(np.float32), ns=sample.ns.astype(np.float32), r=sample.r, ter=sample.ter, a=sample.a)

    def sample(self, batch_size, indexes=None):
        idxs = np.random.randint(0, self.buffer_len, size=batch_size)
        ret = [self.buffer[idx] for idx in idxs]
        return ret

    def clear(self):
        self.buffer = []
        self.buffer_len = 0
        self.insert_id = 0
