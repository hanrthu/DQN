"""Suggested Preprocessors."""

import numpy as np
from PIL import Image

from deeprl_hw2 import utils
from deeprl_hw2.core import Preprocessor
import gym
from gym import spaces
import cv2
from collections import deque


class HistoryPreprocessor(Preprocessor):
    """Keeps the last k states.

    Useful for domains where you need velocities, but the state
    contains only positions.

    When the environment starts, this will just fill the initial
    sequence values with zeros k times.

    Parameters
    ----------
    history_length: int
      Number of previous states to prepend to state being processed.

    """

    def __init__(self, history_length=1):
        self.k = history_length
        self.frames = deque([], maxlen=history_length)

    def process_state_for_network(self, state):
        """You only want history when you're deciding the current action to take."""
        frame = state
        return self.process_state_for_memory(frame)
    
    def process_state_for_memory(self, state):
        frame = state
        self.frames.append(frame)
        return self._get_ob()

    def reset(self, frame):
        """Reset the history sequence.

        Useful when you start a new episode.
        """
        for _ in range(self.k):
            self.frames.append(frame)
        return self._get_ob()

    def get_config(self):
        return {'history_length': self.history_length}
    
    def _get_ob(self):
        assert len(self.frames) == self.k
        frame = np.concatenate(self.frames, axis=-1)
        return frame


class AtariPreprocessor(Preprocessor):
    """Converts images to greyscale and downscales.

    Based on the preprocessing step described in:

    @article{mnih15_human_level_contr_throug_deep_reinf_learn,
    author =	 {Volodymyr Mnih and Koray Kavukcuoglu and David
                  Silver and Andrei A. Rusu and Joel Veness and Marc
                  G. Bellemare and Alex Graves and Martin Riedmiller
                  and Andreas K. Fidjeland and Georg Ostrovski and
                  Stig Petersen and Charles Beattie and Amir Sadik and
                  Ioannis Antonoglou and Helen King and Dharshan
                  Kumaran and Daan Wierstra and Shane Legg and Demis
                  Hassabis},
    title =	 {Human-Level Control Through Deep Reinforcement
                  Learning},
    journal =	 {Nature},
    volume =	 518,
    number =	 7540,
    pages =	 {529-533},
    year =	 2015,
    doi =        {10.1038/nature14236},
    url =	 {http://dx.doi.org/10.1038/nature14236},
    }

    You may also want to max over frames to remove flickering. Some
    games require this (based on animations and the limited sprite
    drawing capabilities of the original Atari).

    Parameters
    ----------
    new_size: 2 element tuple
      The size that each image in the state should be scaled to. e.g
      (84, 84) will make each image in the output have shape (84, 84).
    """

    def __init__(self, new_size):
        self.new_size = new_size

    def process_state_for_memory(self, state):
        """Scale, convert to greyscale and store as uint8.

        We don't want to save floating point numbers in the replay
        memory. We get the same resolution as uint8, but use a quarter
        to an eigth of the bytes (depending on float32 or float64)

        We recommend using the Python Image Library (PIL) to do the
        image conversions.
        """
        frame = state
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # change to one channel
        frame = cv2.resize(
            frame, (self.new_size[0], self.new_size[1]), interpolation=cv2.INTER_AREA
        )  # resize to 84 * 84
        frame = np.expand_dims(frame, -1)
        return frame


    def process_state_for_network(self, state):
        """Scale, convert to greyscale and store as float32.

        Basically same as process state for memory, but this time
        outputs float32 images.
        """
        return self.process_state_for_memory(state).astype(np.float32)
    
    def reset(self, frame):
        return self.process_state_for_memory(frame)

    def process_batch(self, samples):
        """The batches from replay memory will be uint8, convert to float32.

        Same as process_state_for_network but works on a batch of
        samples from the replay memory. Meaning you need to convert
        both state and next state values.
        """
        samples['state'] = samples['state'].astype(np.float32)
        samples['next_state'] = samples['next_state'].astype(np.float32)
        return samples  # convert dtype

    def process_reward(self, reward):
        """Clip reward between -1 and 1."""
        return np.sign(reward)


class PreprocessorSequence(Preprocessor):
    """You may find it useful to stack multiple prepcrocesosrs (such as the History and the AtariPreprocessor).

    You can easily do this by just having a class that calls each preprocessor in succession.

    For example, if you call the process_state_for_network and you
    have a sequence of AtariPreproccessor followed by
    HistoryPreprocessor. This this class could implement a
    process_state_for_network that does something like the following:

    state = atari.process_state_for_network(state)
    return history.process_state_for_network(state)
    """
    def __init__(self, preprocessors):
        self.preprocessors = preprocessors
    
    def process_state_for_network(self, state):
        """Scale, convert to greyscale and store as float32.

        Basically same as process state for memory, but this time
        outputs float32 images.
        """
        frame = state
        for p in self.preprocessors:
            frame = p.process_state_for_network(frame)
        return frame
    
    def process_state_for_memory(self, state):
        frame = state
        for p in self.preprocessors:
            frame = p.process_state_for_memory(frame)
        return frame
    
    def reset(self, frame):
        for p in self.preprocessors:
            frame = p.reset(frame)
        return frame

    def process_reward(self, reward):
        for p in self.preprocessors:
            reward = p.process_reward(reward)
        return reward