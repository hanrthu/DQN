"""RL Policy classes.

We have provided you with a base policy class, some example
implementations and some unimplemented classes that should be useful
in your code.
"""
import torch
import numpy as np
import attr


class Policy:
    """Base class representing an MDP policy.

    Policies are used by the agent to choose actions.

    Policies are designed to be stacked to get interesting behaviors
    of choices. For instances in a discrete action space the lowest
    level policy may take in Q-Values and select the action index
    corresponding to the largest value. If this policy is wrapped in
    an epsilon greedy policy then with some probability epsilon, a
    random action will be chosen.
    """

    def select_action(self, q_values, is_training, **kwargs):
        """Used by agents to select actions.

        Returns
        -------
        Any:
          An object representing the chosen action. Type depends on
          the hierarchy of policy instances.
        """
        raise NotImplementedError('This method should be overriden.')


class UniformRandomPolicy(Policy):
    """Chooses a discrete action with uniform random probability.

    This is provided as a reference on how to use the policy class.

    Parameters
    ----------
    num_actions: int
      Number of actions to choose from. Must be > 0.

    Raises
    ------
    ValueError:
      If num_actions <= 0
    """

    def __init__(self, num_actions):
        assert num_actions >= 1
        self.num_actions = num_actions

    def select_action(self, **kwargs):
        """Return a random action index.

        This policy cannot contain others (as they would just be ignored).

        Returns
        -------
        int:
          Action index in range [0, num_actions)
        """
        return torch.randint(0, self.num_actions, size=[1])

    def get_config(self):  # noqa: D102
        return {'num_actions': self.num_actions}


class GreedyPolicy(Policy):
    """Always returns best action according to Q-values.

    This is a pure exploitation policy.
    """

    def select_action(self, q_values, **kwargs):  # noqa: D102
        return torch.argmax(q_values)


class GreedyEpsilonPolicy(Policy):
    """Selects greedy action or with some probability a random action.

    Standard greedy-epsilon implementation. With probability epsilon
    choose a random action. Otherwise choose the greedy action.

    Parameters
    ----------
    epsilon: float
     Initial probability of choosing a random action. Can be changed
     over time.
    """
    def __init__(self, epsilon, num_actions=None, g=None, u=None):
        if u is not None:
            self.uniform = u
        else:
            self.uniform = UniformRandomPolicy(num_actions)
        if g is not None:
            self.greedy = g
        else:
            self.greedy = GreedyPolicy()
        self.epsilon = epsilon

    def select_action(self, q_values, **kwargs):
        """Run Greedy-Epsilon for the given Q-values.

        Parameters
        ----------
        q_values: array-like
          Array-like structure of floats representing the Q-values for
          each action.

        Returns
        -------
        int:
          The action index chosen.
        """
        if np.random.rand() < self.epsilon:
            return self.uniform.select_action()
        else:
            return self.greedy.select_action(q_values)



class LinearDecayGreedyEpsilonPolicy(Policy):
    """Policy with a parameter that decays linearly.

    Like GreedyEpsilonPolicy but the epsilon decays from a start value
    to an end value over k steps.

    Parameters
    ----------
    start_value: int, float
      The initial value of the parameter
    end_value: int, float
      The value of the policy at the end of the decay.
    num_steps: int
      The number of steps over which to decay the value.

    """

    def __init__(self, start_value, end_value, num_steps, num_actions):  # noqa: D102
        self.start = start_value
        self.end = end_value
        self.num_steps = num_steps
        self.epsilon = start_value
        self.step = np.power(end_value / start_value, 1.0 / num_steps)
        print(self.step)
        self.uniform = UniformRandomPolicy(num_actions)
        self.greedy = GreedyPolicy()

    def select_action(self, q_values, is_training, **kwargs):
        """Decay parameter and select action.

        Parameters
        ----------
        q_values: np.array
          The Q-values for each action.
        is_training: bool, optional
          If true then parameter will be decayed. Defaults to true.

        Returns
        -------
        Any:
          Selected action.
        """
        if is_training:
            p = GreedyEpsilonPolicy(epsilon=self.epsilon, g=self.greedy, u=self.uniform)
            self.epsilon *= self.step
            return p.select_action(q_values)
        else:
            p = GreedyEpsilonPolicy(epsilon=self.end, g=self.greedy, u=self.uniform)
            return p.select_action(q_values)


    def reset(self):
        self.epsilon = self.start
