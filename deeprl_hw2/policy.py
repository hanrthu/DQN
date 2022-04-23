"""RL Policy classes.

We have provided you with a base policy class, some example
implementations and some unimplemented classes that should be useful
in your code.
"""
from collections.abc import Callable

import numpy as np
import torch
import wandb
from deeprl_hw2.core import Policy

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

    def select_action(self, *args, **kwargs):
        """Return a random action index.

        This policy cannot contain others (as they would just be ignored).

        Returns
        -------
        int:
          Action index in range [0, num_actions)
        """
        return np.random.randint(0, self.num_actions)

    def get_config(self):  # noqa: D102
        return {'num_actions': self.num_actions}


class GreedyPolicy(Policy):
    """Always returns best action according to Q-values.

    This is a pure exploitation policy.
    """

    def select_action(
        self,
        state: torch.FloatTensor,
        q_func: Callable[[torch.FloatTensor], torch.FloatTensor],
        *args,
        **kwargs
    ):  # noqa: D102
        q_values = q_func(state)
        # debug
        # q_values_numpy = q_values.cpu().numpy().T
        return q_values.argmax().item()


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
    def __init__(self, epsilon, num_actions):
        assert num_actions >= 1
        self.eps = epsilon
        self.num_actions = num_actions

    def select_action(
        self,
        state: torch.FloatTensor,
        q_func: Callable[[torch.FloatTensor], torch.FloatTensor],
        *args,
        **kwargs
    ):
        """Run Greedy-Epsilon for the given Q-values.

        Parameters
        ----------
        state:
            当前状态
        q_func:
            Q value 函数
        Returns
        -------
        int:
          The action index chosen.
        """
        p = np.random.rand()
        if p > self.eps:
            q_values = q_func(state)
            # return np.argmax(q_values)
            return q_values.argmax().item()
        else:
            return np.random.randint(0, self.num_actions)


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

    def __init__(self, greedy_policy: GreedyPolicy, attr_name, start_value, end_value,
                 num_steps, num_actions):  # noqa: D102
        self.greedy_policy = greedy_policy
        self.attr_name = attr_name
        self.start_value = start_value
        self.end_value = end_value
        self.num_steps = num_steps
        self.num_actions = num_actions
        self.current_step = 0

    def select_action(
        self,
        state: torch.FloatTensor,
        q_func: Callable[[torch.FloatTensor], torch.FloatTensor],
        is_training: bool,
        *args,
        **kwargs
    ):
        """Decay parameter and select action.

        Parameters
        ----------
        state:
            当前状态
        q_func:
            Q value 函数
        is_training: bool, optional
          If true then parameter will be decayed. Defaults to true.

        Returns
        -------
        Any:
          Selected action.
        """
        p = np.random.rand()
        if is_training:
            if self.current_step > self.num_steps:
                eps = self.end_value
            else:
                eps = self.start_value + (self.end_value - self.start_value) * self.current_step / self.num_steps
            self.current_step += 1
            # print("Current_step:{}. Current_eps:{}.".format(self.current_step, eps))
        else:
            eps = self.end_value 
        wandb.log({
            'Epsilon': eps
        })
        if p > eps:
            return self.greedy_policy.select_action(state, q_func)
        else:
            return np.random.randint(0, self.num_actions)

    def reset(self):
        """Start the decay over at the start value."""
        self.current_step = 0
