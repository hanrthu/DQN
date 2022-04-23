"""Main DQN agent."""
from collections import Counter
import os
from pathlib import Path
from typing import Type

import gym
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import wandb

from deeprl_hw2.core import Memory, Policy, Preprocessor
from deeprl_hw2.policy import GreedyEpsilonPolicy, GreedyPolicy, LinearDecayGreedyEpsilonPolicy, UniformRandomPolicy
from deeprl_hw2.utils import eval_model, get_hard_target_model_updates

class DeepQNet(nn.Module):
    def __init__(self, input_channels, num_actions, large=False, duel=False):
        super(DeepQNet, self).__init__()
        self.duel = duel
        if large:
            self.backbone = nn.Sequential(
                nn.Conv2d(kernel_size=8, in_channels=input_channels, out_channels=32, stride=4),
                nn.ReLU(),
                nn.Conv2d(kernel_size=4, in_channels=32, out_channels=64, stride=2),
                nn.ReLU(),
                nn.Conv2d(kernel_size=3, in_channels=64, out_channels=64, stride=1),
                nn.ReLU(),
            )
            if duel:
                self.linear = nn.Linear(in_features=64 * 7 * 7, out_features=512)
                self.v_linear = nn.Sequential(
                    nn.ReLU(),
                    nn.Linear(in_features=512, out_features=128),
                    nn.ReLU(),
                    nn.Linear(in_features=128, out_features=1)
                )
                self.a_linear = nn.Sequential(
                    nn.ReLU(),
                    nn.Linear(in_features=512, out_features=128),
                    nn.ReLU(),
                    nn.Linear(in_features=128, out_features=num_actions)
                )
            else:
                self.linear = nn.Sequential(
                    nn.Linear(in_features=64 * 7 * 7, out_features=512),
                    nn.ReLU(),
                    nn.Linear(in_features=512, out_features=num_actions)
                )
        else:
            self.backbone = nn.Sequential(
                nn.Conv2d(kernel_size=8, in_channels=input_channels, out_channels=16, stride=4),
                nn.ReLU(),
                nn.Conv2d(kernel_size=4, in_channels=16, out_channels=32, stride=2),
                nn.ReLU(),
            )
            if duel:
                self.linear = nn.Linear(in_features=32 * 9 * 9, out_features=256)
                self.v_linear = nn.Sequential(
                    nn.ReLU(),
                    nn.Linear(in_features=256, out_features=1),
                )
                self.a_linear = nn.Sequential(
                    nn.ReLU(),
                    nn.Linear(in_features=256, out_features=num_actions)
                )
            else:
                self.linear = nn.Sequential(
                    nn.Linear(in_features=32 * 9 * 9, out_features=256),
                    nn.ReLU(),
                    nn.Linear(in_features=256, out_features=num_actions)
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        # print(x.shape)
        q_value = self.linear(x.view(x.shape[0], -1))
        if self.duel:
            s_value = self.v_linear(q_value)
            a_value = self.a_linear(q_value)
            q_value = s_value + (a_value - torch.mean(a_value, dim=-1, keepdim=True))
        return q_value


class LinearQNet(nn.Module):
    def __init__(self, input_channels, num_actions):
        super(LinearQNet, self).__init__()
        self.linear = nn.Linear(input_channels * 84 * 84, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print("X_shape:", x.shape)
        x = x.view(x.shape[0], -1)
        # print("X_reshaped:", x.shape)
        return self.linear(x)

class DQNAgent:
    """Class implementing DQN.

    This is a basic outline of the functions/parameters you will need
    in order to implement the DQNAgnet. This is just to get you
    started. You may need to tweak the parameters, add new ones, etc.

    Feel free to change the functions and funciton parameters that the
    class provides.

    We have provided docstrings to go along with our suggested API.

    Parameters
    ----------
    q_network:
      Your Q-network model.
    preprocessor: deeprl_hw2.core.Preprocessor
      The preprocessor class. See the associated classes for more
      details.
    memory: deeprl_hw2.core.Memory
      Your replay memory.
    gamma: float
      Discount factor.
    target_update_freq: float
      Frequency to update the target network. You can either provide a
      number representing a soft target update (see utils.py) or a
      hard target update (see utils.py and Atari paper.)
    num_burn_in: int
      Before you begin updating the Q-network your replay memory has
      to be filled up with some number of samples. This number says
      how many.
    train_freq: int
      How often you actually update your Q-Network. Sometimes
      stability is improved if you collect a couple samples for your
      replay memory, for every Q-network update that you run.
    batch_size: int
      How many samples in each minibatch.
    """

    def __init__(self,
                 q_network: nn.Module,
                 qminus_network: nn.Module,
                 preprocessor: Preprocessor,
                 memory: Memory,
                 gamma: float,
                 start_eps: float,
                 end_eps: float,
                 target_update_freq: int,
                 num_burn_in: int,
                 train_freq: int,
                 eval_freq: int,
                 batch_size: int,
                 iterations: int,
                 final_exploration_frame: int,
                 optimizer,
                 scheduler,
                 device: str,
                 logger,
                 args):
        self.q_network = q_network
        self.qminus_network = qminus_network
        self.preprocessor = preprocessor
        self.memory = memory
        self.gamma = gamma
        self.start_eps = start_eps
        self.end_eps = end_eps
        self.t_update_freq = target_update_freq
        self.num_burn_in = num_burn_in
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.iterations = iterations
        self.final_exploration_frame = final_exploration_frame
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.criterion = torch.nn.MSELoss()
        self.args = args
        self.logger = logger
        self.eval_freq = eval_freq
        if self.args is not None:
            self.double_q = self.args.double_q
        # self.policy = None

    def calc_q_values(self, state: torch.FloatTensor) -> torch.FloatTensor:
        """Given a state (or batch of states) calculate the Q-values.

        Basically run your network on these states.

        Return
        ------
        Q-values for the state(s)
        """
        if len(state.shape) == 3:
            state = state[None]
        with eval_model(self.q_network):
            q_values = self.q_network(state)
        if len(state.shape) == 3:
            q_values = q_values[0]
        return q_values

    def select_policy(self, policy_cls: Type[Policy], num_actions: int) -> Policy:
        # if policy_name == 'Uniform':
        # if policy_cls is UniformRandomPolicy:
        if policy_cls is UniformRandomPolicy:
            self.logger.info("Uniform Random Policy Selected...")
            return UniformRandomPolicy(num_actions)
        elif policy_cls is GreedyEpsilonPolicy:
            self.logger.info("Greedy Epsilon Policy Selected...")
            return GreedyEpsilonPolicy(0.05, num_actions)
        else:
            assert policy_cls is LinearDecayGreedyEpsilonPolicy
            self.logger.info("Linear Decay Greedy Epsilon Policy Selected...")
            greedy = GreedyPolicy()
            return LinearDecayGreedyEpsilonPolicy(greedy, 'lineardecay', self.start_eps, self.end_eps,
                                                  self.final_exploration_frame, num_actions)

    # def select_action(self, state: torch.FloatTensor, iteration: int, is_training: bool):
    #     """Select the action based on the current state.
    #
    #     You will probably want to vary your behavior here based on
    #     which stage of training your in. For example, if you're still
    #     collecting random samples you might want to use a
    #     UniformRandomPolicy.
    #
    #     If you're testing, you might want to use a GreedyEpsilonPolicy
    #     with a low epsilon.
    #
    #     If you're training, you might want to use the
    #     LinearDecayGreedyEpsilonPolicy.
    #
    #     This would also be a good place to call
    #     process_state_for_network in your preprocessor.
    #
    #     Returns
    #     --------
    #     selected action
    #     """
    #     if is_training and iteration < self.num_burn_in:
    #         action = self.policy.select_action()
    #     else:
    #         q_values = self.calc_q_values(state)
    #         action = self.policy.select_action(q_values, is_training)
    #     return action

    def fit(self, env: gym.Env, num_iterations: int, max_episode_length=None):
        """Fit your model to the provided environment.

        It's a good idea to print out things like loss, average reward,
        Q-values, etc to see if your agent is actually improving.

        You should probably also periodically save your network
        weights and any other useful info.

        This is where you should sample actions from your network,
        collect experience samples and add them to your replay memory,
        and update your network parameters.

        Parameters
        ----------
        env: gym.Env
          This is your Atari environment. You should wrap the
          environment using the wrap_atari_env function in the
          utils.py
        num_iterations: int
          How many samples/updates to perform.
        max_episode_length: int
          How long a single episode should last before the agent
          resets. Can help exploration.
        """
        action_num = env.action_space.n
        self.q_network.train()
        # self.logger.debug("Reset Environment Shape: {}".format(env.reset().shape))
        state_m: torch.ByteTensor = self.preprocessor.reset(env.reset())
        lives: int = env.unwrapped.ale.lives()
        self.logger.debug("State Shape After Preprocess: {}".format(state_m.shape))
        policy = self.select_policy(UniformRandomPolicy, action_num)
        losses, rewards = [], []
        epi_losses, epi_rewards = [], []
        output_dir = Path(f'exps/{self.args.expname}')
        output_dir.mkdir(parents=True, exist_ok=True)
        # Need to add some loggings
        for iteration in tqdm(range(num_iterations), ncols=80):
            state_n: torch.FloatTensor = (state_m / 255).to(self.device)
            if iteration == self.num_burn_in:
                self.logger.info("Start Training Q Network!")
                policy = self.select_policy(LinearDecayGreedyEpsilonPolicy, action_num)
            action = policy.select_action(state_n, self.calc_q_values, is_training=True)
            obs, reward, terminate, info = env.step(action)
            # if reward != 0:
            #     print(iteration, reward)
            rewards.append(float(reward))
            reward = self.preprocessor.process_reward(reward)
            next_state: torch.ByteTensor = self.preprocessor.process_state_for_memory(obs)
            self.memory.append(
                state_m,
                action,
                reward + self.args.life_penalty * (lives - info['ale.lives']),
                next_state,
                terminate,
            )
            if not terminate:
                state_m = next_state
                lives = info['ale.lives']
            else:
                state_m = self.preprocessor.reset(env.reset())
                lives = env.unwrapped.ale.lives()
                if iteration >= self.num_burn_in:
                    epi_losses.append(np.mean(losses))
                    epi_rewards.append(np.sum(rewards))
                    # Need to add some loggings
                    wandb.log({
                        'Episode Loss': np.mean(losses),
                        'Episode Reward': np.sum(rewards)
                    })
                losses.clear()
                rewards.clear()

            if iteration >= self.num_burn_in:
                batch = self.memory.sample(self.batch_size)
                replay_batch = self.process_batch(batch)
                q_pred = self.q_network(replay_batch['state'])
                # self.logger.debug("Action List:{}, Pred Shape:{}".format(replay_batch['action'], q_pred.shape))
                out = q_pred[range(self.batch_size), replay_batch['action']]
                # print(q_pred, out, action_list)
                # self.logger.debug("Target Shape:{}, Output Shape:{}".format(q_target.shape, out.shape))
                loss = self.criterion(out, replay_batch['q_target'])
                loss.backward()
                losses.append(loss.item())
                if iteration % self.train_freq == 0:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                if iteration % self.t_update_freq == 0:
                    get_hard_target_model_updates(self.qminus_network, self.q_network)
                if iteration % (num_iterations // 3) == 0 or iteration % 200000 == 0:
                    os.makedirs('./exps/{}'.format(self.args.expname), exist_ok=True)
                    torch.save(
                        self.q_network.state_dict(),
                        output_dir / f'{self.args.env}-{iteration}.pth'
                    )
                if iteration % self.eval_freq == 0:
                    self.logger.info("Evaluating Q Network at iteration {}...".format(iteration))
                    reward_list, _ = self.evaluate(env, 20)
                    wandb.log({
                        "Eval Reward(20 Episodes Mean)": np.mean(reward_list)
                    })
        return "Successfully Fit the model!"

    def process_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        for k, v in batch.items():
            batch[k] = v.to(self.device)
        with eval_model(self.qminus_network):
            qn = self.qminus_network(batch['next_state'])
        if self.double_q:
            with eval_model(self.q_network):
                q = self.q_network(batch['next_state'])
            # print("Q Value:", q)
            # print("Q Index:", q.argmax(dim=-1))
            # print("Q_n::", qn.shape)
            q_target = batch['reward'] + self.gamma * qn[range(self.batch_size), q.argmax(dim=-1)]
        else:
            q_target = batch['reward'] + self.gamma * qn.max(dim=-1)[0]
        q_target[batch['terminate']] = batch['reward'][batch['terminate']]
        return {
            'state': batch['state'],
            'q_target': q_target,
            'action': batch['action'],
        }

    def evaluate_episode(self, env,policy, max_episode_length=None):
        video_frames = [env.reset()]
        state: torch.ByteTensor = self.preprocessor.reset(video_frames[0])
        terminate = False
        total_reward = 0
        iteration = 0
        while not terminate:
            state_n = (state / 255).to(self.device)
            action = policy.select_action(state_n, self.calc_q_values, is_training=False)
            obs, reward, terminate, _ = env.step(action)
            total_reward += reward
            video_frames.append(obs)
            state = self.preprocessor.process_state_for_memory(obs)
            iteration += 1
        return total_reward, video_frames

    def evaluate(self, env, num_episodes, max_episode_length=None):
        """Test your agent with a provided environment.
        
        You shouldn't update your network parameters here. Also if you
        have any layers that vary in behavior between train/test time
        (such as dropout or batch norm), you should set them to test.

        Basically run your policy on the environment and collect stats
        like cumulative reward, average episode length, etc.

        You can also call the render function here if you want to
        visually inspect your policy.
        """
        self.q_network.eval()
        reward_list = []
        video_list = []
        action_num = env.action_space.n
        policy = self.select_policy(GreedyEpsilonPolicy, action_num)
        for i in tqdm(range(num_episodes), ncols=80):
            reward_epi, video_epi = self.evaluate_episode(env, policy)
            reward_list.append(reward_epi)
            video_list.append(video_epi)
        # print("Mean Reward:", np.mean(reward_list))
        self.q_network.train()
        return reward_list, video_list
