"""Main DQN agent."""
import os

import gym
import torch
import torch.nn as nn
import numpy as np
import wandb
from deeprl_hw2.core import Preprocessor, Memory, Policy
from deeprl_hw2.policy import UniformRandomPolicy, GreedyEpsilonPolicy, LinearDecayGreedyEpsilonPolicy, GreedyPolicy
import torchvision.transforms as transforms
from deeprl_hw2.utils import get_hard_target_model_updates
from tqdm import tqdm

class DeepQNet(nn.Module):
    def __init__(self, input_channels, num_actions, large=False):
        super(DeepQNet, self).__init__()
        if large:
            self.backbone = nn.Sequential(
                nn.Conv2d(kernel_size=8, in_channels=input_channels, out_channels=32, stride=4),
                nn.ReLU(),
                nn.Conv2d(kernel_size=4, in_channels=32, out_channels=64, stride=2),
                nn.ReLU(),
                nn.Conv2d(kernel_size=3, in_channels=64, out_channels=64, stride=1),
                nn.ReLU(),
            )
            self.linear = nn.Sequential(
                nn.Linear(in_features=64 * 7 * 7, out_features=512),
                nn.ReLU(),
                nn.Linear(in_features=512, out_features=num_actions)
            )
        else:
            self.backbone = nn.Sequential(
                nn.Conv2d(kernel_size=8, in_channels=input_channels, out_channels=32, stride=4),
                nn.ReLU(),
                nn.Conv2d(kernel_size=4, in_channels=32, out_channels=64, stride=2),
                nn.ReLU(),
            )
            self.linear = nn.Sequential(
                nn.Linear(in_features=64 * 9 * 9, out_features=256),
                nn.ReLU(),
                nn.Linear(in_features=256, out_features=num_actions)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        # print(x.shape)
        x = x.contiguous()
        q_value = self.linear(x.view(x.shape[0], -1))
        return q_value

class LinearQNet(nn.Module):
    def __init__(self, input_channels, num_actions):
        super(LinearQNet, self).__init__()
        self.linear = nn.Linear(input_channels * 84 * 84, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print("X_shape:", x.shape)
        x = x.contiguous()
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
        self.criterion = torch.nn.SmoothL1Loss()
        self.args = args
        self.logger = logger
        self.eval_freq = eval_freq
        self.policy = None

    def calc_q_values(self, state):
        """Given a state (or batch of states) calculate the Q-values.

        Basically run your network on these states.

        Return
        ------
        Q-values for the state(s)
        """
        # state = state.astype(np.float32) / 255
        state = state.astype(np.uint8)
        to_tensor = transforms.ToTensor()
        state = to_tensor(state).to(self.device)
        if len(state.shape) == 3:
            state = torch.unsqueeze(state, dim=0)
        with torch.no_grad():
            q_value = self.q_network(state).detach().cpu().numpy()
        return q_value

    def select_policy(self, policy_name, num_actions):
        if policy_name == 'Uniform':
            self.policy = UniformRandomPolicy(num_actions)
            self.logger.info("Uniform Random Policy Selected...")
        elif policy_name == 'GreedyEps':
            self.policy = GreedyEpsilonPolicy(0.05, num_actions)
            self.logger.info("Greedy Epsilon Policy Selected...")
        else:
            self.logger.info("Start Training Q Network!")
            greedy = GreedyPolicy()
            self.policy = LinearDecayGreedyEpsilonPolicy(greedy, 'lineardecay', self.start_eps, self.end_eps,
                                                         self.final_exploration_frame, num_actions)
            self.logger.info("Linear Decay Greedy Epsilon Policy Selected...")

    def select_action(self, state, iteration, is_training, **kwargs):
        """Select the action based on the current state.

        You will probably want to vary your behavior here based on
        which stage of training your in. For example, if you're still
        collecting random samples you might want to use a
        UniformRandomPolicy.

        If you're testing, you might want to use a GreedyEpsilonPolicy
        with a low epsilon.

        If you're training, you might want to use the
        LinearDecayGreedyEpsilonPolicy.

        This would also be a good place to call
        process_state_for_network in your preprocessor.

        Returns
        --------
        selected action
        """
        if is_training and iteration < self.num_burn_in:
            action = self.policy.select_action()
        else:
            self.logger.debug("Shape of batched state: {}".format(state.shape))
            q_values = self.calc_q_values(state)
            action = self.policy.select_action(q_values, is_training)
        return action

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
        self.logger.debug("Reset Environment Shape: {}".format(env.reset().shape))
        state = self.preprocessor.reset(env.reset())
        self.logger.debug("State Shape After Preprocess: {}".format(state.shape))
        self.select_policy('Uniform', action_num)
        switched = 0
        losses, rewards = [], []
        epi_losses, epi_rewards = [], []
        # Need to add some loggings
        for iteration in tqdm(range(num_iterations)):
            if switched == 0 and iteration >= self.num_burn_in:
                switched = 1
                self.select_policy('LinearDecayGreedyEps', action_num)
            action = self.select_action(state, iteration, True)
            obs, reward, terminate, _ = env.step(action)
            # if reward != 0:
            #     print("Iter:{}, Reward:{}".format(iteration, reward))
            rewards.append(float(reward))
            reward = self.preprocessor.process_reward(reward)
            obs_m = self.preprocessor.process_state_for_memory(obs)
            # obs_n = self.preprocessor.process_state_for_network(obs)
            self.memory.append(state, action, reward, obs_m, terminate)
            if not terminate:
                state = obs_m
            else:
                state = self.preprocessor.reset(env.reset())
                if iteration >= self.num_burn_in:
                    epi_losses.append(np.mean(losses))
                    epi_rewards.append(np.sum(rewards))
                    # Need to add some loggings
                    wandb.log({
                        'Episode Loss': np.mean(losses),
                        'Episode Reward': np.sum(rewards)
                    })
                losses = []
                rewards = []
            if iteration >= self.num_burn_in:
                samples = self.memory.sample(self.batch_size)
                x_in, q_target, action_list = self.process_batch(samples)
                # print("X_in:",x_in)
                q_pred = self.q_network(x_in)
                self.logger.debug("Action List:{}, Pred Shape:{}".format(action_list, q_pred.shape))
                out = q_pred[range(self.batch_size), action_list]
                # print(q_pred, out, action_list)
                self.logger.debug("Target Shape:{}, Output Shape:{}".format(q_target.shape, out.shape))
                loss = self.criterion(out, q_target)
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
                        './exps/{}/{}-{}.pth'.format(self.args.expname, self.args.env, iteration)
                    )
                if iteration % self.eval_freq == 0:
                    self.logger.info("Evaluating Q Network at iteration {}...".format(iteration))
                    reward_list, _ = self.evaluate(env, 20)
                    wandb.log({
                        "Eval Reward(20 Episodes Mean)": np.mean(reward_list)
                    })
        return "Successfully Fit the model!"

    def process_batch(self, samples):
        curr_state_list = []
        next_state_list = []
        action_list = []
        reward_list = []
        terminate_list = []
        for sample in samples:
            curr_state, action, reward, next_state, terminate = sample
            curr_state_list.append(curr_state)
            next_state_list.append(next_state)
            action_list.append(action)
            reward_list.append(reward)
            terminate_list.append(terminate)
        r_list = torch.Tensor(reward_list).to(self.device)
        t_list = torch.BoolTensor(terminate_list).to(self.device)
        # print("State Shape:", len(curr_state_list), curr_state_list[0].shape)
        s_in = torch.from_numpy(np.stack(curr_state_list))
        ns_in = torch.from_numpy(np.stack(next_state_list))
        s_in = (torch.permute(s_in, (0, 3, 1, 2)) / 255).to(self.device)
        ns_in = (torch.permute(ns_in, (0, 3, 1, 2)) / 255).to(self.device)
        with torch.no_grad():
            qn = self.qminus_network(ns_in)
        q_discounted = r_list + self.gamma * torch.max(qn, dim=-1)[0]
        q_target = torch.where(t_list, r_list, q_discounted)
        # print(s_in.shape,q_target.shape)

        # ns_in = torch.zeros([32, 4, 84, 84]).to(self.device)
        # with torch.no_grad():
        #     qn = self.qminus_network(ns_in)
        # s_in = torch.zeros([32, 4, 84, 84]).to(self.device)
        # q_target = torch.zeros([32]).to(self.device)
        # action_list = [0 for i in range(32)]
        return s_in, q_target, action_list

    def evaluate_episode(self, env, max_episode_length=None):
        action_num = env.action_space.n
        state = self.preprocessor.reset(env.reset())
        # self.select_policy('GreedyEps', action_num)
        policy = GreedyEpsilonPolicy(0.05, action_num)
        terminate = False
        total_reward = 0
        video_frames = [env.reset()]
        while not terminate:
            # action = self.select_action(state, 0, False)
            q_values = self.calc_q_values(state)
            action = policy.select_action(q_values)
            obs, reward, terminate, _ = env.step(action)
            total_reward += reward
            video_frames.append(obs)
            state = self.preprocessor.process_state_for_network(obs)
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
        for i in tqdm(range(num_episodes)):
            reward_epi, video_epi = self.evaluate_episode(env)
            reward_list.append(reward_epi)
            video_list.append(video_epi)
        # print("Mean Reward:", np.mean(reward_list))
        self.q_network.train()
        return reward_list, video_list
