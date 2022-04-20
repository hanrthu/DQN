"""Main DQN agent."""
import random
import torch
import numpy as np
import deeprl_hw2.utils as utils
from deeprl_hw2.core import Sample
from tqdm import trange
import os
import json

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
    q_network: keras.models.Model
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
                 source_network,
                 target_network,
                 preprocessor,
                 memory,
                 # policy,
                 optimizer,
                 scheduler,
                 args):
        self.source_network = source_network
        self.preprocessor = preprocessor
        self.memory = memory
        self.target_network = target_network
        # self.policy = policy
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = torch.nn.SmoothL1Loss()
        '''
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.num_burn_in = num_burn_in
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.device = device'''
        self.args = args
        utils.get_hard_target_model_updates(self.target_network, self.source_network)

    def compile(self, optimizer, loss_func):
        """Setup all of the TF graph variables/ops.

        This is inspired by the compile method on the
        keras.models.Model class.

        This is a good place to create the target network, setup your
        loss function and any placeholders you might need.
        
        You should use the mean_huber_loss function as your
        loss_function. You can also experiment with MSE and other
        losses.

        The optimizer can be whatever class you want. We used the
        keras.optimizers.Optimizer class. Specifically the Adam
        optimizer.
        """
        pass


    def update_policy(self):
        """Update your policy.

        Behavior may differ based on what stage of training your
        in. If you're in training mode then you should check if you
        should update your network parameters based on the current
        step and the value you set for train_freq.

        Inside, you'll want to sample a minibatch, calculate the
        target values, update your network, and then update your
        target values.

        You might want to return the loss and other metrics as an
        output. They can help you monitor how training is going.
        """
        pass

    def select_action(self, state, is_training, **kwargs):
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
        state = np.float32(state)
        state = torch.from_numpy(state.reshape(-1)).to(self.args.device)
        with torch.no_grad():
            q_values = self.source_network(state)
        action = self.policy.select_action(q_values, is_training)
        return action.cpu().detach().numpy()


    def fit(self, env, num_iterations, max_episode_length=None):
        """Fit your model to the provided environment.

        Its a good idea to print out things like loss, average reward,
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
        frame = self.preprocessor.reset(env.reset())
        n_actions = env.action_space.n
        ava_loss, ava_reward = [], []
        all_loss, all_reward, all_i = [], [], []

        self.source_network.train()
        fout = open(f'{self.args.path}/info.txt', 'w')
        for i in trange(num_iterations):
            action = self.select_action(frame, True)
            next_frame, reward, done, _ = env.step(action)
            reward = np.sign(reward)
            ava_reward.append(float(reward))
            next_frame = self.preprocessor.process_state_for_network(next_frame)
            self.memory.append(Sample(s=frame, ns=next_frame, r=reward, ter=done, a=action))
            if not done:
                frame = next_frame
            else:
                frame = self.preprocessor.reset(env.reset())
                if i >= self.args.num_burn_in:
                    all_loss.append(float(np.average(np.array(ava_loss))))
                    all_reward.append(float(np.sum(np.array(ava_reward))))
                    all_i.append(i)
                    print(all_loss[-1], all_reward[-1])
                    fout.write(f'{[i, all_loss[-1], all_reward[-1]]}\n')
                    fout.flush()
                ava_loss, ava_reward = [], []
            if i >= self.args.num_burn_in:
                batch = self.memory.sample(self.args.batch_size)
                inp, y, a = self.process_batch(batch)
                output = self.source_network(inp)
                output = output[range(self.args.batch_size), a]
                loss = self.criterion(output, y)
                loss.backward()
                ava_loss.append(float(loss.detach().cpu().numpy()))
                if i % self.args.train_freq == 0:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                if i % self.args.target_update_freq == 0:
                    utils.get_hard_target_model_updates(self.target_network, self.source_network)
            if i % (num_iterations // 3) == 0:
                self.save(i)
        json.dump({'loss': all_loss, 'reward': all_reward, 'frame': all_i}, open(f'{self.args.path}/info.json', 'w'))


    def save(self, i):
        torch.save(self.source_network.state_dict(), f'{self.args.path}/ep-{i}.bin')

    def evaluate_one_epi(self, env):
        frame = env.reset()
        frames = [frame]
        frame = self.preprocessor.reset(frame)
        done = False
        rewards = 0
        while not done:
            action = self.select_action(frame, False)
            next_frame, reward, done, _ = env.step(action)
            rewards += reward
            frames.append(next_frame)
            next_frame = self.preprocessor.process_state_for_network(next_frame)
            frame = next_frame
        return frames, rewards


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
        frames, _ = self.evaluate_one_epi(env)
        utils.tovideo(frames, self.args.path)
        all_rewards = []
        for i in trange(num_episodes):
            _, rewards = self.evaluate_one_epi(env)
            all_rewards.append(rewards)
        all_rewards = np.array(all_rewards)
        print(np.mean(all_rewards), np.std(all_rewards, ddof=1))
        return all_rewards




    def process_batch(self, samples):
        r = []
        ns = []
        s = []
        ter = []
        a = []
        for ele in samples:
            r.append(ele.r)
            ns.append(ele.ns.reshape(-1).astype(np.float32))
            s.append(ele.s.reshape(-1).astype(np.float32))
            ter.append(ele.ter)
            a.append(int(ele.a))
        r = torch.Tensor(r).to(self.args.device)
        ter = torch.BoolTensor(ter).to(self.args.device)
        ns = torch.from_numpy(np.vstack(ns)).to(self.args.device)
        s = torch.from_numpy(np.vstack(s)).to(self.args.device)
        with torch.no_grad():
            qn = self.target_network(ns)
        rn = r + self.args.gamma * qn.max(-1)[0]
        y = torch.where(ter, r, rn)
        # a = torch.LongTensor(a).to(self.args.device)
        return s, y, a




