from argparse import ArgumentParser
import logging
from pathlib import Path

import gym

import numpy as np
import torch

from deeprl_hw2.dqn import DQNAgent, DeepQNet, LinearQNet
from deeprl_hw2.policy import GreedyPolicy
from deeprl_hw2.preprocessors import AtariPreprocessor, HistoryPreprocessor, PreprocessorSequence
from deeprl_hw2.utils import make_env

env = gym.make('SpaceInvaders-v0')
env = make_env(env)

def main():
    parser = ArgumentParser()
    parser.add_argument('--num_trials', type=int, default=20)
    parser.add_argument('--deep', action='store_true')
    parser.add_argument('--large', action='store_true')
    parser.add_argument('--weights', type=Path, default=None)
    args = parser.parse_args()
    HISTORY_LENGTH = 4

    num_actions = env.action_space.n
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.deep:
        q_net = DeepQNet(HISTORY_LENGTH, num_actions, args.large).to(device)
        print(q_net)
    else:
        q_net = LinearQNet(HISTORY_LENGTH, num_actions).to(device)

    if args.weights is not None:
        q_net.load_state_dict(torch.load(args.weights))
        print(f'load state dict from {args.weights}')

    atari_pro = AtariPreprocessor(84)
    history_pro = HistoryPreprocessor(HISTORY_LENGTH)
    preprocessor = PreprocessorSequence([atari_pro, history_pro])
    logger = logging.getLogger('Q-Learning Logger')
    agent = DQNAgent(
        q_net,
        qminus_network=None,
        preprocessor=preprocessor,
        memory=None,
        gamma=None,
        start_eps=None,
        end_eps=None,
        target_update_freq=None,
        num_burn_in=None,
        train_freq=None,
        eval_freq=None,
        batch_size=None,
        iterations=None,
        final_exploration_frame=None,
        optimizer=None,
        scheduler=None,
        device=device,
        logger=logger,
        args=None,
    )

    results = []
    for _ in range(args.num_trials):
        state: torch.ByteTensor = preprocessor.reset(env.reset())
        policy = GreedyPolicy()
        terminate = False
        iteration = 0
        tot = 0
        while not terminate:
            state_n = (state / 255).to(device)
            action = policy.select_action(state_n, agent.calc_q_values, is_training=False)
            obs, reward, terminate, _ = env.step(action)
            tot += reward
            state = preprocessor.process_state_for_memory(obs)
            iteration += 1
        print(tot)
        results.append(tot)
    print(np.mean(results), np.std(results))

if __name__ == '__main__':
    main()
