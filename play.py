from argparse import ArgumentParser
import logging
from pathlib import Path
import random
from typing import Any, Optional

import gym
from matplotlib import pyplot as plt
from matplotlib.backend_bases import KeyEvent
import numpy as np
import torch

from deeprl_hw2.dqn import DQNAgent, DeepQNet
from deeprl_hw2.policy import GreedyPolicy
from deeprl_hw2.preprocessors import AtariPreprocessor, HistoryPreprocessor, PreprocessorSequence

env = gym.make('SpaceInvaders-v0')

plt.ion()
reset_img = env.reset()
im = plt.imshow(reset_img)
fig = plt.gcf()
tot = 0
step = 0

def act(action: int) -> tuple[bool, Optional[np.ndarray[Any, np.uint8]]]:
    global tot, step
    if action not in range(env.action_space.n):
        return False, None
    obs, reward, terminate, _ = env.step(action)
    im.set_data(obs)
    # fig.canvas.draw_idle()
    print(f'{step}\t{action}\t{reward}\t{terminate}\t{tot}')
    tot += reward
    step += 1
    return terminate, obs

def main():
    parser = ArgumentParser()
    parser.add_argument('-i', action='store_true')
    parser.add_argument('--large', action='store_true')
    parser.add_argument('--weights', type=Path, default=None)
    args = parser.parse_args()
    if args.i:
        def on_key_press(event: KeyEvent):
            try:
                action = int(event.key)
            except ValueError:
                return
            act(action)
        fig.canvas.mpl_connect('key_press_event', on_key_press)
        plt.show(block=True)
    else:
        plt.show()
        if args.weights is None:
            while not act(random.randint(0, env.action_space.n - 1))[0]:
                # time.sleep(3)
                plt.pause(0.1)
        else:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            num_actions = env.action_space.n
            HISTORY_LENGTH = 4
            q_net = DeepQNet(HISTORY_LENGTH, num_actions, args.large).to(device)
            # q_net = LinearQNet(HISTORY_LENGTH, num_actions)
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

            state: torch.ByteTensor = preprocessor.reset(reset_img)
            # policy = agent.select_policy(Gre, num_actions)
            policy = GreedyPolicy()
            terminate = False
            iteration = 0
            while not terminate:
                state_n = (state / 255).to(device)
                action = policy.select_action(state_n, agent.calc_q_values, is_training=False)
                terminate, obs = act(action)
                plt.pause(0.001)
                state = preprocessor.process_state_for_memory(obs)
                iteration += 1

if __name__ == '__main__':
    main()
