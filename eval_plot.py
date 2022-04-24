import json
from pathlib import Path

import gym
from matplotlib import pyplot as plt
import torch
from torch import nn
from tqdm import tqdm

from deeprl_hw2.dqn import DeepQNet, LinearQNet, evaluate

env = gym.make('SpaceInvaders-v0')
save_dir = Path('exps')

def main():
    num_actions = env.action_space.n
    fig, ax = plt.subplots(figsize=(8, 5))
    for q_net, exp_name in [
        (LinearQNet(4, num_actions), 'LinearQN'),
        (LinearQNet(4, num_actions), 'DLinearQN'),
        (DeepQNet(4, num_actions, large=True, duel=False), 'DeepQN'),
        (DeepQNet(4, num_actions, large=True, duel=False), 'DDeepQN'),
        (DeepQNet(4, num_actions, large=True, duel=True), 'Duel-DeepQN'),
    ]:
        mean_rewards = []
        exp_save_dir = save_dir / exp_name
        exp_plot_data_dir = exp_save_dir / 'plot-data.json'
        if exp_plot_data_dir.exists():
            with open(exp_plot_data_dir) as f:
                mean_rewards: list[float] = json.load(f)
        save_its = [i * 100000 for i in range(1, 50)]
        save_its.append(4999998)
        q_net: nn.Module = q_net.to('cuda')
        for save_it in tqdm(save_its[len(mean_rewards):], ncols=80, desc=f'evaluating {exp_name}'):
            q_net.load_state_dict(torch.load(exp_save_dir / f'SpaceInvaders-v0-{save_it}.pth'))
            mean, std = evaluate(env, q_net, num_episodes=20, pbar=False)
            mean_rewards.append(mean)
            with open(exp_plot_data_dir, 'w') as f:
                json.dump(mean_rewards, f, indent=4)
        ax.plot(save_its, mean_rewards, label=exp_name)

    ax.legend()
    fig.savefig('performance-plot.pdf')
    plt.show()

if __name__ == '__main__':
    main()
