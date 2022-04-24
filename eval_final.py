import json
from pathlib import Path

import gym
import torch

from deeprl_hw2.dqn import DeepQNet, LinearQNet, evaluate

env = gym.make('SpaceInvaders-v0')
save_dir = Path('exps')

def main():
    results = {}
    results_path = Path('final-results.json')
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
    num_actions = env.action_space.n
    for q_net, exp_name in [
        (LinearQNet(4, num_actions), 'LinearQN'),
        (LinearQNet(4, num_actions), 'DLinearQN'),
        (DeepQNet(4, num_actions, large=True, duel=False), 'DeepQN'),
        (DeepQNet(4, num_actions, large=True, duel=False), 'DDeepQN'),
        (DeepQNet(4, num_actions, large=True, duel=True), 'Duel-DQN'),
    ]:
        if exp_name in results:
            continue
        q_net = q_net.cuda()
        q_net.load_state_dict(torch.load(save_dir / exp_name / f'SpaceInvaders-v0-4999998.pth'))
        mean, std, best_video = evaluate(env, q_net, num_episodes=100, pbar=True)
        results[exp_name] = {'mean': mean, 'std': std}
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
    with open('final-results.tex', 'w') as f:
        print(r'\toprule', file=f)
        print(r'model & reward \\', file=f)
        print(r'\midrule', file=f)
        for exp_name, result in results.items():
            print(fr'{exp_name} & ${result["mean"]} \pm {result["std"]:.3f}$ \\', file=f)
        print(r'\bottomrule', file=f, end='')

if __name__ == '__main__':
    main()
