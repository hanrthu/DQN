import json
from pathlib import Path

import cv2
import gym
import torch

from deeprl_hw2.dqn import DeepQNet, LinearQNet, evaluate

env = gym.make('SpaceInvaders-v0')
save_dir = Path('exps')
video_dir = Path('videos')
video_dir.mkdir(parents=True, exist_ok=True)

def main():
    num_actions = env.action_space.n
    for q_net, exp_name in [
        (LinearQNet(4, num_actions), 'DLinearQN'),
        (DeepQNet(4, num_actions, large=True, duel=False), 'DeepQN'),
        (DeepQNet(4, num_actions, large=True, duel=False), 'DDeepQN'),
        (DeepQNet(4, num_actions, large=True, duel=True), 'Duel-DQN'),
    ]:
        print(exp_name)
        q_net = q_net.cuda()
        for i in range(4):
            video_path = video_dir / f'{exp_name}-{i}-3.avi'
            if video_path.exists():
                continue
            if i:
                q_net.load_state_dict(torch.load(save_dir / exp_name / f'SpaceInvaders-v0-{i * 1666666}.pth'))
            mean, std, best_video = evaluate(env, q_net, num_episodes=20, pbar=True)
            video_writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 50.0, (160, 210))
            for frame in best_video:
                video_writer.write(frame)

if __name__ == '__main__':
    main()
