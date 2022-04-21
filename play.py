import gym
from matplotlib import pyplot as plt

from deeprl_hw2.utils import make_env

def main():
    env = gym.make('Enduro-v0')
    env = make_env(env)
    obs = env.reset()
    im = plt.imshow(obs)
    plt.show(block=False)
    while True:
        action = int(input('action:'))
        obs, reward, terminate, _ = env.step(action)
        print(reward, terminate)
        im.set_data(obs)
        plt.gcf().canvas.draw_idle()

if __name__ == '__main__':
    main()
