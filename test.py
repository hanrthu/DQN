import gym

import numpy as np

from deeprl_hw2.utils import make_env

env = gym.make('Enduro-v0')
env = make_env(env)

def main():
    results = []
    for _ in range(20):
        env.reset()
        tot = 0
        while True:
            _, reward, terminate, _ = env.step(1)
            tot += reward
            if terminate:
                break
        print(tot)
        results.append(tot)
    print(np.mean(results), np.std(results))

if __name__ == '__main__':
    main()
