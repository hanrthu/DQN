import matplotlib.pyplot as plt
import collections

plt.figure(figsize=(10, 5), dpi=100)
step = 50000

def process(fn):
    fin = open(fn)
    keys = collections.defaultdict(list)
    for lines in fin.readlines():
        lis = eval(lines.strip())
        keys[lis[0] // step].append(lis[2])
    x, y = [], []
    for key, val in keys.items():
        x.append(key * step + step / 2)
        y.append(sum(val) / len(val))
    return x, y


def plot(fn, name):
    x, y = process(fn)
    plt.plot(x, y, label=name)


data = ['NoReplayDQN', 'LinearDQN', 'LinearDDQN', 'DeepDQN', 'DeepDDQN']
for ele in data:
    plot(ele + '/info.txt', ele)

plt.legend()
plt.savefig('plot.jpg')