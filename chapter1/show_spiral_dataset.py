import sys
sys.path.append('..')
from dataset import spiral
import matplotlib.pylab as plt

x, t = spiral.load_data()

# データ点のプロット
# xは2次元のデータが300個ある
# 0～100m 101～200, 201～300でそれぞれ異なるマークでプロット
N = 100
CLS_NUM = 3
markers = ['o', 'x', '^']
for i in range(CLS_NUM):
    plt.scatter(x[i*N:(i+1)*N, 0], x[i*N:(i+1)*N, 1], s=40, marker=markers[i])
plt.show()