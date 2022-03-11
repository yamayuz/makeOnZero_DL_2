"""
特異値分解（SVD:Singular Value Decomposition）により共起行列を分解し、
単語ベクトルの次元を下げる。
直交行列を基底として、その線形結合の形に分解する。
係数の値が小さくベクトルへの寄与が小さい方向については
削除することで次元を下げる。
"""
import numpy as np
import matplotlib.pyplot as plt
from pretreatment import *
from co_matrix import *
from ppmi import *


if __name__ == "__main__":
    courps, word2id, id2word = preprocess('You say goodbye and I say hello.')
    word_matrix = create_co_matrix(courps, len(word2id))
    M = ppmi(word_matrix)

    # SVGを実行
    U, S, V = np.linalg.svd(M)

    # np.set_printoptions(precision=3)
    # print(word_matrix[0])
    # print(M)
    # print(V[0])

    # グラフを描画
    for word, word_id in word2id.items():
        plt.annotate(word, (U[word_id, 1], U[word_id, 0]))

    plt.scatter(U[:,1], U[:,0], alpha=0.5)
    plt.show()