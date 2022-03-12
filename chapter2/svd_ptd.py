"""
PTBデータセットを使用し、SVDの評価を行う
"""
import sys
from co_matrix import create_co_matrix
sys.path.append('..')
from dataset import ptb
from ppmi import *
import numpy as np

if __name__ == "__main__":
    window_size = 2
    wordvec_size = 100

    # 訓練用のテキストデータからコーパス、辞書を作成し、
    # 共起行列、ppmi行列を作成
    corpus, word_to_id, id_to_word = ptb.load_data('train')
    vocab_size = len(word_to_id)
    print('counting co-occurrence ...')
    C = create_co_matrix(corpus, vocab_size, window_size)
    print('calculating PPMI ...')
    W = ppmi(C, verbose=True)

    # SVDによりppmi行列の次元を削減
    print('calculating SVD ...')
    try:
        from sklearn.utils.extmath import randomized_svd
        U, S, V = randomized_svd(W, n_components=wordvec_size, n_iter=5, random_state=None)
    except ImportError:
        U, S, V = np.linalg.svd(W)

    word_vecs = U[:, :wordvec_size]

    # クエリで指定した単語に対して、
    # 類似度が上位5つまでの単語を表示する
    querys = ['you', 'year', 'car', 'toyota']
    for query in querys:
        most_similar(query, id_to_word, word_to_id, word_vecs, top=5)