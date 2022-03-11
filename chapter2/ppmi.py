""" 
共起行列の代わりに相互情報量（PMI:Pointwise Mutual Information）を使用する。
共起行列では単語の共起回数をカウントしているため、
性能があまりよくない（冠詞などの高頻出単語との関係性が高く算出される）
"""
import numpy as np
from pretreatment import *
from co_matrix import *

def ppmi(C, verbose=False, eps=1e-8):
    """ 共起行列を基にppmiを計算する。
    Parameters
    ----------
    C : ndarray([vocab_size, vocab_size])
        共起行列
    verbose : boolean
        進捗率の表示有無（True:表示、False:非表示）

    Returns
    -------
    M : ndarray([vocab_size, vocab_size])
        ppmi行列
    """
    # M:PPMI格納用の行列
    # N:総単語数（共起行列の総和を近似的に使用）
    # S:各単語の出現回数（単語ベクトルの要素の和を近似的に使用）
    # total:共起行列の要素数
    M = np.zeros_like(C, dtype=np.float32)
    N = np.sum(C)
    S = np.sum(C, axis=0)
    total = C.size
    cnt = 0

    # ppmiを計算する
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i, j] * N / (S[j]*S[i] + eps))
            M[i, j] = max(0, pmi)

            # 進捗率の表示
            if verbose:
                cnt += 1
                if cnt % (total//100 + 1) == 0:
                    print('%.1f%% done' % (100*cnt/total))
    return M



if __name__ == "__main__":
    courps, word2id, id2word = preprocess('You say goodbye and I say hello.')
    word_matrix = create_co_matrix(courps, len(word2id))

    # ppmiの算出
    M = ppmi(word_matrix)

    np.set_printoptions(precision=3)
    print('covariance matrix')
    print(word_matrix)
    print('-'*50)
    print('PPMI')
    print(M)
