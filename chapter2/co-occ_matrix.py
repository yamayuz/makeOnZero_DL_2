"""
共起行列：単語をベクトルとして表現し、全単語分のベクトルをまとめた行列。
行：各単語に対応するベクトル。
列：ベクトルの成分。
"""
from pretreatment import preprocess
import numpy as np

def create_co_matrix(courps, vocab_size, window_size=1):
    """ コーパスを基に励起行列を作成する。

    Parameters
    ----------
    courps : ndarray
        コーパス。
    vocab_size : int
        単語の数（単語辞書の要素数）
    window_size : int
        励起行列を作成する際のウィンドウサイズ。

    Returns
    -------
    co_matrix : ndarray([vocab_size, vocab_size])
        励起行列。
    """
    corpus_size = len(courps)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for idx, word_id in enumerate(courps):
        left_idx = idx - window_size
        right_idx = idx + window_size

        if left_idx >= 0:
            left_word_id = courps[left_idx]
            co_matrix[word_id][left_word_id] += 1

        if right_idx < corpus_size:
            right_word_id = courps[right_idx]
            co_matrix[word_id][right_word_id] += 1

    return co_matrix



if __name__ == "__main__":
    courps, word2id, id2word = preprocess('You say goodbye and I say hello.')
    co_matrix = create_co_matrix(courps, len(word2id))
    print(co_matrix)