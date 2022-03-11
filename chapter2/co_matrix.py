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


def cos_similarity(x, y, eps=1e-8):
    """ ベクトルxとyの類似度を計算する。
    Returns
    -------
    similarity : int32
        ベクトルxとyの類似度。
    """
    # ベクトルのノルムが0の場合に、0除算を防ぐためespを加算する。
    x = x / (np.sqrt(np.sum(x**2)) + eps)
    y = y / (np.sqrt(np.sum(y**2)) + eps)
    return np.dot(x, y)


def most_similar(query, id2word, word2id, word_matrix, top=5):
    """ 単語の類似度を計算し、ランキング(降順)で表示する。
    Parameters
    ----------
    query : string
        類似度を計算する対象の単語。
    id2word : dictionary
        単語リスト（単語ID→単語）
    word2id : dictionary
        単語リスト（単語→単語ID）
    word_matrix : ndarray([vocab_size, vocab_size])
        励起行列。
    top : int
        表示する単語数（上位）
    """
    if query not in word2id:
        print('%s is not found' % query)
        return

    print('\n[query] ' + query)
    query_id = word2id[query]
    query_vector = word_matrix[query_id]

    # 類似度を計算
    similarity = np.zeros(len(id2word))
    for idx, word_vector in enumerate(word_matrix):
        similarity[idx] = cos_similarity(query_vector, word_vector)

    # 類似度を降順にソートして表示
    # argsort()には降順にソートするオプションがないため「-1」を掛ける
    count = 0
    for i in (- similarity).argsort():
        if id2word[i] == query:
            continue
        print(' %s: %s'% (id2word[i], similarity[i]))

        count += 1
        if count >= top:
            return




if __name__ == "__main__":
    courps, word2id, id2word = preprocess('You say goodbye and I say hello.')
    word_matrix = create_co_matrix(courps, len(word2id))

    # 「you」と「i」の類似度を計算
    # x = word_matrix[word2id['you']]
    # y = word_matrix[word2id['i']]
    # print(similarity(x,y))

    # 類似度のランキングを表示
    query = 'you'
    most_similar(query, id2word, word2id, word_matrix)