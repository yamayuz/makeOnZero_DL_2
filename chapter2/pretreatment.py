""" 自然言語の前処理
"""
import numpy as np

def preprocess(text):
    """ テキストを基にコーパス/単語辞書を作成する。

    Parameters
    ----------
    text : string
        対象のテキスト。

    Returns
    -------
    courps : ndarray
        textを単語IDに変換したリスト。
    word2id : dictionary
        単語リスト（単語→単語ID）
    id2word : dictionary
        単語リスト（単語ID→単語）
    """
    text = text.lower()
    text = text.replace('.', ' .')
    words = text.split()

    word2id = {}
    id2word = {}

    for word in words:
        if not word in word2id:
            word2id[word] = len(word2id)
            id2word[len(word2id)] = word

    courps = np.array([word2id[word] for word in words])

    return courps, word2id, id2word



if __name__ == "__main__":
    courps, word2id, id2word = preprocess('You say goodbye and I say hello.')
    print(word2id)
    print(id2word)
    print(courps)