import numpy as np

# ソフトマックス関数
# dim=1の場合は単純に最大値を取得すればよいが、
# dim=2の場合は各行毎(各データ毎)に最大値を取得する
# 必要があるので、場合分けする
def softmax(x):
    if x.ndim == 2:
        x = x - x.max(axis=1, keepdims=True)
        x = np.exp(x)
        x /= x.sum(axis=1, keepdims=True)
    elif x.ndim == 1:
        x = x - np.max(x)
        x = np.exp(x) / np.sum(np.exp(x))
    return x


# 交差エントロピー誤差
def cross_entropy_error(x, t):
    if x.ndim == 1:
        t = t.reshape(1, t.size)
        x = x.reshape(1, x.size)

    batch_size = x.shape[0]
    return -np.sum(t * np.log(x + 1e-7)) / batch_size


# def cross_entropy_error(y, t):
#     if y.ndim == 1:
#         t = t.reshape(1, t.size)
#         y = y.reshape(1, y.size)
        
#     # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
#     if t.size == y.size:
#         t = t.argmax(axis=1)
             
#     batch_size = y.shape[0]

#     return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
