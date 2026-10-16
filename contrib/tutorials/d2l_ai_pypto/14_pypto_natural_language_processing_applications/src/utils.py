"""纯 PyTorch 公共工具模块（替代 d2l 包中的常用函数）。

提供 d2l 常用函数的纯 PyTorch 等价实现，供各章 notebook 通过
``from src.utils import ...`` 复用，避免依赖 d2l 包。

覆盖函数：
- 文本处理：read_time_machine, tokenize, Vocab, truncate_pad, TokenEmbedding
- 数据迭代：seq_data_iter_random, seq_data_iter_sequential, SeqDataLoader
- 数据加载：load_corpus_time_machine, load_data_time_machine,
            read_imdb, load_data_imdb
- 可视化：Animator, plot, set_figsize
- 训练辅助：Timer, Accumulator, sgd, synthetic_data, accuracy,
            evaluate_accuracy_gpu, train_batch_ch13, train_ch13
- 设备选择：try_gpu, try_all_gpus
"""

import collections
import hashlib
import logging
import math
import os
import random
import re
import tarfile
import time
import urllib.request
import zipfile

# pypto 导入会把根日志级别改为 DEBUG，导致 matplotlib 等库输出大量 DEBUG 日志，先恢复为 WARNING
logging.getLogger().setLevel(logging.WARNING)

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from IPython import display

# =========================================================================
# 设备选择
# =========================================================================


def try_gpu(i=0):
    """返回 npu(i)/gpu(i) 如果存在，否则返回 cpu()。"""
    if hasattr(torch, "npu") and torch.npu.device_count() >= i + 1:
        return torch.device(f"npu:{i}")
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f"cuda:{i}")
    return torch.device("cpu")


def try_all_gpus():
    """返回所有可用的 NPU/GPU，如果没有则返回 [cpu()]。"""
    devices = []
    if hasattr(torch, "npu"):
        devices += [torch.device(f"npu:{i}")
                    for i in range(torch.npu.device_count())]
    devices += [torch.device(f"cuda:{i}")
                for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device("cpu")]

# =========================================================================
# 数据下载
# =========================================================================


DATA_HUB = {}
DATA_URL = "https://d2l-data.s3-accelerate.amazonaws.com/"


def download(name, cache_dir=os.path.join("..", "data")):
    """下载 DATA_HUB 中的文件并返回本地文件名。"""
    assert name in DATA_HUB, f"{name} 不在 DATA_HUB 中"
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split("/")[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, "rb") as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname
    print(f"正在从 {url} 下载 {fname}...")
    r = urllib.request.urlopen(url)
    with open(fname, "wb") as f:
        f.write(r.read())
    return fname


def download_extract(name, folder=None):
    """下载并解压 zip/tar 文件。"""
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == ".zip":
        fp = zipfile.ZipFile(fname, "r")
    elif ext in (".tar", ".gz"):
        fp = tarfile.open(fname, "r")
    else:
        assert False, "只有 zip/tar 文件可以被解压缩"
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir


DATA_HUB["glove.6b.50d"] = (
    DATA_URL + "glove.6B.50d.zip",
    "0b8703943ccdb6eb788e6f091b8946e82231bc4d",
)
DATA_HUB["glove.6b.100d"] = (
    DATA_URL + "glove.6B.100d.zip",
    "cd43bfb07e44e6f27cbcc7bc9ae3d80284fdaf5a",
)
DATA_HUB["glove.42b.300d"] = (
    DATA_URL + "glove.42B.300d.zip",
    "b5116e234e9eb9076672cfeabf5469f3eec904fa",
)
DATA_HUB["aclImdb"] = (
    "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
    "01ada507287d82875905620988597833ad4e0903",
)


DATA_HUB["time_machine"] = (
    DATA_URL + "timemachine.txt",
    "090b5e7e70c295757f55df93cb0a180b9691891a",
)

DATA_HUB["the_war_of_the_worlds"] = (
    "https://www.gutenberg.org/files/36/36-0.txt",
    "f0daadda9d37fdf7a7c5587dd725c04d240c48b9",
)

DATA_HUB["SNLI"] = (
    "https://nlp.stanford.edu/projects/snli/snli_1.0.zip",
    "9fcde07509c7e87ec61c640c1b2753d9041758e4",
)

DATA_HUB["bert.base"] = (
    DATA_URL + "bert.base.torch.zip",
    "225d66f04cae318b841a13d32af3acc165f253ac",
)
DATA_HUB["bert.small"] = (
    DATA_URL + "bert.small.torch.zip",
    "c72329e68a732bef0452e4b96a1c341c8910f81f",
)

# =========================================================================
# 文本预处理
# =========================================================================


def read_time_machine():
    """将时间机器数据集加载到文本行的列表中。"""
    with open(download("time_machine"), "r") as f:
        lines = f.readlines()
    return [re.sub("[^A-Za-z]+", " ", line).strip().lower() for line in lines]


def read_the_war_of_the_worlds():
    """将世界大战数据集加载到文本行的列表中。"""
    with open(download("the_war_of_the_worlds"), "r") as f:
        lines = f.readlines()
    return [re.sub("[^A-Za-z]+", " ", line).strip().lower() for line in lines]


def tokenize(lines, token="word"):
    """将文本行拆分为词元列表。

    Args:
        lines: 字符串列表
        token: "word" 表示按空格分词，"char" 表示按字符分词

    Returns:
        词元列表的列表
    """
    if token == "word":
        return [line.split() for line in lines]
    elif token == "char":
        return [list(line) for line in lines]
    else:
        raise ValueError(f"未知的词元类型：{token}")


class Vocab:
    """文本词表。"""

    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        self.idx_to_token = ["<unk>"] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs


def count_corpus(tokens):
    """统计词元的频率。"""
    if len(tokens) == 0 or isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


def truncate_pad(line, num_steps, padding_token):
    """截断或填充文本序列。"""
    if len(line) > num_steps:
        return line[:num_steps]
    return line + [padding_token] * (num_steps - len(line))


class TokenEmbedding:
    """词元的嵌入向量（加载预训练的 GloVe/fastText 词向量）。"""

    def __init__(self, embedding_name):
        self.idx_to_token, self.idx_to_vec = self._load_embedding(
            embedding_name)
        self.unknown_idx = 0
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}

    def _load_embedding(self, embedding_name):
        idx_to_token, idx_to_vec = ["<unk>"], []
        data_dir = download_extract(embedding_name)
        # GloVe website: https://nlp.stanford.edu/projects/glove/
        # fastText website: https://fasttext.cc/
        with open(os.path.join(data_dir, "vec.txt"), "r") as f:
            for line in f:
                elems = line.rstrip().split(" ")
                token, elems = elems[0], [float(elem) for elem in elems[1:]]
                # 跳过头部信息，例如 fastText 的第一行
                if len(elems) > 1:
                    idx_to_token.append(token)
                    idx_to_vec.append(elems)
        idx_to_vec = [[0] * len(idx_to_vec[0])] + idx_to_vec
        return idx_to_token, torch.tensor(idx_to_vec)

    def __getitem__(self, tokens):
        indices = [self.token_to_idx.get(token, self.unknown_idx)
                   for token in tokens]
        vecs = self.idx_to_vec[torch.tensor(indices)]
        return vecs

    def __len__(self):
        return len(self.idx_to_token)

# =========================================================================
# 序列数据迭代器
# =========================================================================


def seq_data_iter_random(corpus, batch_size, num_steps):
    """使用随机抽样生成一个小批量子序列。

    Args:
        corpus: 整数索引列表
        batch_size: 批量大小
        num_steps: 时间步数

    Yields:
        (X, Y): 特征和标签张量
    """
    corpus = corpus[random.randint(0, num_steps - 1):]
    num_subseqs = (len(corpus) - 1) // num_steps
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    random.shuffle(initial_indices)

    def data(pos):
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)


def seq_data_iter_sequential(corpus, batch_size, num_steps):
    """使用顺序分区生成一个小批量子序列。

    Args:
        corpus: 整数索引列表
        batch_size: 批量大小
        num_steps: 时间步数

    Yields:
        (X, Y): 特征和标签张量
    """
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y


class SeqDataLoader:
    """加载序列数据的迭代器。"""

    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential
        self.corpus, self.vocab = load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)


def load_corpus_time_machine(max_tokens=-1):
    """返回时光机器数据集的词元索引列表和词表。"""
    lines = read_time_machine()
    tokens = tokenize(lines, "char")
    vocab = Vocab(tokens)
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab


def load_data_time_machine(batch_size, num_steps,
                           use_random_iter=False, max_tokens=10000):
    """返回时光机器数据集的迭代器和词表。"""
    data_iter = SeqDataLoader(
        batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab


def load_corpus_the_war_of_the_worlds(max_tokens=-1):
    """返回世界大战数据集的词元索引列表和词表。"""
    lines = read_the_war_of_the_worlds()
    tokens = tokenize(lines, "char")
    vocab = Vocab(tokens)
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab


class _SeqDataLoaderGeneric:
    """通用序列数据加载器（支持自定义 corpus/vocab）。"""

    def __init__(self, batch_size, num_steps, use_random_iter, corpus, vocab):
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential
        self.corpus = corpus
        self.vocab = vocab
        self.batch_size = batch_size
        self.num_steps = num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)


def load_data_the_war_of_the_worlds(batch_size, num_steps,
                                     use_random_iter=False, max_tokens=10000):
    """返回世界大战数据集的迭代器和词表。"""
    corpus, vocab = load_corpus_the_war_of_the_worlds(max_tokens)
    data_iter = _SeqDataLoaderGeneric(
        batch_size, num_steps, use_random_iter, corpus, vocab)
    return data_iter, vocab


# ---------------------------------------------------------------------------
# IMDb 情感分析数据集
# ---------------------------------------------------------------------------


def read_imdb(data_dir, is_train):
    """读取 IMDb 评论数据集的文本序列和标签。"""
    data, labels = [], []
    for label in ("pos", "neg"):
        folder_name = os.path.join(data_dir, "train" if is_train else "test",
                                   label)
        for file in os.listdir(folder_name):
            with open(os.path.join(folder_name, file), "rb") as f:
                review = f.read().decode("utf-8").replace("\n", "")
                data.append(review)
                labels.append(1 if label == "pos" else 0)
    return data, labels


def load_data_imdb(batch_size, num_steps=500):
    """返回 IMDb 评论数据集的数据迭代器和词表。"""
    data_dir = download_extract("aclImdb", "aclImdb")
    train_data = read_imdb(data_dir, True)
    test_data = read_imdb(data_dir, False)
    train_tokens = tokenize(train_data[0], token="word")
    test_tokens = tokenize(test_data[0], token="word")
    vocab = Vocab(train_tokens, min_freq=5)
    train_features = torch.tensor([truncate_pad(
        vocab[line], num_steps, vocab["<pad>"]) for line in train_tokens])
    test_features = torch.tensor([truncate_pad(
        vocab[line], num_steps, vocab["<pad>"]) for line in test_tokens])
    train_iter = load_array(
        (train_features, torch.tensor(train_data[1])), batch_size)
    test_iter = load_array(
        (test_features, torch.tensor(test_data[1])),
        batch_size, is_train=False)
    return train_iter, test_iter, vocab

# ---------------------------------------------------------------------------
# SNLI 自然语言推断数据集
# ---------------------------------------------------------------------------


def read_snli(data_dir, is_train):
    """将SNLI数据集解析为前提、假设和标签。"""
    def extract_text(s):
        # 删除我们不会使用的信息
        s = re.sub('\\(', '', s)
        s = re.sub('\\)', '', s)
        # 用一个空格替换两个或多个连续的空格
        s = re.sub('\\s{2,}', ' ', s)
        return s.strip()
    label_set = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
    file_name = os.path.join(data_dir, 'snli_1.0_train.txt'
                             if is_train else 'snli_1.0_test.txt')
    with open(file_name, 'r') as f:
        rows = [row.split('\t') for row in f.readlines()[1:]]
    premises = [extract_text(row[1]) for row in rows if row[0] in label_set]
    hypotheses = [extract_text(row[2]) for row in rows if row[0]
                  in label_set]
    labels = [label_set[row[0]] for row in rows if row[0] in label_set]
    return premises, hypotheses, labels


class SNLIDataset(torch.utils.data.Dataset):
    """用于加载SNLI数据集的自定义数据集。"""

    def __init__(self, dataset, num_steps, vocab=None):
        self.num_steps = num_steps
        all_premise_tokens = tokenize(dataset[0])
        all_hypothesis_tokens = tokenize(dataset[1])
        if vocab is None:
            self.vocab = Vocab(all_premise_tokens +
                               all_hypothesis_tokens, min_freq=5,
                               reserved_tokens=['<pad>'])
        else:
            self.vocab = vocab
        self.premises = self._pad(all_premise_tokens)
        self.hypotheses = self._pad(all_hypothesis_tokens)
        self.labels = torch.tensor(dataset[2])
        print('read ' + str(len(self.premises)) + ' examples')

    def _pad(self, lines):
        return torch.tensor([truncate_pad(
            self.vocab[line], self.num_steps, self.vocab['<pad>'])
                         for line in lines])

    def __getitem__(self, idx):
        return (self.premises[idx], self.hypotheses[idx]), self.labels[idx]

    def __len__(self):
        return len(self.premises)


def load_data_snli(batch_size, num_steps=50):
    """下载SNLI数据集并返回数据迭代器和词表。"""
    num_workers = get_dataloader_workers()
    data_dir = download_extract('SNLI')
    train_data = read_snli(data_dir, True)
    test_data = read_snli(data_dir, False)
    train_set = SNLIDataset(train_data, num_steps)
    test_set = SNLIDataset(test_data, num_steps, train_set.vocab)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size,
                                             shuffle=True,
                                             num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(test_set, batch_size,
                                            shuffle=False,
                                            num_workers=num_workers)
    return train_iter, test_iter, train_set.vocab

# =========================================================================
# 计时器
# =========================================================================


class Timer:
    """记录多次运行时间。"""

    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """启动计时器。"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中。"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间。"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和。"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间。"""
        return torch.tensor(self.times).cumsum(dim=0).tolist()

# =========================================================================
# 累加器
# =========================================================================


class Accumulator:
    """在 n 个变量上累加求和。"""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# =========================================================================
# 可视化
# =========================================================================


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """设置 matplotlib 坐标轴的属性。"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    if xlim:
        axes.set_xlim(xlim)
    if ylim:
        axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


def use_svg_display():
    """使用 SVG 格式在 notebook 中显示图形。"""
    try:
        import matplotlib_inline
        matplotlib_inline.backend_inline.set_matplotlib_formats("svg")
    except (ImportError, AttributeError):
        pass


def set_figsize(figsize=(3.5, 2.5)):
    """设置 matplotlib 图表大小。"""
    use_svg_display()
    plt.rcParams["figure.figsize"] = figsize


class Animator:
    """在动画中绘制数据。"""

    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale="linear", yscale="linear",
                 fmts=("-", "m--", "g-.", "r:"), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        if legend is None:
            legend = []
        use_svg_display()
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes]
        self.config_axes = lambda: set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        """向图表中添加多个数据点。"""
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x_vals, y_vals, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x_vals, y_vals, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)


def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale="linear", yscale="linear",
         fmts=("-", "m--", "g-.", "r:"), figsize=(3.5, 2.5), axes=None):
    """绘制数据点。"""
    if legend is None:
        legend = []

    def _set_axes(ax):
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        if legend:
            ax.legend(legend)
        ax.grid()

    if axes is None:
        _, axes = plt.subplots(1, 1, figsize=figsize)
    axes.cla()
    if Y is not None:
        if isinstance(Y, (list, tuple)):
            Y = [y.detach().cpu().numpy() if torch.is_tensor(y) else y
                 for y in Y]
        elif torch.is_tensor(Y):
            Y = Y.detach().cpu().numpy()
        if isinstance(X, (list, tuple)):
            X = [x.detach().cpu().numpy() if torch.is_tensor(x) else x
                 for x in X]
        elif torch.is_tensor(X):
            X = X.detach().cpu().numpy()
        if isinstance(X, (list, tuple)) and isinstance(Y, (list, tuple)):
            for x_i, y_i in zip(X, Y):
                axes.plot(x_i, y_i)
        else:
            axes.plot(X, *Y) if isinstance(Y, list) else axes.plot(X, Y)
    else:
        if isinstance(X, (list, tuple)):
            if len(X) > 0 and isinstance(X[0], (list, tuple, np.ndarray)):
                for x_i in X:
                    if torch.is_tensor(x_i):
                        x_i = x_i.detach().cpu().numpy()
                    axes.plot(x_i)
            elif torch.is_tensor(X[0]) if len(X) > 0 else False:
                X = [x_i.detach().cpu().numpy() if torch.is_tensor(x_i) else x_i
                     for x_i in X]
                for x_i in X:
                    axes.plot(x_i)
            else:
                axes.plot(list(X))
        else:
            if torch.is_tensor(X):
                X = X.detach().cpu().numpy()
            axes.plot(X)
    _set_axes(axes)
    plt.show()

# =========================================================================
# 优化
# =========================================================================


def sgd(params, lr, batch_size):
    """小批量随机梯度下降。"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


def synthetic_data(w, b, num_examples, device=None):
    """生成 y = Xw + b + noise 的合成数据。"""
    X = torch.normal(0, 1, (num_examples, len(w)), device=device)
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape, device=device)
    return X, y.reshape((-1, 1))

# =========================================================================
# 数据加载辅助
# =========================================================================


def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器。"""
    dataset = TensorDataset(*data_arrays)
    return DataLoader(dataset, batch_size, shuffle=is_train)


def get_dataloader_workers():
    """使用 4 个进程来读取数据。"""
    return 4


# =========================================================================
# 模型评估
# =========================================================================


def evaluate_loss(net, data_iter, loss):
    """评估给定数据集上模型的损失。"""
    metric = Accumulator(2)
    for X, y in data_iter:
        out = net(X)
        l = loss(out, y)
        metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]


def accuracy(y_hat, y):
    """计算预测正确的数量。"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy_gpu(net, data_iter, device=None):
    """使用 GPU 计算模型在数据集上的准确率。"""
    if isinstance(net, nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

# =========================================================================
# RNN 预测
# =========================================================================


def predict_ch8(prefix, num_preds, net, vocab, device):
    """在 prefix 后生成新字符。

    Args:
        prefix:   字符串前缀
        num_preds: 要生成的字符数
        net:      RNN 模型（有 begin_state 方法）
        vocab:    词表
        device:   设备

    Returns:
        生成的字符串
    """
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape(
        (1, 1))

    for y in prefix[1:]:
        _, state = net(get_input(), state)
        outputs.append(vocab[y])

    for _ in range(num_preds):
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))

    return "".join([vocab.idx_to_token[i] for i in outputs])


# =========================================================================
# 梯度裁剪
# =========================================================================


def grad_clipping(net, theta):
    """裁剪梯度。

    Args:
        net:   模型（nn.Module 或有 .params 属性）
        theta: 裁剪阈值
    """
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


# =========================================================================
# RNN 训练
# =========================================================================


def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """训练网络一个迭代周期（第8章风格）。

    Args:
        net:             模型
        train_iter:      训练数据迭代器
        loss:            损失函数
        updater:         优化器 或 自定义更新函数
        device:          设备
        use_random_iter: 是否使用随机采样

    Returns:
        (困惑度, 速度)
    """
    state, timer = None, Timer()
    metric = Accumulator(2)

    for X, Y in train_iter:
        if state is None or use_random_iter:
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                state = state.detach()
            else:
                for s in state:
                    s.detach_()

        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long())

        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            updater(batch_size=1)

        metric.add(l * y.numel(), y.numel())

    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()


def train_ch8(net, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False, use_plot=True, loss_fn=None,
              verbose=True):
    """训练 RNN 模型（第8章风格）。

    Args:
        net:             模型
        train_iter:      训练数据迭代器
        vocab:           词表
        lr:              学习率
        num_epochs:      迭代周期数
        device:          设备
        use_random_iter: 是否使用随机采样
        use_plot:        是否使用 Animator 绘图（CLI 环境建议 False）
        loss_fn:         可选，自定义损失函数（签名 loss(logits, labels) → 标量）
                          默认 nn.CrossEntropyLoss()
        verbose:         是否打印中间训练信息（每 10 epoch 的预测文本和困惑度）。
                          设为 False 时仅输出最终结果，避免在参考答案等场景产生冗余输出。
                          默认 True，保持原有行为不变。
    """
    loss = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()
    animator = None
    if use_plot:
        animator = Animator(xlabel="epoch", ylabel="perplexity",
                            legend=["train"], xlim=[10, num_epochs])

    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: sgd(net.params, lr, batch_size)

    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)

    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter)
        # 仅在 verbose=True 时打印中间训练进度；verbose=False 可避免参考答案等场景的冗余输出
        if verbose and (epoch + 1) % 10 == 0:
            print(predict("time traveller"))
            if animator is not None:
                animator.add(epoch + 1, [ppl])
            else:
                print(f"  [epoch {epoch+1}/{num_epochs}] perplexity={ppl:.1f}")

    # 最终结果始终打印
    print(f"困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}")
    print(predict("time traveller"))
    print(predict("traveller"))


# =========================================================================
# 多 GPU 训练（第13章风格）
# =========================================================================


def train_batch_ch13(net, X, y, loss, trainer, devices):
    """使用多 GPU 训练一个小批量（第13章）。"""
    if isinstance(X, list):
        # BERT 微调时会用到
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = accuracy(pred, y)
    return train_loss_sum, train_acc_sum


def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
               devices=None):
    """使用多 GPU 训练模型（第13章）。"""
    if devices is None:
        devices = try_all_gpus()
    timer, num_batches = Timer(), len(train_iter)
    animator = Animator(xlabel="epoch", xlim=[1, num_epochs], ylim=[0, 1],
                        legend=["train loss", "train acc", "test acc"])
    # NPU 不支持 nn.DataParallel，统一采用单设备训练
    net = net.to(devices[0])
    for epoch in range(num_epochs):
        # 训练损失之和、训练准确率之和、样本数、预测数
        metric = Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_ch13(
                net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3],
                              None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f"loss {metric[0] / metric[2]:.3f}, train acc "
          f"{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}")
    print(f"{metric[2] * num_epochs / timer.sum():.1f} examples/sec on "
          f"{str(devices)}")


# =========================================================================
# 情感分析预测
# =========================================================================


def predict_sentiment(net, vocab, sequence):
    """预测文本序列的情感。"""
    sequence = torch.tensor(vocab[sequence.split()], device=try_gpu())
    label = torch.argmax(net(sequence.reshape(1, -1)), dim=1)
    return 'positive' if label == 1 else 'negative'
