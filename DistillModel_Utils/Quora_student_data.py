# -*- coding: utf-8 -*-

import re
import numpy as np
import torch
from torch.utils.data import Dataset
from Config import opt


def get_word_list(text):
    text = text.lower()
    # 删除（）里的内容
    text = re.sub('[,?"#();:!…？“”]', '', text)
    text = re.sub('[/]', ' ', text)
    text = re.sub(r"can't", 'can not', text)
    text = re.sub(r"n't", ' not', text)
    text = re.sub(r"'ve", " 've", text)
    text = re.sub(r"'d", " 'd", text)
    text = re.sub(r"'ll", " 'll", text)
    text = re.sub(r"'m", " 'm", text)
    text = re.sub(r"'s", " 's", text)

    return [word for word in text.split()]

def build_dataset(p, h, label):
    datasets = []
    for i in range(len(p)):
        dataset = {}
        dataset['texta'] = p[i]
        dataset['textb'] = h[i]
        dataset['label'] = label[i]
        datasets.append(dataset)
    return datasets

class Quora_Dataset(Dataset):
    def __init__(self, tag="train", max_char_len=150):
        # word2idx= load_bow_dictionary()
        self.p_list, self.h_list, self.label = load_sentences(tag)
        self.datasets = build_dataset(self.p_list,self.h_list,self.label)
        self.max_length = max_char_len

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.datasets[idx]


    def collate_fn_data(self, batches):
        '''
        Puts each data field into a tensor with outer dimension batch size"
        '''
        texta = [b['texta'] for b in batches]
        textb = [b['textb'] for b in batches]
        label = [b['label'] for b in batches]
        texta = pad_sequences(texta, maxlen=self.max_length)
        textb = pad_sequences(textb, maxlen=self.max_length)
        texta = torch.from_numpy(texta).type(torch.long)
        textb = torch.from_numpy(textb).type(torch.long)
        label = torch.Tensor(label).type(torch.long)
        return texta, textb, label

# 加载word_index训练数据
def load_sentences(tag):
    if tag == "train":
        src_file = opt.quora_train_file
    elif tag == "valid":
        src_file = opt.quora_dev_file
    else:
        src_file = opt.quora_test_file
    sentences_1, sentences_2, labels = [], [], []
    for line in open(src_file, 'r', encoding='utf-8').readlines():
        line = line.strip().split('\t')
        if len(line) != 3:
            continue
        sentences_1.append(line[0])
        sentences_2.append(line[1])
        labels.append(int(line[2]))
    sentences_1 = map(get_word_list, sentences_1)
    sentences_2 = map(get_word_list, sentences_2)
    word2idx, _, _ = load_vocab(opt.quora_vocab_file)
    sent_1, sent_2 = word_index(sentences_1,sentences_2, word2idx)


    return sent_1, sent_2, labels

# word->index
def word_index(p_sentences, h_sentences, word2idx):
    p_list, h_list = [], []
    for p_sentence, h_sentence in zip(p_sentences, h_sentences):
        p = [word2idx[word] for word in p_sentence if word in word2idx.keys()]
        h = [word2idx[word] for word in h_sentence if word in word2idx.keys()]
        p_list.append(p)
        h_list.append(h)
    return p_list,  h_list


# 加载字典
def load_vocab(vocab_file):
    vocab = [line.strip() for line in open(vocab_file, encoding='utf-8').readlines()]
    word2idx = {word: index for index, word in enumerate(vocab)}
    idx2word = {index: word for index, word in enumerate(vocab)}
    return word2idx, idx2word, vocab


''' 把句子按字分开，中文按字分，英文数字按空格, 大写转小写，繁体转简体'''

def pad_sequences(sequences, maxlen=None, dtype='int32', padding='post',
                  truncating='post', value=0.):
    """ pad_sequences
    把序列长度转变为一样长的，如果设置了maxlen则长度统一为maxlen，如果没有设置则默认取
    最大的长度。填充和截取包括两种方法，post与pre，post指从尾部开始处理，pre指从头部
    开始处理，默认都是从尾部开始。
    Arguments:
        sequences: 序列
        maxlen: int 最大长度
        dtype: 转变后的数据类型
        padding: 填充方法'pre' or 'post'
        truncating: 截取方法'pre' or 'post'
        value: float 填充的值
    Returns:
        x: numpy array 填充后的序列维度为 (number_of_sequences, maxlen)
    """
    lengths = [len(s) for s in sequences]
    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)
    x = (np.ones((nb_samples, maxlen)) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError("Truncating type '%s' not understood" % padding)
        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError("Padding type '%s' not understood" % padding)
    return x
