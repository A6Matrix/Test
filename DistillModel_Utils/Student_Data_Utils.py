# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 15:30:14 2020

@author: zhaog
"""
import re
import numpy as np
import torch
from hanziconv import HanziConv
from torch.utils.data import Dataset
from Config import opt
#from Data_preprocess import read_src_files
import jieba

# def load_bow_dictionary():
#     # load vocab
#     # logging.info("Loading vocab from disk: %s" % (opt.res_data_dir))
#     word2idx, _ = torch.load(opt.res_data_dir + '/vocab.pt', 'wb')
#     # assign vocab to opt
#     # opt.bow_dictionary = bow_dictionary
#     # logging.info('#(bow dictionary size)=%d' % len(bow_dictionary))
#
#     return word2idx

def get_word_list(query):
    query = HanziConv.toSimplified(query.strip())
    regEx = re.compile('[\\W]+')  # 我们可以使用正则表达式来切分句子，切分的规则是除单词，数字外的任意字符串
    res = re.compile(r'([\u4e00-\u9fa5])')  # [\u4e00-\u9fa5]中文范围
    sentences = regEx.split(query.lower())
    str_list = []
    for sentence in sentences:
        if res.split(sentence) == None:
            str_list.append(sentence)
        else:
            ret = res.split(sentence)
            str_list.extend(ret)
    return [w for w in str_list if len(w.strip()) > 0]

def build_dataset(p, h, label):
    datasets = []
    # tokenized_src = read_src_files(tag)
    for i in range(len(p)):
        dataset = {}
        dataset['texta'] = p[i]
        dataset['textb'] = h[i]
        dataset['label'] = label[i]
        datasets.append(dataset)
    return datasets

class LCQMC_Dataset(Dataset):
    def __init__(self, tag="train", max_char_len=50):

        self.p_list, self.h_list, self.label = load_sentences(tag,max_char_len)
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
        # src_bow = [b['src_bow'] for b in batches]
        # src_bow = self._pad_bow(src_bow)
        return texta, textb, label
# 加载word_index训练数据
def load_sentences(tag,max_length):
    if tag == "train":
        src_file = opt.train_file
    elif tag == "valid":
        src_file = opt.dev_file
    else:
        src_file = opt.test_file
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
    word2idx, _, _ = load_vocab(opt.vocab_file)
    sent_1, sent_2 = word_index(sentences_1,sentences_2,word2idx)


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
