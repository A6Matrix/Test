import torch
import torch.nn
from config import args
# from DataUtils import load_vocab
fr_chn = open('./data/w2v.txt','r',encoding='utf-8',errors='ignore')
fr_eng = open('./data/eng_w2v.txt','r',encoding='utf-8')
# word2idx , _ = load_vocab('./open_data/content_vocab.txt')
def load_vocab(vocab_path):
    fr = open(vocab_path, 'r', encoding='utf-8')
    vocab = []
    for line in fr.readlines():
        data = line.strip().split()
        vocab.append(data[0])
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}

    return word2idx, idx2word

def get_weight(vocab_path):
    word2idx, _ = load_vocab(vocab_path)
    weight = torch.randn(len(word2idx), 300)
    for line in fr_chn.readlines():
        line = eval(line)
        key = [key for key, value in line.items()][0]#得到预训练好的词向量的键
        id = word2idx.get(key)#判断当前词是否在词表中
        if id is not None:
            value = torch.FloatTensor(list(line.values())[0])#line.values得到的类型是dict.values，而且是二维列表，所以需要转换为列表取出第一个
            weight[id] = value#将初始化的词向量替换为预训练好的
    return weight

def get_eng_weight(vocab_path):
    word2idx, _ = load_vocab(vocab_path)
    weight = torch.randn(300000, 300)
    for line in fr_eng.readlines():
        line = eval(line)
        key = [key for key, value in line.items()][0]  # 得到预训练好的词向量的键
        id = word2idx.get(key)  # 判断当前词是否在词表中
        if id is not None:
            value = torch.FloatTensor(list(line.values())[0])  # line.values得到的类型是dict.values，而且是二维列表，所以需要转换为列表取出第一个
            weight[id] = value  # 将初始化的词向量替换为预训练好的
    return weight
