import jieba
import random
import re
import torch
from torch.utils.data import Dataset
import torch.nn.utils.rnn as rnn_utils

def load_vocab(vocab_path):
    fr = open(vocab_path, 'r', encoding='utf-8')
    vocab = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
    for line in fr.readlines():
        data = line.strip().split()
        vocab.append(data[0])
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}

    return word2idx, idx2word

class Data_prepare():
    def readfile(self, filename):
        fr = open(filename,'r',encoding='utf-8')
        texta = []
        textb = []
        tag = []
        for line in fr.readlines():
            line = line.strip().split("\t")
            texta.append(self.clear(line[0]))
            textb.append(self.clear(line[1]))
            tag.append(line[2])
            # shuffle
        index = [x for x in range(len(texta))]
        random.shuffle(index)
        texta_new = [texta[x] for x in index]
        textb_new = [textb[x] for x in index]
        tag_new = [tag[x] for x in index]
        return texta_new,textb_new,tag_new

    def clear(self,sentence):
        # 去除标点符号和特殊符号
        # sentence = re.sub("[:：+——()?【】《》“”！，；。？、~@￥%……&*（）]+", "", sentence)
        sentence = re.sub('（[^（.]*）', "", sentence)
        sentence = ''.join([x for x in sentence if '\u4e00' <= x <= '\u9fa5'])
        return sentence



    def process(self,filename,word2idx):
        texta_list = []
        textb_list = []
        text_a,text_b,tags = self.readfile(filename)
        for line_a, line_b in zip(text_a, text_b):
            texta_tok = list(jieba.cut(line_a))
            texta_id = [word2idx.get(word, 0) for word in texta_tok]
            texta_list.append(texta_id)

            textb_tok = list(jieba.cut(line_b))
            textb_id = [word2idx.get(word, 0) for word in textb_tok]
            textb_list.append(textb_id)

            type = list(set(tags))
            dicts = {}
            tags_vec = []
            for x in tags:
                if x not in dicts.keys():
                    dicts[x] = 1
                else:
                    dicts[x] += 1
                temp = [0] * len(type)
                temp[int(x)] = 1
                tags_vec.append(temp)
            # print(dicts)
            return texta_list, textb_list, tags

class batch_data(Dataset):
    #1. __len__ 使用``len(dataset)`` 可以返回数据集的大小
    # 2. __getitem__ 支持索引, 以便于使用 dataset[i] 可以 获取第i个样本(0索引)
    def __init__(self,texta_input,textb_input,tag):
        self.texta_input = texta_input
        self.textb_input = textb_input
        self.tag = tag

    def __len__(self):
        return len(self.texta_input)

    def __getitem__(self, i):
        return self.texta_input[i],self.textb_input[i],self.tag[i]

def pad_collate_fn(insts):
    text_a,text_b,tag = list(*zip(insts))
    texta_input = collate_fn(text_a)
    textb_input = collate_fn(text_b)
    tag_input = torch.Tensor(tag)
    return (texta_input,textb_input,tag_input)

def collate_fn(insts):
    ''' Pad the instance to the max seq length in batch '''
    # insts.sort(key=lambda x:len(x),reversed=True)
    insts = [torch.Tensor(text) for text in insts]
    batch_insts = rnn_utils.pad_sequence(insts, batch_first=True,padding_value=0)
    return batch_insts
    # max_len = max(len(inst) for inst in insts)
    # max_len = 40
    # batch_seq = np.array([
    #     inst + [0] * (max_len - len(inst))
    #     for inst in insts])
    # batch_pos = np.array([
    #     [pos_i + 1 if w_i != 0 else 0
    #      for pos_i, w_i in enumerate(inst)] for inst in batch_seq])
    # batch_seq = torch.LongTensor(batch_seq)
    # batch_pos = torch.LongTensor(batch_pos)
    # return batch_seq,batch_pos