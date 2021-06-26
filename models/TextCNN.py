# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Config import opt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# class Config(object):
#
#     """配置参数"""
#     def __init__(self, dataset, embedding):
#         self.model_name = 'TextCNN'
#         self.train_path = dataset + '/data/train.txt'                                # 训练集
#         self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
#         self.test_path = dataset + '/data/test.txt'                                  # 测试集
#         self.class_list = [x.strip() for x in open(
#             dataset + '/data/class.txt').readlines()]                                # 类别名单
#         self.vocab_path = dataset + '/data/vocab.pkl'                                # 词表
#         self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
#         self.log_path = dataset + '/log/' + self.model_name
#         self.embedding_pretrained = torch.tensor(
#             np.load(dataset + '/data/' + embedding)["embeddings"].astype('float32'))\
#             if embedding != 'random' else None                                       # 预训练词向量
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
#
#         self.dropout = 0.5                                              # 随机失活
#         self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
#         self.num_classes = len(self.class_list)                         # 类别数
#         self.n_vocab = 0                                                # 词表大小，在运行时赋值
#         self.num_epochs = 20                                            # epoch数
#         self.batch_size = 128                                           # mini-batch大小
#         self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
#         self.learning_rate = 1e-3                                       # 学习率
#         self.embed = self.embedding_pretrained.size(1)\
#             if self.embedding_pretrained is not None else 300           # 字向量维度
#         self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
#         self.num_filters = 256                                          # 卷积核数量(channels数)
#

'''Convolutional Neural Networks for Sentence Classification'''


class textcnn(nn.Module):
    def __init__(self, embeddings, nums_label=2,dropout=0.2):
        super(textcnn, self).__init__()
        self.filter_sizes = (2, 3, 4)
        self.embeds_dim = embeddings.shape[1]
        self.embed = nn.Embedding(embeddings.shape[0], embeddings.shape[1])
        self.embed.weight = nn.Parameter(torch.from_numpy(embeddings))
        self.embed.float()
        self.embed.to(device)
        self.embed.weight.requires_grad = False
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, opt.num_filters, (k, self.embeds_dim)) for k in self.filter_sizes])
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(opt.num_filters * len(self.filter_sizes)*4, opt.num_filters * len(self.filter_sizes))
        self.fc = nn.Linear(opt.num_filters * len(self.filter_sizes), nums_label)
        self.relu = nn.ReLU(inplace=True)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, q1, q2):
        q1 = self.embed(q1)
        q2 = self.embed(q2)
        q1_out = q1.unsqueeze(1)
        q2_out = q2.unsqueeze(1)
        q1_out = torch.cat([self.conv_and_pool(q1_out, conv) for conv in self.convs], 1)#bz*768
        q2_out = torch.cat([self.conv_and_pool(q2_out, conv) for conv in self.convs], 1)
        q1_out = self.dropout(q1_out)
        q2_out = self.dropout(q2_out)
        combined = torch.cat([q1_out,q2_out,q1_out-q2_out,q1_out*q2_out],dim=-1)
        projected = self.linear(combined)
        out = self.relu(projected)
        out = self.dropout(out)
        logit = self.fc(out)
        probility = F.softmax(logit,dim=-1)
        return logit,probility
