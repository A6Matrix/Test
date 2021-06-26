from datetime import timedelta
from Student_train import *
from itertools import chain
from torch.optim import Adam
from Bert_train import *
from models.Bert import *
from models.TextCNN import textcnn
from sklearn.metrics import classification_report
#from NTM_Utils.NTM_data_loader import load_data_and_vocab
# from DistillModel_Utils.Student_Data_Utils import load_bow_dictionary
import time
import torch
import os
from Config import opt

import gensim
# import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def load_embeddings(embdding_path):
    model = gensim.models.KeyedVectors.load_word2vec_format(embdding_path, binary=False)
    embedding_matrix = np.zeros((len(model.index2word) + 1, model.vector_size))
    #填充向量矩阵
    for idx, word in enumerate(model.index2word):
        embedding_matrix[idx + 1] = model[word]#词向量矩阵
    return embedding_matrix

def load_eng_embed(embedding_path):
    fr = open(embedding_path,'r',encoding='utf-8')
    embedding_matrix = np.zeros((len(fr.readlines()) + 1, 300))
    # 填充向量矩阵
    for idx,line in enumerate(fr.readlines()):
        values = line.split()
        embedding_matrix[idx+1] = np.asarray(values[1:], "float32")
    return embedding_matrix

if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    # _,bow_dictionary = load_bow_dictionary()
    embeddings = load_embeddings(opt.embeddings_file)
    #embeddings = load_eng_embed(opt.eng_embed_file)
    if opt.train_teacher:
        bert_train(opt.target_dir)
        bert_test(opt.en_pretrained_file)

    if opt.train_student:
        # ntm_model = NTM().to(device)
        stu_model = textcnn(embeddings=embeddings).to(device)
        bert_model = BertModel().to(device)
        total_params = sum(p.numel() for p in stu_model.parameters())
        print("model parameter:%.2fM"%(total_params/1e6))
        # optimizer_conv, optimizer_ntm, optimizer_whole = init_optimizers(stu_model, ntm_model)
        # optimizer_conv = init_optimizers(stu_model)
        # train_bow_loader, valid_bow_loader = load_data_and_vocab()
        # train_model(ntm_model, bert_model, stu_model, embeddings,optimizer_ntm, optimizer_whole, bow_dictionary, train_bow_loader, valid_bow_loader)
        train_model(bert_model, stu_model, embeddings)

    if opt.test_student:
        # ntm_model = NTM().to(device)
        # test_bow_loader = load_data_and_vocab(load_train=False)
        # s_predict = student_predict(ntm_model)
        s_predict = student_predict(embeddings)
       # bert_test(opt.pretrained_file)



