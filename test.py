# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 17:40:37 2019
Module: preprocessing data
@author: daijun.chen
"""

# import modules
import os
import re
import nltk
import json
import numpy as np
import pandas as pd
import gensim as gensim
import logging


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

work_dir = '..'


# clean sentence: lower case, good/bad --> good / bad, nltk tokenizer 
def clean_sentence(sentence):
    sentence = str(sentence).lower()        # convert to lower case
    sentence = re.sub(' - ', '-', sentence) # refit dashes (single words)

    # clean punctuation
    for p in '/+-^*÷#!"(),.:;<=>?@[\]_`{|}~\'¿€$%&£±': 
        sentence = sentence.replace(p, ' ' + p + ' ') # good/bad --> good / bad

    sentence = sentence.strip()        # strip leading and trailing white space
    tokenized_sentence = nltk.tokenize.word_tokenize(sentence) # nltk tokenizer
    #tokenized_sentence = [w for w in tokenized_sentence if w in printable] # remove non ascii characters

    return tokenized_sentence


# load word2vec embedding matrix / dict / inverse dict
def load_w2v(w2v_dim=300):
    '''load word2vec'''
    saving_path = work_dir + '/w2v/'+'embeddings_'+str(w2v_dim)+'d.p'
    vocab_path = work_dir + '/w2v/'+'vocab_'+str(w2v_dim)+'d.p'

    if not os.path.exists(saving_path):
        print("\n Creating word_embeddings...")

        df = pd.read_csv(work_dir+"/data/duplicate_questions.tsv", delimiter='\t') # load data
        df1 = df[['question1']].rename(index=str, columns={"question1": "question"}) # questions 1
        df2 = df[['question2']].rename(index=str, columns={"question2": "question"}) # questions 2

        unique_questions = pd.concat([df1,df2]).question.unique() # unique questions
        corpus = list(unique_questions)
        print('Collected', len(corpus), 'unique sentences (questions).')

        corpus = list(map(clean_sentence, corpus))            # preprocess text [clean_sent(sent) for sent in corpus]
        corpus.append(['UNK','UNK', 'UNK', 'UNK', 'UNK']) # unknown word
        corpus.append(['EOS','EOS', 'EOS', 'EOS', 'EOS']) # padding

        my_model = gensim.models.word2vec.Word2Vec(size=w2v_dim, min_count=2, sg=1) # initialize W2V model: collect 87 116 word types from a corpus of 7 161 626 (+10) raw words 
        my_model.build_vocab(corpus) # 48 096 (+2) not unique words (min_count=2). --> 55% of word types / 99% of raw words
        my_model.intersect_word2vec_format(work_dir + '/w2v/' + 'glove.6B.' + str(w2v_dim) + 'd.txt', binary=False) # update with GloVe: 44 373 retrieved (84%)
        
        weights = my_model.wv.syn0     # word embeddings
        np.save(open(saving_path, 'wb'), weights)

        vocab = dict([(k, v.index) for k, v in my_model.wv.vocab.items()]) # dictionary
        with open(vocab_path, 'w') as f:
            f.write(json.dumps(vocab))
            
    # load word embedding as weights matrix
    with open(saving_path, 'rb') as f:
        weights = np.load(f)
    print('\n Loaded Word_embeddings Matrix', weights.shape)

    # load word embedding vocabulary (json file)
    with open(vocab_path, 'r') as f:
        data = json.loads(f.read())

    word2idx = data # from word to index dict
    idx2word = dict([(v, k) for k, v in data.items()]) # from index to word dict
    print('Loaded Vocabulary Mapping (size {})'.format(len(word2idx)))

    return weights, word2idx, idx2word


# create/load word embedding, word2index, index2word
weights, word2idx, idx2word = load_w2v()
pad_tok = word2idx['EOS'] # pad token


# map tokenized sentence to words ids by word2idx
def sentence2ids(tokenized_sentence):
    sentence_ids = []

    for word in tokenized_sentence:
        try:
            sentence_ids.append(word2idx[word])
        except:
            sentence_ids.append(word2idx['UNK']) # Unknown words

    return np.asarray(sentence_ids)


# load questions to dictionary with (key) question id and (value) word index list 
def load_all(maxlen=40):
    folder = work_dir + '/corpus/'
    saving_path = 'Question_id.npy'
    
    if not os.path.exists(folder):
        os.makedirs(folder)

    if not os.path.exists(folder + saving_path):
        print('\n Creating word id corpus (all)...')
        df = pd.read_csv(work_dir + "/data/duplicate_questions.tsv", delimiter='\t') # load UNLABELED data
        df1 = df[['question1']].rename(index=str, columns={"question1": "question"})
        df2 = df[['question2']].rename(index=str, columns={"question2": "question"})

        unique_questions = pd.concat([df1,df2]).question.unique()              # unique questions
        print('\n Loaded {} unique questions (corpus)'.format(len(unique_questions)))

        corpus = list(unique_questions)
        corpus = list(map(clean_sentence, corpus))     # preprocess text

        corpus_ids = list(map(sentence2ids, corpus)) # map to wid 
        corpus_ids = dict(zip(np.arange(len(corpus_ids)), corpus_ids))  # {0:w0ids, 1:w1ids, 2:w2ids}

        np.save(folder + saving_path, corpus_ids)

    corpus_ids = np.load(folder + saving_path).item()  # A question = a list of word_index

    print("\n Loaded word id corpus (all)")

    return corpus_ids


# load questions from duplicate_question.tsv
corpus_ids = load_all()


# load train/dev/test questions to dictionary with (key) question id and (value) word index list 
def load_split(name='train', maxlen=40):

    folder = work_dir + '/corpus/'
    saving_path = '_' + name + '_id.npy'

    if not os.path.exists(folder + 'Xs' + saving_path):
        print('\n Creating wid corpus ({})...'.format(name))
        # load train.csv/dev.csv/test.csv 
        df = pd.read_csv(work_dir + "/data/split/" + name + ".csv", sep=',') # load LABELED data
        df = df[['question1', 'question2', 'is_duplicate']]
        print('\n Loading ' + name + ' dataset (', len(df), 'question pairs)')

        # similar question pair
        df_true_duplicate = df[df['is_duplicate']==1]        # duplicate questions
        Xs = list(df_true_duplicate['question1'].values)     # Xs[k] and Ys[k] are duplicates
        Ys = list(df_true_duplicate['question2'].values)

        print('\n Duplicate questions pairs: (', len(Xs), 'pairs):')
        print(Xs[0], '\n', Ys[0])             # Show the 1st similar pair

        Xs = list(map(clean_sentence, Xs))    # preprocess text
        Ys = list(map(clean_sentence, Ys))

        Xs_ids = list(map(sentence2ids, Xs))      # map sentence to word ids
        Ys_ids = list(map(sentence2ids, Ys))

        Xs_ids = dict(zip(np.arange(len(Xs_ids)), Xs_ids))  # {0: q0_wids, 1: q1_wids}
        Ys_ids = dict(zip(np.arange(len(Ys_ids)), Ys_ids))  

        np.save(folder + 'Xs' + saving_path, Xs_ids)        # question1: Xs_train_id.npy
        np.save(folder + 'Ys' + saving_path, Ys_ids)        # question2: Ys_train_id.npy

        # dissimilar question pair
        df_false_duplicate = df[df['is_duplicate']==0]      # dissimilar questions
        Xa = list(df_false_duplicate['question1'].values)   # Xa[k] and Ya[k] are NOT duplicates
        Ya = list(df_false_duplicate['question2'].values)

        print('\n Not duplicated questions pairs: (',len(Xa),'pairs):')
        print(Xa[0], '\n', Ya[0])

        Xa = list(map(clean_sentence, Xa))      # preprocess text
        Ya = list(map(clean_sentence, Ya))

        Xa_ids = list(map(sentence2ids, Xa))      # map sentence to word ids
        Ya_ids = list(map(sentence2ids, Ya))

        Xa_ids = dict(zip(np.arange(len(Xa_ids)), Xa_ids))   # {0: q0_wids, 1: q1_wids}
        Ya_ids = dict(zip(np.arange(len(Ya_ids)), Ya_ids))

        np.save(folder + 'Xa' + saving_path, Xa_ids)      # question1: Xa_train_id.npy
        np.save(folder + 'Ya' + saving_path, Ya_ids)      # question2: Ya_train_id.npy

    # If the Xa_train_id.npy/Xs_train_id.npy and Ya_train_id.npy/Ys_train_id.npy are prepared
    Xs_ids = np.load(folder + 'Xs' + saving_path).item()
    Ys_ids = np.load(folder + 'Ys' + saving_path).item()
    Xa_ids = np.load(folder + 'Xa' + saving_path).item()
    Ya_ids = np.load(folder + 'Ya' + saving_path).item()
    print("\n Loaded word ids corpus ({})".format(name))

    return Xs_ids, Ys_ids, Xa_ids, Ya_ids


# load/generate Xa_train_id.npy/Xs_train_id.npy and Ya_train_id.npy/Ys_train_id.npy
Xs_train_ids, Ys_train_ids, Xa_train_ids, Ya_train_ids = load_split(name='train') # train
Xs_dev_ids, Ys_dev_ids, Xa_dev_ids, Ya_dev_ids = load_split(name='dev')           # dev
Xs_test_ids, Ys_test_ids, Xa_test_ids, Ya_test_ids = load_split(name='test')      # test


# load questions to dictionary with (key) question id and (value) word index list 
def load_4rank(name='dev', maxlen=40):
    folder = work_dir + '/corpus/'
    saving_path = 'rank_' + name + '_id.npy'

    if not os.path.exists(folder + saving_path):
        print('\n Creating ranking word id corpus (' + name + ')')
        if name == 'train&dev':
            df_train = pd.read_csv(work_dir + "/data/split/train_rank.csv", sep=',')
            df1_train = df_train[['question', 'qid']]
            df_dev = pd.read_csv(work_dir + '/data/split/dev_rank.csv', sep=',')
            df1_dev = df_dev[['question', 'qid']]
            unique_questions = pd.concat([df1_train, df1_dev])#.question.unique()
            print('\n Loaded {} unique {} questions for rank'.format(len(unique_questions), name))
        elif name == 'train&test':
            df_train = pd.read_csv(work_dir + "/data/split/train_rank.csv", sep=',')
            df1_train = df_train[['question', 'qid']]
            df_test = pd.read_csv(work_dir + '/data/split/test_rank.csv', sep=',')
            df1_test = df_test[['question', 'qid']]
            unique_questions = pd.concat([df1_train, df1_test])#.question.unique()
            print('\n Loaded {} unique {} questions for rank'.format(len(unique_questions), name))            
        elif name == 'train&dev&test':
            df_train = pd.read_csv(work_dir + '/data/split/train_rank.csv', sep=',')
            df1_train = df_train[['question', 'qid']]
            df_dev  = pd.read_csv(work_dir + '/data/split/dev_rank.csv', sep=',')
            df1_dev = df_dev[['question', 'qid']]
            df_test = pd.read_csv(work_dir + '/data/split/test_rank.csv', sep=',')
            df1_test = df_test[['question', 'qid']]
            unique_questions = pd.concat([df1_train, df1_dev, df1_test])
            print('\n Loaded {} unique {} quesitons for rank'.format(len(unique_questions), name))
        else:    
            df = pd.read_csv(work_dir + "/data/split/"+name+"_rank.csv", sep=',') # load UNLABELED data
            df1 = df[['question', 'qid']]
            unique_questions = df1 #.question.unique()   # unique questions
            print('\n Loaded {} unique {} questions for rank'.format(len(unique_questions), name))

        corpus = list(unique_questions.question) 
        corpus_qinit = corpus                          # get initial question 
        corpus = list(map(clean_sentence, corpus))     # preprocess text
        corpus_ids = list(map(sentence2ids, corpus))   # map to word id

        corpus_qid = list(unique_questions.qid)

        corpus_combine = [{'question':corpus_ids[i], 'qid':corpus_qid[i], 'qinit':corpus_qinit[i]} for i in range(len(corpus_ids))]
        corpus_ids = dict(zip(np.arange(len(corpus_ids)), corpus_combine))  # {0:w0ids, 1:w1ids, 2:w2ids}
        
        np.save(folder + saving_path, corpus_ids)

    corpus_ids = np.load(folder + saving_path).item()  # A question = a list of word_index
    print("\n Loaded word id corpus (" + name + ") for rank")

    return corpus_ids

# load word ids from train/dev/test data
Rank_traindevtest_ids = load_4rank(name='train&dev&test')
Rank_traindev_ids = load_4rank(name='train&dev')
Rank_traintest_ids = load_4rank(name='train&test')
Rank_train_ids = load_4rank(name='train')
Rank_dev_ids = load_4rank(name='dev')
Rank_test_ids = load_4rank(name='test')


# pad sequence: ids[:padlen], min(len(ids), padlen) 
def pad_sequence(ids, padlen=40, target_offset=False):
    if target_offset == False:
        return ids[:padlen] + [pad_tok] * max(padlen - len(ids), 0), min(len(ids), padlen)
    else:
        return [pad_tok] + ids[:padlen-1] + [pad_tok] * max(padlen-1 - len(ids), 0), min(len(ids), padlen) # shift decoder's input


# load VAD/VAE (repeat/reformulate) padded sequence 
def load_VAD_corpus(corpus_ids, Xs_ids, Ys_ids, mode='VAE', padlen=40):            ##### keep questions / len(q) <= maxlen (repeat, reformulate)
    d1 = {}
    d2 = {}
    
    n1 = len(corpus_ids)   # number of unique questions 
    n2 = 0

    # VAE: non-duplicate (repeat) case
    for i in range(n1):
        d1[i] = pad_sequence(list(corpus_ids[i]), padlen=padlen, target_offset=False) 
        d2[i] = pad_sequence(list(corpus_ids[i]), padlen=padlen, target_offset=True)   # REPEAT (VAE): q -> q

    # VAD: duplicate case
    if mode == 'VAD':    
        n2 += len(Xs_ids)
        
    for i in range(n2):
        d1[n1+i] = pad_sequence(list(Xs_ids[i]), padlen=padlen, target_offset=False)
        d2[n1+i] = pad_sequence(list(Ys_ids[i]), padlen=padlen, target_offset=True)    # REFORMULATE (VAD) q -> q'
        
    for i in range(n2):
        d1[n1+n2+i] = pad_sequence(list(Ys_ids[i]), padlen=padlen, target_offset=False)
        d2[n1+n2+i] = pad_sequence(list(Xs_ids[i]), padlen=padlen, target_offset=True) # REFORMULATE (VAD) q' -> q

    print('\n Loaded generative corpus: {} sentence pairs ({} VAE[repeat] / {} VAD[reformulate])'.format(n1+2*n2, n1, 2*n2))

    return d1, d2, n1+2*n2    # d1:q1, d2:q2, total repeat/reformulate pairs


# load classification q1, q2, is_duplicate, total question pair number
def load_CLF_corpus(Xs_ids, Ys_ids, Xa_ids, Ya_ids, padlen=40):

    n1 = len(Xs_ids)
    n2 = len(Xa_ids)
    d1, d2, d3 = {}, {}, {}

    for i in range(n1):     # Duplicate
        d1[i] = pad_sequence(list(Xs_ids[i]), padlen=padlen)  # tuple (tokens, seq_length)
        d2[i] = pad_sequence(list(Ys_ids[i]), padlen=padlen)  
        d3[i] = 1

    for i in range(n2):     # Not Duplicate
        d1[n1+i] = pad_sequence(list(Xa_ids[i]), padlen=padlen) 
        d2[n1+i] = pad_sequence(list(Ya_ids[i]), padlen=padlen) 
        d3[n1+i] = 0

    print('\n Loaded discriminative corpus: {} sentence pairs ({} duplicate / {} non duplicate)'.format(n1+n2, n1, n2))

    return d1, d2, d3, n1+n2 # CLF corpus (q1, q2, is_duplicate, total pair number)


# create index batches w/wo shuffle
def create_batches(data_size, batch_size=64, shuffle=True):
    batches = []       # create batches by index
    ids = np.arange(data_size)

    # create batchese by shuffling 
    if shuffle:
        np.random.shuffle(np.asarray(ids))

    for i in range(np.floor(data_size / batch_size).astype(int)):
        start = i * batch_size
        end = (i + 1) * batch_size
        batches.append(ids[start:end])

    return batches     # batch by indices


# obtain data from ids for VAD/VAE 
def fetch_data_ids_VAD(d1, d2, idx_batch, padlen=40):
    batch_size = len(idx_batch)        # single batch
    q1 = np.zeros([batch_size, padlen])
    q2 = np.zeros([batch_size, padlen])
    q1_len = np.zeros([batch_size])
    q2_len = np.zeros([batch_size])

    for i, idx in enumerate(idx_batch):
        q1[i], q1_len[i] = d1[idx]     # padded sequence target_offset=False
        q2[i], q2_len[i] = d2[idx]     # padded sequence target_offset=True

    return q1, q1_len, q2, q2_len      # q1, q' in one batches


# obtain data from ids/labels fro classification (CLF)
def fetch_data_ids_CLF(d1, d2, d3, idx_batch, padlen=40):
    batch_size = len(idx_batch)
    q1 = np.zeros([batch_size, padlen])
    q2 = np.zeros([batch_size, padlen])
    q1_len = np.zeros([batch_size])
    q2_len = np.zeros([batch_size])
    labels = np.zeros([batch_size])

    for i, idx in enumerate(idx_batch):
        q1[i], q1_len[i] = d1[idx]
        q2[i], q2_len[i] = d2[idx]
        labels[i] = d3[idx]

    q_stack = np.concatenate((q1, q2), axis=0)
    q_len_stack = np.concatenate((q1_len, q2_len), axis=0)

    return q_stack, q_len_stack, labels # q1, q2, label


# Main function part 
#if __name__ == '__main__':
    # define parameters
#    batch_size = 64
#    padlen= 40
#    
#    # Show word2vec, debugging
#    print(word2idx['linear'], word2idx['algebra'])
#    print(idx2word[2619], idx2word[3718])
#    
#    # Show corpus ids and index2word
#    print(corpus_ids[0])
#    print([idx2word[i] for i in corpus_ids[0]])
#
#    # VAD corpus
#    d1, d2, num_repeat_reformulate = load_VAD_corpus(corpus_ids, Xs_train_ids, Ys_train_ids, mode='VAD', padlen=padlen)
#    batches = create_batches(num_repeat_reformulate, batch_size=batch_size)
#
#    q1, q1_len, q2, q2_len = fetch_data_ids_VAD(d1, d2, batches[0], padlen=padlen)
#    print([idx2word[i] for i in q1[0]]) # q1 in batches[0]
#    print([idx2word[i] for i in q2[0]]) # q2 in batches[0]
#    print(q1_len[0], q2_len[0])         # lenght of q1 & q2
#
#    # CLF corpus
#    d1, d2, d3, num_question_pairs = load_CLF_corpus(Xs_dev_ids, Ys_dev_ids, Xa_dev_ids, Ya_dev_ids, padlen=padlen)
#    batches = create_batches(num_question_pairs, batch_size=batch_size)
#    q_stack, q_len_stack, labels = fetch_data_ids_CLF(d1, d2, d3, batches[0], padlen=padlen)
