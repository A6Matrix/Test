import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-train_file',type=str, default='./data/LCQMC/train.txt')
parser.add_argument('-quora_train_file',type=str, default='./data/Quora/quora_train.txt')
parser.add_argument('-dev_file',type=str, default='./data/LCQMC/dev.txt')
parser.add_argument('-quora_dev_file',type=str, default='./data/Quora/quora_dev.txt')
parser.add_argument('-test_file',type=str,default='./data/Test.txt')
parser.add_argument('-quora_test_file',type=str,default='./data/Quora/quora_test.txt')
parser.add_argument('-vocab_file',type=str,default='./bert_pretrain/vocab.txt')
parser.add_argument('-quora_vocab_file',type=str,default='./bert_base_cased/vocab.txt')
parser.add_argument('-target_dir', type=str, default='./data/Bert_Models')
parser.add_argument('-pretrained_file', type=str, default='./data/Bert_Models/best.pth.tar')
parser.add_argument('-en_pretrained_file', type=str, default='./data/Bert_Models/quora_best.pth.tar')
parser.add_argument('-embeddings_file', type=str, default='./data/sgns.wiki.word')
parser.add_argument('-eng_embed_file', type=str, default='./data/eng_w2v.txt')
parser.add_argument('-max_length',type=int,default=50)
parser.add_argument('-max_src_len', type=int, default=100,help="Max length of the source sequence")
parser.add_argument('-batch_size',type=int,default=32)
parser.add_argument('-epoch',type=int,default=20)
parser.add_argument('-vocab_size',type=int,default=50000)
parser.add_argument('-num_filters',type=int,default=256)
parser.add_argument('-learning_rate', type=float, default=1e-3)
parser.add_argument('-train_teacher',action='store_true',default=False)
parser.add_argument('-train_student',action='store_true',default=False)
parser.add_argument('-test_student', action='store_true',default=True)




opt = parser.parse_args()


