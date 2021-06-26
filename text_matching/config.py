import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('-epoch', type=int, default=2)
parser.add_argument('-batch_size', type=int, default=1)
parser.add_argument('-embed_size', type=int, default=300)
parser.add_argument('-hidden_size', type=int, default=200)
parser.add_argument('-max_seq_len', type=int, default=512)
parser.add_argument('-n_head', type=int, default=8)
parser.add_argument('-n_warmup_steps', type=int, default=4000)
# parser.add_argument('-d_k', type=int, default=12) #d_model//n_head
parser.add_argument('-learning_rate', type=float, default=1e-3)
parser.add_argument('-dropout', type=float, default=0.1)
parser.add_argument('---cuda', action='store_true',default=True)

parser.add_argument('---save_model', type=str, default='./data/save_model.pkl')
parser.add_argument('---save_eng_model', type=str, default='./data/save_eng_model.pkl')

parser.add_argument('---train_path', type=str, default='./data/train.txt')
parser.add_argument('---dev_path', type=str, default='./data/dev.txt')
parser.add_argument('---test_path', type=str, default='./data/test.txt')

parser.add_argument('---quora_train_path', type=str, default='./data/quora_train.txt')
parser.add_argument('---quora_dev_path', type=str, default='./data/quora_dev.txt')
parser.add_argument('---quora_test_path', type=str, default='./data/quora_test.txt')

parser.add_argument('---vocab_path', type=str, default='./data/vocab.txt')
parser.add_argument('---quora_vocab_path', type=str, default='./data/eng_vocab.txt')


parser.add_argument('---save_vocab', type=str, default='./data/vocab.pickle')
parser.add_argument('---save_eng_vocab', type=str, default='./data/eng_vocab.pickle')

parser.add_argument('---eng_pretrain_train', type=str, default='./data/ENG_sim_train.txt')
parser.add_argument('---eng_pretrain_test', type=str, default='./data/ENG_sim_test.txt')

parser.add_argument('---pretrain_train', type=str, default='./data/CHN_sim_train.txt')
parser.add_argument('---pretrain_test', type=str, default='./data/CHN_sim_test.txt')
parser.add_argument('---pretrain_train1', type=str, default='./data/chn_nosim.txt')

parser.add_argument('---save_pretrain_eng_vocab', type=str, default='./pretrain_model/eng_vocab.pickle')
parser.add_argument('---save_pretrain_chn_vocab', type=str, default='./pretrain_model/vocab.pickle')

parser.add_argument('---save_pre_eng_model', type=str, default='./pretrain_model/save_pre_eng_model.pkl')
parser.add_argument('---save_pre_eng_model1', type=str, default='./pretrain_model/save_pre_eng_model1.pkl')
parser.add_argument('---save_pre_eng_model2', type=str, default='./pretrain_model/save_pre_eng_model2.pkl')
parser.add_argument('---save_pre_chn_model', type=str, default='./pretrain_model/save_pre_chn_model.pkl')
parser.add_argument('---save_pre_chn_model1', type=str, default='./pretrain_model/save_pre_chn_mode1.pkl')
parser.add_argument('---save_pre_chn_model2', type=str, default='./pretrain_model/save_pre_chn_mode2.pkl')
args = parser.parse_args()
device = torch.device('cuda' if args.cuda else 'cpu')
