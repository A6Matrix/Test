import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from data_prepare import Data_Prepare
from tensorflow.contrib import learn
from get_emb import get_eng_weight
from config import args
import random
from Optim import ScheduledOptim
from mse_loss import Loss
from mse_model import VAE
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
device = torch.device('cuda' if args.cuda else 'cpu')
data_pre = Data_Prepare()
pretrain_embed = get_eng_weight(args.quora_vocab_path)


def pre_processing():
    train_text = data_pre.read_pre_file(args.eng_pretrain_train)
    data = []
    data.extend(train_text)
    data_pre.build_vocab(data, args.save_pretrain_eng_vocab)


def data_shuffle(text):
    index = [x for x in range(len(text))]
    random.shuffle(index)
    text_new = [text[x] for x in index]
    return text_new


def load_vocab(text):
    # 加载词典
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(args.save_pretrain_eng_vocab)
    text_embedding = torch.tensor(list(vocab_processor.transform(text)))

    return text_embedding


def get_batches(text):
    num_batch = int(len(text) / args.batch_size)
    for i in range(num_batch):
        a = text[i * args.batch_size:(i + 1) * args.batch_size]
        yield a


def train_epoch(model, train_text, loss, optimizer, embed, device):
    model.train()
    total_loss = []
    for text in tqdm(
            get_batches(train_text)):
        text = embed(text).to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(text)
        train_loss = loss(recon_batch, text, mu, logvar)
        train_loss.backward()
        optimizer.step_and_update_lr()
        total_loss.append(train_loss.data)

    return total_loss


def eval_epoch(model, text, loss, embed, device):
    total_loss = []
    model.eval()
    with torch.no_grad():
        for text in tqdm(
                get_batches(text)):
            text = embed(text).to(device)
            recon_batch, mu, logvar = model(text)
            test_loss = loss(recon_batch, text, mu, logvar)
            total_loss.append(test_loss.data)

    return total_loss


def train(model, optimizer, loss, embed, device):
    for epoch in range(args.epoch):
        train_text = data_pre.read_pre_file(args.eng_pretrain_train)
        text = data_shuffle(train_text)
        train_text_embed = load_vocab(text)

        print("training " + str(epoch + 1) + ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        train_loss = train_epoch(model, train_text_embed, loss, optimizer, embed, device)

        print("第" + str((epoch + 1)) + "次迭代训练集的损失为：" + str(torch.mean(torch.Tensor(train_loss)).item()))

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), args.save_pre_eng_model2)


def test(model, test_text, loss, embed, device):
    model.load_state_dict(torch.load(args.save_pre_eng_model2))
    test_loss = eval_epoch(model, test_text, loss, embed, device)
    print("测试集的损失为：" + str(torch.mean(torch.Tensor(test_loss)).item()))


if __name__ == '__main__':
    model = VAE().to(device)
    pre_processing()
    embed = nn.Embedding.from_pretrained(pretrain_embed)
    # embed = nn.Embedding(400000,300)
    loss = Loss()
    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda x: x.requires_grad, model.parameters()),
            betas=(0.9, 0.98), eps=1e-03),
        init_lr=args.learning_rate, n_warmup_steps=args.n_warmup_steps)

    test_text = data_pre.read_pre_file(args.eng_pretrain_test)
    test_text_embed = load_vocab(test_text)

    train(model, optimizer, loss, embed, device)
    test(model, test_text_embed, loss, embed, device)
