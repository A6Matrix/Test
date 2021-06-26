import torch
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
# from DataUtils import load_vocab,Data_prepare,batch_data,pad_collate_fn
from tqdm import tqdm
from data_prepare1 import Data_Prepare
from get_emb import get_weight
from baseline1 import sim_model
from config import args
from Optim import ScheduledOptim
from sklearn.metrics import f1_score
from sklearn import metrics
from tensorflow.contrib import learn
import torch.nn as nn
import os
import random

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
device = torch.device('cuda' if args.cuda else 'cpu')
data_pre = Data_Prepare()
pretrain_embed = get_weight(args.vocab_path)


def pre_processing():
    train_texta, train_textb, train_tag = data_pre.readfile(args.train_path)
    #dev_texta, dev_textb, dev_tag = data_pre.readfile(args.quora_dev_path)
    data = []
    data.extend(train_texta)
    data.extend(train_textb)
    #data.extend(dev_texta)
    #data.extend(dev_textb)
    data_pre.build_vocab(data, args.save_vocab)


def data_shuffle(texta, textb, tag):
    index = [x for x in range(len(texta))]
    random.shuffle(index)
    texta_new = [texta[x] for x in index]
    textb_new = [textb[x] for x in index]
    tag_new = [tag[x] for x in index]
    return texta_new, textb_new, tag_new


def load_vocab(texta, textb, tag):
    # 加载词典
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(args.save_vocab)
    texta_embedding = torch.tensor(list(vocab_processor.transform(texta)))
    textb_embedding = torch.tensor(list(vocab_processor.transform(textb)))

    return texta_embedding, textb_embedding, torch.FloatTensor(tag)


def get_batches(texta, textb, tag):
    num_batch = int(len(texta) / args.batch_size)
    for i in range(num_batch):
        a = texta[i * args.batch_size:(i + 1) * args.batch_size]
        b = textb[i * args.batch_size:(i + 1) * args.batch_size]
        t = tag[i * args.batch_size:(i + 1) * args.batch_size]
        yield a, b, t


def train_epoch(model, train_texta, train_textb, tag, optimizer, embed, device):
    model.train()
    total_loss = []
    total_acc = []
    for texta, textb, tag in tqdm(
            get_batches(train_texta, train_textb, tag)):
        # train_texta, train_textb, train_tag = map(lambda x: x.to(device), batch)
        #a_mask = (torch.zeros_like(texta) == texta).float().to(device)
        #b_mask = (torch.zeros_like(textb) == textb).float().to(device)
        texta = embed(texta).to(device)
        textb = embed(textb).to(device)
        tag = tag.to(device)
        optimizer.zero_grad()
        output = model(texta, textb)
        loss = F.binary_cross_entropy(output, tag)
        # loss = (F.kl_div(output,tag)+F.kl_div(tag,output))/2
        loss.backward()
        optimizer.step_and_update_lr()
        pred = torch.argmax(output, dim=1)
        target = tag[:, 1].long()
        acc = (((pred == target).sum()).item()) / len(target)
        # acc = (pred == target).mean()
        total_loss.append(loss.data)
        total_acc.append(acc)
        # predic = torch.max(output.data, 1)[1]
        # acc = metrics.accuracy_score(tag, predic)

    return total_acc, total_loss


def eval_epoch(model, texta, textb, tag, embed, device):
    total_loss = []
    total_acc = []
    model.eval()
    with torch.no_grad():
        for texta, textb, tag in tqdm(
                get_batches(texta, textb, tag)):
            # texta, textb, tag = map(lambda x:x.to(device),batch)
            #a_mask = (torch.zeros_like(texta) == texta).float().to(device)
            #b_mask = (torch.zeros_like(textb) == textb).float().to(device)
            texta = embed(texta).to(device)
            textb = embed(textb).to(device)
            tag = tag.to(device)
            output = model(texta, textb)
            loss = F.binary_cross_entropy(output, tag)
            pred = torch.argmax(output, dim=1)
            target = tag[:, 1].long()
            acc = (((pred == target).sum()).item()) / len(target)
            total_loss.append(loss.data)
            total_acc.append(acc)

            # loss = F.cross_entropy(output,tag)
            # # pre = torch.argmax(output, dim=1).data
            # predic = torch.max(output.data, 1)[1]
            # acc =  metrics.accuracy_score(tag, predic)
            # # total_pre.append(pre)
            # total_loss.append(loss.data)
            # total_acc.append(acc)

    # f1 = f1_score(np.array(tag), np.array(total_pre), average='weighted')
    return total_acc, total_loss


def train(model, dev_texta, dev_textb, dev_tag, optimizer, embed, device):
    for epoch in range(args.epoch):
        train_texta, train_textb, train_tag = data_pre.readfile(args.train_path)
        texta, textb, tag = data_shuffle(train_texta, train_textb, train_tag)
        train_texta_embed, train_textb_embed, train_tag_embed = load_vocab(texta, textb, tag)

        print("training " + str(epoch + 1) + ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        train_acc, train_loss = train_epoch(model, train_texta_embed, train_textb_embed, train_tag_embed, optimizer,
                                            embed,
                                            device)

        print("第" + str((epoch + 1)) + "次迭代训练集的损失为：" + str(torch.mean(torch.Tensor(train_loss)).item()) + ";准确率为：" +
              str(torch.mean(torch.Tensor(train_acc)).item()))

        if (epoch + 1) % 5 == 0:
            dev_acc, dev_loss = eval_epoch(model, dev_texta, dev_textb, dev_tag, embed, device)
            print("第" + str((epoch + 1)) + "次迭代验证集的损失为：" + str(torch.mean(torch.Tensor(dev_loss)).item()) + ";准确率为：" +
                  str(torch.mean(torch.Tensor(dev_acc)).item()))
            torch.save(model.state_dict(), args.save_model)


def test(model, test_texta, test_textb, test_tag, embed, device):
    model.load_state_dict(torch.load(args.save_model))
    test_acc, test_loss = eval_epoch(model, test_texta, test_textb, test_tag, embed, device)
    print("测试集的损失为：" + str(torch.mean(torch.Tensor(test_loss)).item()) + ";准确率为：" +
          str(torch.mean(torch.Tensor(test_acc)).item()))


if __name__ == '__main__':
    model = sim_model().to(device)
    pre_processing()
    embed = nn.Embedding.from_pretrained(pretrain_embed)
    # embed = nn.Embedding(400000,300)

    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda x: x.requires_grad, model.parameters()),
            betas=(0.9, 0.98), eps=1e-03),
        init_lr=args.learning_rate, n_warmup_steps=args.n_warmup_steps)

    # train_texta,train_textb,train_tag = data_pre.readfile(args.train_path)
    dev_texta, dev_textb, dev_tag = data_pre.readfile(args.dev_path)
    test_texta, test_textb, test_tag = data_pre.readfile(args.test_path)
    dev_texta_embed, dev_textb_embed, dev_tag_embed = load_vocab(dev_texta, dev_textb, dev_tag)
    test_texta_embed, test_textb_embed, test_tag_embed = load_vocab(test_texta, test_textb, test_tag)

    train(model, dev_texta_embed, dev_textb_embed, dev_tag_embed, optimizer, embed, device)
    test(model, test_texta_embed, test_textb_embed, test_tag_embed, embed, device)
