import torch
import torch.nn.functional as F
import numpy as np
# from DataUtils import load_vocab,Data_prepare,batch_data,pad_collate_fn
from tqdm import tqdm
from data_prepare import Data_Prepare
from Optim import ScheduledOptim
import torch.optim as optim
# from get_emb import get_weight
from baseline import sim_model
from config import args
from sklearn.metrics import f1_score
from sklearn import metrics
from tensorflow.contrib import learn
import torch.nn as nn

device = torch.device('cuda' if args.cuda else 'cpu')
data_pre = Data_Prepare()

def pre_processing():
    train_texta, train_textb, train_tag = data_pre.readfile(args.train_path)
    dev_texta, dev_textb, dev_tag = data_pre.readfile(args.dev_path)
    data = []
    data.extend(train_texta)
    data.extend(train_textb)
    data.extend(dev_texta)
    data.extend(dev_textb)

    data_pre.build_vocab(data, args.save_vocab)
    # 加载词典
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(args.save_vocab)
    train_texta_embedding = torch.tensor(list(vocab_processor.transform(train_texta)))
    train_textb_embedding = torch.tensor(list(vocab_processor.transform(train_textb)))

    dev_texta, dev_textb, dev_tag = data_pre.readfile(args.dev_path)
    dev_texta_embedding = torch.tensor(list(vocab_processor.transform(dev_texta)))
    dev_textb_embedding = torch.tensor(list(vocab_processor.transform(dev_textb)))

    test_texta, test_textb, test_tag = data_pre.readfile(args.test_path)
    test_texta_embedding = torch.tensor(list(vocab_processor.transform(test_texta)))
    test_textb_embedding = torch.tensor(list(vocab_processor.transform(test_textb)))

    return train_texta_embedding, train_textb_embedding, torch.tensor(train_tag), \
           dev_texta_embedding, dev_textb_embedding, torch.tensor(dev_tag), \
           test_texta_embedding, test_textb_embedding, torch.tensor(test_tag)


def get_batches( texta, textb, tag):
    num_batch = int(len(texta) / args.batch_size)
    for i in range(num_batch):
        a = texta[i * args.batch_size:(i + 1) * args.batch_size]
        b = textb[i * args.batch_size:(i + 1) * args.batch_size]
        t = tag[i * args.batch_size:(i + 1) * args.batch_size]
        yield a, b, t


def get_length(trainX_batch):
    # sentence length
    lengths = []
    for sample in trainX_batch:
        count = 0
        for index in sample:
            if index != 0:
                count += 1
            else:
                break
        lengths.append(count)
    return lengths
def train_epoch(model,train_texta,train_textb,tag,optimizer,device):
    model.train()
    total_loss = []
    total_acc = []
    for texta, textb, tag in tqdm(
            get_batches(train_texta.to(device),train_textb.to(device),tag.to(device))):
        # train_texta, train_textb, train_tag = map(lambda x: x.to(device), batch)

        optimizer.zero_grad()
        output = model(texta, textb).to(device)
        loss = F.hinge_embedding_loss(output, tag)
        loss.backward()
        optimizer.step_and_update_lr()
        zeros = torch.zeros(output.shape)
        ones = torch.ones(output.shape)
        out_new = torch.where(output > 0, ones, zeros).long()
        acc = (out_new == tag).sum()
        acc_data= acc.item()/len(tag)
        total_loss.append(loss.data)
        total_acc.append(acc_data)
        # predic = torch.max(output.data, 1)[1]
        # acc = metrics.accuracy_score(tag, predic)
    return total_acc,total_loss

def eval_epoch(model,texta,textb,tag,device):
    total_loss = []
    total_acc = []
    model.eval()
    with torch.no_grad():
        for texta, textb, tag in tqdm(
                get_batches(texta.to(device),textb.to(device),tag.to(device))):
            # texta, textb, tag = map(lambda x:x.to(device),batch)

            output = model(texta,textb).to(device)
            loss = F.hinge_embedding_loss(output, tag)
            zeros = torch.zeros(output.shape)
            ones = torch.ones(output.shape)
            out_new = torch.where(output > 0, ones, zeros).long()
            acc = (out_new == tag).sum()
            acc_data = acc.item() / len(tag)
            total_loss.append(loss.data)
            total_acc.append(acc_data)
            # loss = F.cross_entropy(output,tag)
            # # pre = torch.argmax(output, dim=1).data
            # predic = torch.max(output.data, 1)[1]
            # acc =  metrics.accuracy_score(tag, predic)
            # # total_pre.append(pre)
            # total_loss.append(loss.data)
            # total_acc.append(acc)

    # f1 = f1_score(np.array(tag), np.array(total_pre), average='weighted')
    return total_acc,total_loss

def train(model,train_texta, train_textb, train_tag,dev_texta, dev_textb, dev_tag,optimizer,device):

    for epoch in range(args.epoch):
        print("training " + str(epoch + 1) + ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        train_acc ,train_loss = train_epoch(model,train_texta, train_textb, train_tag,optimizer,device)

        print("第" + str((epoch + 1)) + "次迭代训练集的损失为：" + str(np.mean(np.array(train_loss))) + ";准确率为：" +
              str(np.mean(np.array(train_acc))))

        if (epoch+1) % 5 == 0:
            dev_acc,dev_loss = eval_epoch(model,dev_texta, dev_textb, dev_tag,device)
            print("第" + str((epoch + 1)) + "次迭代验证集的损失为：" + str(np.mean(np.array(dev_loss))) + ";准确率为：" +
                  str(np.mean(np.array(dev_acc))))
            torch.save(model.state_dict(), args.save_model)


def test(args,model,test_texta,test_textb,test_tag,device):
    model.load_state_dict(torch.load(args.save_model))
    test_acc,test_loss = eval_epoch(model,test_texta,test_textb,test_tag,device)
    print("测试集的损失为：" + str(np.mean(np.array(test_loss))) + ";准确率为：" +
          str(np.mean(np.array(test_acc))))

if __name__ == '__main__':
    model = sim_model()

    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda x: x.requires_grad, model.parameters()),
            betas=(0.9, 0.98), eps=1e-03),
        init_lr=args.learning_rate, n_warmup_steps=args.n_warmup_steps)

    train_texta, train_textb, train_tag, \
    dev_texta, dev_textb, dev_tag, \
    test_texta,test_textb,test_tag = pre_processing()

    train(model,\
          train_texta, train_textb, train_tag,\
          dev_texta, dev_textb, dev_tag,optimizer,device)

    test(args,model,test_texta,test_textb,test_tag,device)