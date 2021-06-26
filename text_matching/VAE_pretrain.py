import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
#from data_prepare1 import Data_Prepare
#from tensorflow.contrib import learn
from get_emb import get_weight
from config import args
import random
from Optim import ScheduledOptim
from mse_loss import Loss
from mse_model import VAE
import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if args.cuda else 'cpu')
#data_pre = Data_Prepare()
pretrain_embed = get_weight(args.vocab_path)

# def gen_pic(texta,textb):
#     emb = texta.size(2)
#     x = texta.view(-1,emb)
#     x = x.permute(1,0).cpu().numpy()
#     #x = x.cpu().numpy()
#     y = textb.view(-1,emb)
#     y = y.permute(1,0).cpu().detach().numpy()
#     #y = y.cpu().numpy()
#     pca = PCA(n_components=2)
#     x_new = pca.fit_transform(x)
#     y_new = pca.fit_transform(y)
#     plt.scatter(x_new[:,0], x_new[:,1], marker='o',s=0.5, c='r')
#     plt.scatter(y_new[:,0], y_new[:,1], marker='o',s=0.5, c='g')
#     fig = plt.gcf()
#     plt.show()
#     fig.savefig('test3.png',dpi=100)

def pre_processing():
    train_texta, train_textb = data_pre.read_pre_file(args.pretrain_train1)
    data = []
    data.extend(train_texta)
    data.extend(train_textb)
    data_pre.build_vocab(data, args.save_pretrain_chn_vocab)


def data_shuffle(texta, textb):
    index = [x for x in range(len(texta))]
    random.shuffle(index)
    texta_new = [texta[x] for x in index]
    textb_new = [textb[x] for x in index]
    return texta_new, textb_new


# def load_vocab(texta, textb):
#     # 加载词典
#     vocab_processor = learn.preprocessing.VocabularyProcessor.restore(args.save_pretrain_chn_vocab)
#     texta_embedding = torch.tensor(list(vocab_processor.transform(texta)))
#     textb_embedding = torch.tensor(list(vocab_processor.transform(textb)))
# 
#     return texta_embedding, textb_embedding


def get_batches(texta, textb):
    num_batch = int(len(texta) / args.batch_size)
    for i in range(num_batch):
        a = texta[i * args.batch_size:(i + 1) * args.batch_size]
        b = textb[i * args.batch_size:(i + 1) * args.batch_size]
        yield a, b


def train_epoch(model, train_texta, train_textb, loss, optimizer, embed, device):
    model.train()
    total_loss = []
    for texta, textb in tqdm(
            get_batches(train_texta, train_textb)):
        texta = embed(texta).to(device)
        textb = embed(textb).to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(texta)
        # gen_pic(textb,recon_batch)
        train_loss = loss(recon_batch, textb, mu, logvar)
        train_loss.backward()
        optimizer.step_and_update_lr()
        total_loss.append(train_loss.data)

    return total_loss


def eval_epoch(model, texta, textb, loss, embed, device):
    total_loss = []
    model.eval()
    with torch.no_grad():
        for texta, textb in tqdm(
                get_batches(texta, textb)):
            texta = embed(texta).to(device)
            print(texta.size())
            textb = embed(textb).to(device)
            print(textb.size())
            recon_batch, mu, logvar = model(texta)
            test_loss = loss(recon_batch, textb, mu, logvar)
            total_loss.append(test_loss.data)
            break

    return total_loss


def train(model, optimizer, loss, embed, device):
    for epoch in range(args.epoch):
        train_texta, train_textb = data_pre.read_pre_file(args.pretrain_train)
        texta, textb = data_shuffle(train_texta, train_textb)
        # train_texta_embed, train_textb_embed = load_vocab(texta, textb)

        print("training " + str(epoch + 1) + ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        # train_loss = train_epoch(model, train_texta_embed, train_textb_embed, loss, optimizer, embed, device)

        # print("第" + str((epoch + 1)) + "次迭代训练集的损失为：" + str(torch.mean(torch.Tensor(train_loss)).item()))

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), args.save_pre_chn_model)


def test(model, test_texta, test_textb, loss, embed, device):
    model.load_state_dict(torch.load(args.save_pre_chn_model))
    test_loss = eval_epoch(model, test_texta, test_textb, loss, embed, device)
    print("测试集的损失为：" + str(torch.mean(torch.Tensor(test_loss)).item()))


if __name__ == '__main__':
    model = VAE().to(device)
    total_params = 0
    Trainable_params = 0
    NonTrainable_params = 0
    for param in model.parameters():
        mulValue = np.prod(param.size())  # 使用numpy prod接口计算参数数组所有元素之积
        total_params += mulValue  # 总参数量
        if param.requires_grad:
            Trainable_params += mulValue  # 可训练参数量
        else:
            NonTrainable_params += mulValue  # 非可训练参数量
    print(f'Total params: {total_params/1e6}')
    print(f'Trainable params: {Trainable_params/1e6}')
    print(f'Non-trainable params: {NonTrainable_params/1e6}')

    pre_processing()
    embed = nn.Embedding.from_pretrained(pretrain_embed)
    # embed = nn.Embedding(400000,300)
    loss = Loss()
    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda x: x.requires_grad, model.parameters()),
            betas=(0.9, 0.98), eps=1e-03),
        init_lr=args.learning_rate, n_warmup_steps=args.n_warmup_steps)

    test_texta, test_textb = data_pre.read_pre_file(args.pretrain_train1)
    # test_texta_embed, test_textb_embed = load_vocab(test_texta, test_textb)

    # train(model, optimizer, loss, embed, device)
    # test(model, test_texta_embed, test_textb_embed, loss, embed, device)
