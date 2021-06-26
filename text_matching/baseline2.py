import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import math
import numpy as np
from config import args
# from get_emb import get_weight
# d_model = 96
# n_head = 8
# d_word = 300
# batch_size = 6
# dropout = 0.1
# seq_len = 10
# d_k = d_model//n_head

d_model = args.hidden_size
n_head = args.n_head
d_word = args.embed_size
batch_size = args.batch_size
dropout = args.dropout
seq_len = args.max_seq_len
d_k = d_model//n_head
pre_dic = torch.load('./pretrain_model/save_pre_eng_model.pkl')
pre_dic1 = torch.load('./pretrain_model/save_pre_eng_model1.pkl')
# pretrain_embed = get_weight(args.vocab_path)
# device = torch.device('cuda' if args.cuda else 'cpu')

def mask_logits(target, mask):
    return target * (1-mask) + mask * (-1e30)
#
class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=seq_len):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
#
#
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.q_linear = nn.Linear(d_word, d_model)
        self.v_linear = nn.Linear(d_word, d_model)
        self.k_linear = nn.Linear(d_word, d_model)
        # self.BN = nn.BatchNorm1d()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, d_model)
        self.a = 1 / math.sqrt(d_k)

    def forward(self, x, mask):
        bs, l_x, _ = x.size()
        # x = x.transpose(1, 2)
        k = self.k_linear(x).view(bs, l_x, n_head, d_k)
        q = self.q_linear(x).view(bs, l_x, n_head, d_k)
        v = self.v_linear(x).view(bs, l_x, n_head, d_k)
        q = q.permute(2, 0, 1, 3).contiguous().view(bs * n_head, l_x, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(bs * n_head, l_x, d_k)
        v = v.permute(2, 0, 1, 3).contiguous().view(bs * n_head, l_x, d_k)
        mask = mask.unsqueeze(1).expand(-1, l_x, -1).repeat(n_head, 1, 1)

        attn = torch.bmm(q, k.transpose(1, 2)) * self.a
        attn = mask_logits(attn, mask)
        attn = F.softmax(attn, dim=2)
        attn = self.dropout(attn)

        out = torch.bmm(attn, v)
        out = out.view(n_head, bs, l_x, d_k).permute(1, 2, 0, 3).contiguous().view(bs, l_x, d_model)
        out = self.fc(out)
        out = self.dropout(out)
        return out


class Highway(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.linear = nn.Linear(size, size)
        self.gate =nn.Linear(size, size)

    def forward(self, x):
        # x = x.transpose(1, 2)
        gate = torch.sigmoid(x)
        nonlinear = F.relu(x)
        x = gate * nonlinear + (1 - gate) * x
        # x = x.transpose(1, 2)
        return x

class Embedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.high = Highway(d_word)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        # bz,seq,d_word = x.size()
        # wd_emd = self.BN(x.contiguous().view(-1,d_word))
        # wd_emd = wd_emd.view(bz,seq,d_word)
        wd_emd = self.dropout(x)
        emd = self.high(wd_emd)
        return emd

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, k, bias=True):
        super().__init__()
        self.depthwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=in_ch, kernel_size=k, groups=in_ch,
                                            padding=k // 2, bias=bias)
        self.pointwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, bias=bias)
        if k==3:
            self.depthwise_conv.weight = nn.Parameter(pre_dic['conv1.depthwise_conv.weight'])
            self.depthwise_conv.bias = nn.Parameter(pre_dic['conv1.depthwise_conv.bias'])
            self.pointwise_conv.weight = nn.Parameter(pre_dic['conv1.pointwise_conv.weight'])
            self.pointwise_conv.bias = nn.Parameter(pre_dic['conv1.pointwise_conv.bias'])
        if k == 5:
            self.depthwise_conv.weight = nn.Parameter(pre_dic1['conv1.depthwise_conv.weight'])
            self.depthwise_conv.bias = nn.Parameter(pre_dic1['conv1.depthwise_conv.bias'])
            self.pointwise_conv.weight = nn.Parameter(pre_dic1['conv1.pointwise_conv.weight'])
            self.pointwise_conv.bias = nn.Parameter(pre_dic1['conv1.pointwise_conv.bias'])
        # nn.init.kaiming_normal_(self.depthwise_conv.weight)
        # nn.init.constant_(self.depthwise_conv.bias, 0.0)
        # nn.init.kaiming_normal_(self.depthwise_conv.weight)
        # nn.init.constant_(self.pointwise_conv.bias, 0.0)

    def forward(self, x):
        return self.pointwise_conv(self.depthwise_conv(x))

class EncoderBlock(nn.Module):
    def __init__(self, in_ch,out_ch, k1,k2):
        super().__init__()
        self.convs1 = DepthwiseSeparableConv(in_ch,out_ch,k1)
        self.convs2 = DepthwiseSeparableConv(in_ch,out_ch,k2)
        # self.convs3 = DepthwiseSeparableConv(in_ch,out_ch,k2)
        self.self_att = MultiHeadAttention()
        self.pos = PositionalEmbedding(d_word)
        self.norm = nn.BatchNorm1d(3*d_model)

    def forward(self, x,mask):
        # bz,s_l,_ = x.size()
        x_atten = x + self.pos(x)
        # out = self.normb(out)
        out1 = self.self_att(x_atten,mask)
        out1 = out1.transpose(1,2)
        x_conv = x.transpose(1,2)
        #(bz*d_model*seq_len)
        out2 = self.convs1(x_conv)#(bz*d_model*seq_len)
        out3 = self.convs2(x_conv)#(bz*d_model*seq_len)
        # out = torch.cat([out1,out2,out3],dim=1)#bz*3d_model*sl
        out = torch.cat([out1,out2,out3],dim=1)#bz*3d_model*sl
        out = self.norm(out)
        out = out.transpose(1, 2)
        out = F.relu(out)
        # out = self.normb(out)
        # res = out
        # out = F.dropout(out,p=dropout,training=True)
        # out = self.self_att(out, mask)
        # out = out + res
        return out

def soft_attention_align( x1, x2):
    '''
    x1: batch_size * seq_len * dim
    x2: batch_size * seq_len * dim
    '''
    # attention: batch_size * seq_len * seq_len
    attention = torch.matmul(x1, x2.transpose(1, 2))

    # weight: batch_size * seq_len * seq_len
    weight1 = F.softmax(attention,dim=-1)
    x1_align = torch.matmul(weight1, x2)
    weight2 = F.softmax(attention.transpose(1, 2), dim=-1)
    x2_align = torch.matmul(weight2, x1)
    # x_align: batch_size * seq_len * hidden_size
    return x1_align, x2_align

#cosine-similarity
class sim_model(nn.Module):
    # def __init__(self,d_word,d_model,max_seq_len,context_pretrain_embed):
    def __init__(self):
        super().__init__()
        # self.word_embed = nn.Embedding.from_pretrained(pretrain_embed)
        # self.word_embed = nn.Embedding(1000,300)
#         self.texta_conv = DepthwiseSeparableConv(d_word,d_model,5)
#         self.textb_conv = DepthwiseSeparableConv(d_word,d_model,5)
        self.embed = Embedding()
        self.texta_enc = EncoderBlock(in_ch=d_word,out_ch=d_model,k1=3,k2=5)
        # self.texta_enc = EncoderBlock(in_ch=d_word,out_ch=d_model,k1=3)
        self.fc = nn.Sequential(
            nn.Linear(24*d_model, 12*d_model),
            # nn.Linear(16*d_model, 8*d_model),
            nn.BatchNorm1d(12*d_model),
            nn.ELU(inplace=True),
            # nn.Dropout(dropout),
            nn.Linear(12*d_model, 6*d_model),
            # nn.BatchNorm1d(6*d_model),
            nn.ELU(inplace=True),
            # nn.Dropout(dropout),
            nn.Linear(6*d_model, d_model),
            nn.BatchNorm1d(d_model),
            nn.ELU(inplace=True),
            nn.Linear(d_model, 2),
            nn.ELU()
        )

    def apply_multiple(self,x):
        # input: batch_size * seq_len * (2 * hidden_size)
        p1 = F.avg_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        p2 = F.max_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        # output: batch_size * (4 * hidden_size)
        return torch.cat([p1, p2], 1)

    def forward(self,texta,textb,a_mask,b_mask):
        # a_mask = (torch.zeros_like(texta) == texta).float()
        # b_mask = (torch.zeros_like(textb) == textb).float()
        #
        # texta_embed = self.word_embed(texta)
        # textb_embed = self.word_embed(textb)

        a_embed = self.embed(texta)
        b_embed = self.embed(textb)

        a_enc = self.texta_enc(a_embed,a_mask)
        b_enc = self.texta_enc(b_embed,b_mask)

        a_attn, b_attn = soft_attention_align(a_enc, b_enc)
        a_out = torch.cat([a_enc,a_attn],dim=-1)
        b_out = torch.cat([b_enc,b_attn],dim=-1)

        a_out = self.apply_multiple(a_out)
        b_out = self.apply_multiple(b_out)

        out = torch.cat([a_out,b_out],dim=1)
        out = self.fc(out)
        out = F.sigmoid(out)
        # out = torch.cosine_similarity(texta_out,textb_out,dim=1)
        return out

if __name__ == '__main__':
    model = sim_model()
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

    loss_total = []
    acc_total = []
    for i in range(5):
        yb = torch.FloatTensor([[0, 1], [1, 0], [0, 1], [1, 0]])
        out = model(texta, textb)
        # print ("model(xb):",out)
        loss = F.binary_cross_entropy(out, yb)
        pred = torch.argmax(out, dim=1)
        target = yb[:, 1].long()
        acc = (((pred == target).sum()).item()) / len(target)

        loss_total.append(loss.data)
        acc_total.append(acc)
        print("loss:", loss.data)
        print("acc:", acc)
    print(np.mean(np.array(loss_total)))
    print(np.mean(np.array(acc_total)))

    # print(acc_total/len(yb))
