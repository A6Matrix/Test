import torch.nn as nn
import torch.nn.functional as F
import torch
from sklearn.metrics import classification_report

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, k,dim=1, bias=True):
        super().__init__()
        if dim == 1:
            self.depthwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=in_ch, kernel_size=k, groups=in_ch,
                                            padding=k // 2, bias=bias)
            self.pointwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, bias=bias)
        elif dim == 2:
            self.depthwise_conv = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=k, groups=in_ch,
                                            padding=k // 2, bias=bias)
            self.pointwise_conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, bias=bias)
        else:
            raise Exception("Wrong dimension for Depthwise Separable Convolution!")
        nn.init.kaiming_normal_(self.depthwise_conv.weight)
        nn.init.constant_(self.depthwise_conv.bias, 0.0)
        nn.init.kaiming_normal_(self.depthwise_conv.weight)
        nn.init.constant_(self.pointwise_conv.bias, 0.0)

    def forward(self, x):
        return self.pointwise_conv(self.depthwise_conv(x))


class Conv_Topic(nn.Module):
    def __init__(self,embeddings,in_ch=600, out_ch=300, k=3, nums_label=2, dropout=0.3):
        super().__init__()
        self.embeds_dim = embeddings.shape[1]
        self.embed = nn.Embedding(embeddings.shape[0], embeddings.shape[1])
        self.embed.weight = nn.Parameter(torch.from_numpy(embeddings))
        self.embed.float()
        self.embed.weight.requires_grad = True
        self.convs = DepthwiseSeparableConv(in_ch=in_ch, out_ch=out_ch, k=k)
        # self.Linear = nn.Linear(out_ch, nums_label)
        self.projected = nn.Linear(4*out_ch, 2*out_ch)
        self.fc1 = nn.Linear(2*out_ch, out_ch)
        self.fc2 = nn.Linear(out_ch, nums_label)
        self.dropout = nn.Dropout(dropout)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace=True)
    def apply_multiple(self, x):
        # input: batch_size * seq_len * (2 * hidden_size)
        p1 = F.avg_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        # p2 = F.max_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        # output: batch_size * (4 * hidden_size)
        return p1

    def forward(self,text_a,text_b,texta_trigram,textb_trigram,topic_represent=None):
        # mask1, mask2 = text_a.eq(0), text_a.eq(0)
        texta_embed = self.embed(text_a)
        textb_embed = self.embed(text_b)
        texta_trigram = self.embed(texta_trigram)
        textb_trigram = self.embed(textb_trigram)
        texta_input = torch.cat((texta_embed,texta_trigram),-1)
        textb_input = torch.cat((textb_embed,textb_trigram),-1)
        # texta_input = texta_input.mean(dim=1)
        # textb_input = textb_input.mean(dim=1)
        texta_out = F.relu(self.convs(texta_input.transpose(1, 2)).transpose(1,2))
        textb_out = F.relu(self.convs(textb_input.transpose(1, 2)).transpose(1,2))

        # texta_out = texta_out * mask1.eq(0).unsqueeze(2).float()
        # textb_out = textb_out * mask2.eq(0).unsqueeze(2).float()
        # topic_represent = topic_represent.unsqueeze(1)

        combined = torch.cat([texta_out, textb_out, texta_out - textb_out, texta_out * textb_out], dim=-1) #bz*sq*4H

        out = self.apply_multiple(combined)

        project = self.projected(out)

        project = self.dropout(project)
        
        project = self.tanh(project)

        out = self.fc1(project)

        out = self.dropout(out)

        out = self.relu(out)

        out = self.fc2(out)

        return out











