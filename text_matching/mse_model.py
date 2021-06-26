import torch.nn as nn
from torch.autograd import Variable
import torch
from config import args

d_word = args.embed_size
dropout = args.dropout
d_model = args.hidden_size
class Depth_Conv(nn.Module):
    def __init__(self, in_ch, out_ch, k, stride, bias=True):
        super().__init__()
        self.depthwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=in_ch, kernel_size=k, stride=stride,
                                        groups=in_ch,
                                        padding=k // 2, bias=bias)
        self.pointwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=stride, padding=0,
                                        bias=bias)

    def forward(self, x):
        return self.pointwise_conv(self.depthwise_conv(x))


class Depth_TranseConv(nn.Module):
    def __init__(self, in_ch, out_ch, k, stride, bias=True):
        super().__init__()
        self.depth_TranseConv = nn.ConvTranspose1d(in_channels=in_ch, out_channels=out_ch, kernel_size=k, stride=stride,
                                                   padding=k // 2, bias=bias)
        self.point_TranseConv = nn.ConvTranspose1d(in_channels=out_ch, out_channels=out_ch, kernel_size=1,
                                                   stride=stride, bias=bias)

    def forward(self, x):
        return self.point_TranseConv(self.depth_TranseConv(x))


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # Encoder
        self.conv1 = Depth_Conv(d_word, d_model, k=3, stride=1)
        self.bn1 = nn.BatchNorm1d(d_model)
        self.conv2 = Depth_Conv(d_model, d_model, k=3, stride=1)
        self.bn2 = nn.BatchNorm1d(d_model)

        self.fc21 = nn.Linear(d_model, d_model)
        self.fc22 = nn.Linear(d_model, d_model)

        # Decoder
        self.fc3 = nn.Linear(d_model, d_model)

        self.conv3 = Depth_TranseConv(d_model, d_model, k=3, stride=1)
        self.bn3 = nn.BatchNorm1d(d_model)
        self.conv4 = Depth_TranseConv(d_model, d_word, k=3, stride=1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    def encode(self, x):
        conv1 = self.relu(self.bn1(self.conv1(x)))  # 20
        res = conv1  # 20
        conv2 = self.relu(self.bn2(self.conv2(conv1)))  # 8
        conv2_out = conv2  # 8
        conv2_out = conv2_out.permute(0, 2, 1)

        return self.dropout(self.fc21(conv2_out)), self.dropout(self.fc22(conv2_out))

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        fc3 = self.relu(self.fc3(z))
        fc3_out = fc3.permute(0, 2, 1)
        conv3 = self.relu(self.bn3(self.conv3(fc3_out)))
        out = self.conv4(conv3)
        out = out.permute(0, 2, 1)
        return out

    def forward(self, x):
        x = x.permute(0, 2, 1)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


if __name__ == '__main__':
    x = torch.randn(10, 30, 300)
    x = x.permute(0, 2, 1)
    vae = VAE()
    decode, mu, log = vae(x)
    print()
