import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        if torch.__version__ >= '1.5.0':
            padding = 1
        else:
            padding = 2
        self.Conv_one = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=padding,
                                  padding_mode='circular')
        self.Normalized = nn.BatchNorm1d(c_in)
        self.Conv_activ = nn.ELU()
        self.Conv_maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.Conv_one(x.permute(0, 2, 1))
        x = self.Normalized(x)
        x = self.Conv_activ(x)
        x = self.Conv_maxPool(x)
        x = x.transpose(1, 2)
        return x

class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.en_attnlayers = nn.ModuleList(attn_layers)
        self.Normalized = None
        if conv_layers is not None:
            self.en_convlayers = nn.ModuleList(conv_layers)
        else:
            self.en_convlayers = None

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        en_attnlist = []
        if self.en_convlayers is not None:
            for en_attnlayers, en_convlayers in zip(self.en_attnlayers, self.en_convlayers):
                x, attn = en_attnlayers(x, attn_mask=attn_mask)
                x = en_convlayers(x)  # pooling后再减半，还是为了速度考虑
                en_attnlist.append(attn)
            x, attn = self.en_attnlayers[-1](x, attn_mask=attn_mask)
            en_attnlist.append(attn)
        else:
            for en_attnlayers in self.en_attnlayers:
                x, attn = en_attnlayers(x, attn_mask=attn_mask)
                en_attnlist.append(attn)

        if self.Normalized is not None:
            x = self.Normalized(x)

        return x, en_attnlist

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.En_attention = attention
        self.conv_one = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv_two = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.Normalized1 = nn.LayerNorm(d_model)
        self.Normalized2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        if activation == "relu":
            self.En_activation = F.relu
        else:
            self.En_activation = F.gelu

    def forward(self, x, attn_mask=None):

        en_x, en_attn = self.En_attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(en_x)
        y = x = self.Normalized1(x)

        # 1. 将输入张量的时间维度和特征维度进行交换
        transpose_y = y.transpose(-1, 1)
        # 2. 对转置后的张量进行一维卷积操作
        conv_y = self.conv_one(transpose_y)
        # 3. 对卷积结果进行激活函数处理，使用的是ReLU或GELU激活函数
        activation_y = self.En_activation(conv_y)
        # 4. 对激活后的张量进行随机失活操作
        dropout_activation_y = self.dropout(activation_y)
        # 5. 将随机失活后的张量赋值给变量 y
        y = dropout_activation_y

        # 1. 进行一维卷积操作
        conv_y = self.conv_two(y)
        # 2. 将卷积结果的最后一个维度和第一个维度进行交换
        transpose_conv_y = conv_y.transpose(-1, 1)
        # 3. 对转置后的张量进行随机失活操作
        dropout_transpose_conv_y = self.dropout(transpose_conv_y)
        # 4. 将随机失活后的张量赋值给变量 y
        y = dropout_transpose_conv_y

        return self.Normalized2(x + y), en_attn


class EncoderStack(nn.Module):
    def __init__(self, encoders, en_num):
        super(EncoderStack, self).__init__()
        self.encoderstack = nn.ModuleList(encoders)
        self.en_num = en_num

    def forward(self, x, attn_mask=None):
        x_stack = [];
        enstack_attnlist = []
        for i, encoder in zip(self.en_num, self.encoderstack):
            currinp_len = x.shape[1] // (2 ** i)
            curren_x_out, attn = encoder(x[:, -currinp_len:, :])
            x_stack.append(curren_x_out);
            enstack_attnlist.append(attn)
        x_stack = torch.cat(x_stack, -2)

        return x_stack, enstack_attnlist
