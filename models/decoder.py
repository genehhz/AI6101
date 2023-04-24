import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(Decoder, self).__init__()
        self.de_layers = nn.ModuleList(layers)
        self.Normalized = norm_layer

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for de_layer in self.de_layers:
            x = de_layer(x, cross, x_mask=None, cross_mask=None)

        if self.Normalized is not None:
            x = self.Normalized(x)

        return x

class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.de_self_attention = self_attention
        self.endecross_attention = cross_attention
        self.conv_one = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv_two = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.Normalized_1 = nn.LayerNorm(d_model)
        self.Normalized_2 = nn.LayerNorm(d_model)
        self.Normalized_3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        if activation == "relu":
            self.En_activation = F.relu
        else:
            self.En_activation = F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):

        x_plus_attn = x + self.dropout(self.de_self_attention(x, x, x, attn_mask=x_mask)[0])
        normalized_output = self.Normalized_1(x_plus_attn)
        x = normalized_output

        dropout_output = self.dropout(self.endecross_attention(x, cross, cross, attn_mask=cross_mask)[0])
        x_plus_cross = x + dropout_output
        x = x_plus_cross

        y = x = self.Normalized_2(x)
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

        return self.Normalized_3(x+y)

