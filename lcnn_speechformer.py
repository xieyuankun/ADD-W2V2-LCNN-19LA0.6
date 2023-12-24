import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Function
import os
from torch import Tensor
import numpy as np
from torch.utils import data
from collections import OrderedDict
from torch.nn.parameter import Parameter
from pytorch_model_summary import summary
import math

class Speech_MSA(nn.Module):
    ''' Speech-based Multi-Head Self-Attention (Speech-MSA)

    Input dimension order is (batch_size, seq_len, input_dim).
    '''

    def __init__(self, embed_dim, num_heads, local_size, dropout=0., bias=True, overlap=False):
        super(Speech_MSA, self).__init__()
        self.qdim = embed_dim
        self.kdim = embed_dim
        self.vdim = embed_dim
        self.local_size = int(local_size)
        self.overlap = overlap  # overlap = True may have nondeterministic behavior.

        self.project_qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.project_out = nn.Linear(embed_dim, embed_dim, bias=bias)

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.scaling = float(self.head_dim) ** -0.5

    def get_overlap_segments(self, x: torch.Tensor, window_size: int):
        '''Get overlap segments for local attention.

        Args:
            x: Input sequence in shape (B, T, C).
            window_size: The needed length of the segment. Must be an odd number.
        '''
        # assert window_size % 2, f'window_size must be an odd number, but get {window_size}.'
        if not window_size % 2:
            window_size += 1  # window_size must be an odd number

        b, t, c = x.shape
        pad_len = (window_size - 1) // 2
        x = F.pad(x, (0, 0, pad_len, pad_len), value=0)

        stride = x.stride()
        out_shape = (b, t, window_size, c)
        out_stride = (stride[0], stride[1], stride[1], stride[2])

        return torch.as_strided(x, size=out_shape, stride=out_stride)

    def forward(self, x):
        '''
        Args:
            key_padding_mask: if provided, specified padding elements in the key will
                be ignored by the attention. This is an binary mask. When the value is True,
                the corresponding value on the attention layer will be filled with -inf.
            attn_mask: mask that prevents attention to certain positions. This is an additive mask
                (i.e. the values will be added to the attention layer).
        Shape:
            Inputs:
            - x: :math:`(B, T, E)` where T is the target sequence length, B is the batch size, E is
              the embedding dimension.
            - key_padding_mask: :math:`(B, S)`, ByteTensor, where B is the batch size, S is the source sequence length.
              3-D key_padding_mask with math:`(B, T, S)` is supported now, where T is the target sequence length.
            - attn_mask: :math:`(T, S)` where T is the target sequence length, S is the source sequence length.
        '''
        bsz, tgt_len = x.shape[:2]

        if self.local_size == -1:
            local_size = tgt_len
            global_attn = True
        else:
            local_size = self.local_size
            global_attn = False

        if not self.overlap:
            need_pad = tgt_len % local_size
            if need_pad:
                pad = local_size - need_pad
                x = F.pad(x, (0, 0, 0, pad), mode='constant', value=0)
                tgt_len += pad
        else:
            need_pad = 0

        Q, K, V = self.project_qkv(x).chunk(3, dim=-1)
        Q = Q * self.scaling
        Q = Q.transpose(0, 1).contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        K = K.transpose(0, 1).contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        V = V.transpose(0, 1).contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if (self.overlap) and (not global_attn):
            Q = Q.unsqueeze(dim=2)
            K = self.get_overlap_segments(K, window_size=local_size).transpose(-1, -2)
            V = self.get_overlap_segments(V, window_size=local_size)

            attn_output_weights = torch.matmul(Q, K)
            attn_output_weights = F.softmax(attn_output_weights, dim=-1)
            attn_output_weights = F.dropout(attn_output_weights, p=self.dropout, training=self.training)

            attn_output = torch.matmul(attn_output_weights, V).squeeze(dim=2)
        else:
            Q = Q.contiguous().view(-1, local_size, self.head_dim)
            K = K.contiguous().view(-1, local_size, self.head_dim)
            V = V.contiguous().view(-1, local_size, self.head_dim)

            src_len = K.size(1)
            attn_output_weights = torch.bmm(Q, K.transpose(1, 2))

            assert list(attn_output_weights.size()) == [bsz * self.num_heads * tgt_len / local_size, local_size,
                                                        src_len]

            attn_output_weights = F.softmax(attn_output_weights, dim=-1)
            attn_output_weights = F.dropout(attn_output_weights, p=self.dropout, training=self.training)

            attn_output = torch.bmm(attn_output_weights, V)

            assert list(attn_output.size()) == [bsz * self.num_heads * tgt_len / local_size, local_size, self.head_dim]
            attn_output = attn_output.view(bsz * self.num_heads, tgt_len, self.head_dim)

        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, self.embed_dim).transpose(0, 1)
        attn_output = self.project_out(attn_output)

        if need_pad:
            attn_output = attn_output[:, :-pad, :]

        return attn_output


def _get_activation_fn(activation: str):
    """ Returns the activation function corresponding to `activation` """
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "tanh":
        return torch.tanh
    elif activation == "linear":
        return lambda x: x
    else:
        raise RuntimeError("--activation-fn {} not supported".format(activation))


class SpeechFormerEncoder(nn.Module):
    def __init__(self, embed_dim, ffn_embed_dim=2048, local_size=0, num_heads=8, dropout=0.5, attention_dropout=0.5,
                 activation='relu', overlap=False):
        super().__init__()
        self.dropout = dropout
        self.activation_fn = _get_activation_fn(activation)

        self.attention = Speech_MSA(embed_dim, num_heads, local_size, attention_dropout, overlap=overlap)
        self.attention_layer_norm = nn.LayerNorm(embed_dim)

        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def add_position(self, x, position=None, mask=None):
        '''add position information to the input x

        x: B, T, C
        position: T, C
        mask: B, T
        '''
        if position is None:
            return x
        else:
            B, T = x.shape[:2]
            position = position[:T].unsqueeze(dim=0).repeat(B, 1, 1)  # -> B, T, C
            position = position * ((1 - mask.unsqueeze(-1).type_as(x))) if mask is not None else position
            return x + position

    def forward(self, x, x_position=None):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """
        residual = x
        x = self.add_position(x, x_position)

        x = self.attention(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.attention_layer_norm(x)

        residual = x
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.final_layer_norm(x)
        return x


class MaxFeatureMap2D(nn.Module):
    """ Max feature map (along 2D)
    MaxFeatureMap2D(max_dim=1)
    l_conv2d = MaxFeatureMap2D(1)
    data_in = torch.rand([1, 4, 5, 5])
    data_out = l_conv2d(data_in)
    Input:
    ------
    data_in: tensor of shape (batch, channel, ...)
    Output:
    -------
    data_out: tensor of shape (batch, channel//2, ...)
    Note
    ----
    By default, Max-feature-map is on channel dimension,
    and maxout is used on (channel ...)
    """

    def __init__(self, max_dim=1):
        super(MaxFeatureMap2D, self).__init__()
        self.max_dim = max_dim

    def forward(self, inputs):
        # suppose inputs (batchsize, channel, length, dim)

        shape = list(inputs.size())

        if self.max_dim >= len(shape):
            print("MaxFeatureMap: maximize on %d dim" % (self.max_dim))
            print("But input has %d dimensions" % (len(shape)))
            sys.exit(1)
        if shape[self.max_dim] // 2 * 2 != shape[self.max_dim]:
            print("MaxFeatureMap: maximize on %d dim" % (self.max_dim))
            print("But this dimension has an odd number of data")
            sys.exit(1)
        shape[self.max_dim] = shape[self.max_dim] // 2
        shape.insert(self.max_dim, 2)

        # view to (batchsize, 2, channel//2, ...)
        # maximize on the 2nd dim
        m, i = inputs.view(*shape).max(self.max_dim)
        return m

class LCNNSF(nn.Module):
    def __init__(self, num_nodes, enc_dim, nclasses=2):
        super(LCNNSF, self).__init__()
        self.num_nodes = num_nodes
        self.enc_dim = enc_dim
        self.nclasses = nclasses
        self.conv1 = nn.Sequential(nn.Conv2d(1, 64, (5, 5), 1, padding=(2, 2)),
                                   MaxFeatureMap2D(),
                                   nn.MaxPool2d((2, 2), (2, 2)))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, (1, 1), 1, padding=(0, 0)),
                                   MaxFeatureMap2D(),
                                   nn.BatchNorm2d(32, affine=False))
        self.conv3 = nn.Sequential(nn.Conv2d(32, 96, (3, 3), 1, padding=(1, 1)),
                                   MaxFeatureMap2D(),
                                   nn.MaxPool2d((2, 2), (2, 2)),
                                   nn.BatchNorm2d(48, affine=False))
        self.conv4 = nn.Sequential(nn.Conv2d(48, 96, (1, 1), 1, padding=(0, 0)),
                                   MaxFeatureMap2D(),
                                   nn.BatchNorm2d(48, affine=False))
        self.conv5 = nn.Sequential(nn.Conv2d(48, 128, (3, 3), 1, padding=(1, 1)),
                                   MaxFeatureMap2D(),
                                   nn.MaxPool2d((2, 2), (2, 2)))
        self.conv6 = nn.Sequential(nn.Conv2d(64, 128, (1, 1), 1, padding=(0, 0)),
                                   MaxFeatureMap2D(),
                                   nn.BatchNorm2d(64, affine=False))
        self.conv7 = nn.Sequential(nn.Conv2d(64, 64, (3, 3), 1, padding=(1, 1)),
                                   MaxFeatureMap2D(),
                                   nn.BatchNorm2d(32, affine=False))
        self.conv8 = nn.Sequential(nn.Conv2d(32, 64, (1, 1), 1, padding=(0, 0)),
                                   MaxFeatureMap2D(),
                                   nn.BatchNorm2d(32, affine=False))
        self.conv9 = nn.Sequential(nn.Conv2d(32, 64, (3, 3), 1, padding=[1, 1]),
                                   MaxFeatureMap2D(),
                                   nn.MaxPool2d((2, 2), (2, 2)))
        self.speechtransformer1 = SpeechFormerEncoder(embed_dim = 768, local_size = 0, overlap= True)
        #self.speechtransformer2 = SpeechFormerEncoder(embed_dim = 368, local_size = 0, overlap= True)
        self.out = nn.Sequential(nn.Dropout(0.7),
                                 nn.Linear(24576, 160),
                                 MaxFeatureMap2D(),
                                 nn.Linear(80, self.enc_dim))
        self.fc_mu = nn.Linear(enc_dim, nclasses) if nclasses >= 2 else nn.Linear(enc_dim, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = x.view(x.shape[0],32,-1)
        x = self.speechtransformer1(x)
        feat = torch.flatten(x, 1)
        feat = self.out(feat)
        out = self.fc_mu(feat)

        return feat, out
if __name__ == "__main__":
    # print(summary(se_res2net50_v1b(pretrained=False, num_classes=2), torch.randn((1, 1, 1, 750)), show_input=False))
    lfcc = torch.randn(128, 1, 128, 750)
    # res2net = SpeechFormer(input_dim=550, ffn_embed_dim=512, num_classes=2, hop=0.01, num_layers=[2, 2, 4, 4],
    #                        expand=[1, 1, 1, -1], num_heads=8, dropout=0.1, attention_dropout=0.1)
    lcnn = LCNNSF(128,256,nclasses=2)
    feat,out = lcnn(lfcc)
    print(feat.shape)
    # print(summary(lcnn(input_dim=550, ffn_embed_dim=2048, num_classes=2, hop=0.01, num_layers=[2, 2, 4, 4],
    #                        expand=[1, 1, 1, -1], num_heads=8, dropout=0.1, attention_dropout=0.1), torch.randn((32, 128, 750)), show_input=False))
    # print(feat.shape)
    # print(pred.shape)