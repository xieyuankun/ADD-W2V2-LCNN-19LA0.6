import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def create_PositionalEncoding(input_dim, max_seq_len=2000):
    position_encoding = np.array([
        [pos / np.power(10000, 2.0 * (j // 2) / input_dim) for j in range(input_dim)]
        for pos in range(max_seq_len)])

    position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
    position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])

    position_encoding = torch.from_numpy(position_encoding.astype(np.float32))
    position_encoding = nn.Parameter(position_encoding, requires_grad=False)

    return position_encoding


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
    def __init__(self, embed_dim, ffn_embed_dim=2304, local_size=0, num_heads=8, dropout=0.1, attention_dropout=0.1,
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


def statistical_information(hop=0.01):  # unit: second
    hop *= 1000
    Merge = [50, 250, 1000]
    Locals = [50, 400, 2000]
    Merge = [s // hop for s in Merge]
    Locals = [l // hop for l in Locals]
    Merge.append(-1)  # -1 means global
    Locals.append(-1)
    return Merge, Locals


class MergeBlock(nn.Module):
    ''' Merge features between tow phases.

        The number of tokens is decreased while the dimension of token is increased.
    '''

    def __init__(self, in_channels, merge_scale: int, expand: int = 2):
        super().__init__()

        out_channels = in_channels * expand
        self.MS = int(merge_scale)
        self.pool = nn.AdaptiveAvgPool2d((1, in_channels))
        self.fc = nn.Linear(in_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        ms = T if self.MS == -1 else self.MS

        need_pad = T % ms
        if need_pad:
            pad = ms - need_pad
            x = F.pad(x, (0, 0, 0, pad), mode='constant', value=0)
            T += pad

        x = x.view(B, T // ms, ms, C)
        x = self.pool(x).squeeze(dim=-2)
        x = self.norm(self.fc(x))

        return x


def make_layers(Merge, Locals, input_dim, ffn_embed_dim, num_layers, num_heads, dropout, attention_dropout, expand,
                use_position=True):
    layers = []
    last_merge = 1
    for ms, l, exp, num in zip(Merge, Locals, expand, num_layers):
        _l = l // last_merge if l != -1 else -1
        _ms = ms // last_merge if ms != -1 else -1
        module1 = SpeechFormerBlock(num, input_dim, ffn_embed_dim, _l, num_heads, dropout, attention_dropout,
                                    use_position=use_position)
        layers += [module1]

        if _ms != -1:
            module2 = MergeBlock(input_dim, _ms, expand=exp)
            layers += [module2]
            input_dim *= exp
            ffn_embed_dim *= exp

        last_merge = ms
        if use_position:
            use_position = not use_position  # only the first layer use positional embedding.
    return nn.Sequential(*layers)


class SpeechFormerBlock(nn.Module):
    def __init__(self, num_layers, embed_dim, ffn_embed_dim=2304, local_size=0, num_heads=8, dropout=0.1,
                 attention_dropout=0.1, activation='relu', use_position=False):
        super().__init__()
        self.position = create_PositionalEncoding(embed_dim) if use_position else None
        self.input_norm = nn.LayerNorm(embed_dim)
        self.local = int(local_size)
        self.layers = nn.ModuleList([SpeechFormerEncoder(embed_dim, ffn_embed_dim, local_size, num_heads, dropout,
                                                         attention_dropout, activation, overlap=True) for _ in
                                     range(num_layers)])

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        output = self.input_norm(x)

        for layer in self.layers:
            output = layer(output, self.position)

        return output


class SpeechFormer(nn.Module):
    def __init__(self, input_dim, ffn_embed_dim, num_layers, num_heads, hop, num_classes, expand, dropout=0.1,
                 attention_dropout=0.1, device='cuda', **kwargs):
        super(SpeechFormer,self).__init__()

        self.input_dim = input_dim // num_heads * num_heads
        Merge, Locals = statistical_information(hop)
        assert isinstance(num_layers, list)

        self.layers = make_layers(Merge, Locals, self.input_dim, ffn_embed_dim, num_layers, num_heads, dropout,
                                  attention_dropout, expand)

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        dim_expand = 1
        expand.pop()
        for e in expand:
            dim_expand *= e

        classifier_dim = self.input_dim * dim_expand
        self.classifier = nn.Sequential(
            nn.Linear(classifier_dim, classifier_dim // 2),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(classifier_dim // 2, classifier_dim // 4),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(classifier_dim // 4, num_classes),
        )

    def forward(self, x):
        if self.input_dim != x.shape[-1]:
            x = x[:, :, :self.input_dim]
        
        #x = self.layers(x).squeeze(dim=1)
        print(x.shape)
        x = self.avgpool(x.transpose(-1, -2)).squeeze(dim=-1)
        print(x.shape)
        pred = self.classifier(x)

        return x,pred


if __name__ == "__main__":
    # print(summary(se_res2net50_v1b(pretrained=False, num_classes=2), torch.randn((1, 1, 1, 750)), show_input=False))
    lfcc = torch.randn(64, 768, 201)
    res2net = SpeechFormer(input_dim=201, ffn_embed_dim=512, num_classes=2, hop=0.01, num_layers=[2, 2, 4, 4],
                           expand=[1, 1, 1, -1], num_heads=8, dropout=0.1, attention_dropout=0.1)
    feat = res2net(lfcc)
    feat_optimizer = torch.optim.Adam(res2net.parameters())
    print(feat)
    print(feat.shape)