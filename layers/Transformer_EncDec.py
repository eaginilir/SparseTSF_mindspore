from mindspore import ops
import mindspore.nn as nn


class ConvLayer(nn.Cell):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()

        self.downConv = nn.Conv1d(in_channels=c_in,
                                   out_channels=c_in,
                                   kernel_size=3,
                                   padding=2,
                                   pad_mode='pad')

        self.norm = nn.BatchNorm1d(c_in)

        self.activation = nn.ELU()

        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def construct(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        
        x = x.transpose(1, 2)
        
        return x


class EncoderLayer(nn.Cell):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm([d_model])
        self.norm2 = nn.LayerNorm([d_model])
        self.dropout = nn.Dropout(keep_prob=1-dropout)
        self.activation = nn.ReLU() if activation == "relu" else nn.GELU()

    def construct(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)

        y = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(0, 2, 1))))
        y = self.dropout(self.conv2(y).transpose(0, 2, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Cell):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.CellList(attn_layers)
        self.conv_layers = nn.CellList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def construct(self, x, attn_mask=None):
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Cell):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm((d_model,))
        self.norm2 = nn.LayerNorm((d_model,))
        self.norm3 = nn.LayerNorm((d_model,))
        self.dropout = nn.Dropout(keep_prob=1 - dropout)
        self.activation = ops.ReLU() if activation == "relu" else ops.GeLU()

    def construct(self, x, cross, x_mask=None, cross_mask=None):
        # Self-attention
        self_attn_output, _ = self.self_attention(x, x, x, attn_mask=x_mask)
        x = x + self.dropout(self_attn_output)
        x = self.norm1(x)

        # Cross-attention
        cross_attn_output, _ = self.cross_attention(x, cross, cross, attn_mask=cross_mask)
        x = x + self.dropout(cross_attn_output)
        x = self.norm2(x)

        # Feed-forward network
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(0, 2, 1))))
        y = self.dropout(self.conv2(y).transpose(0, 2, 1))

        return self.norm3(x + y)


class Decoder(nn.Cell):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.CellList(layers)
        self.norm = norm_layer
        self.projection = projection

    def construct(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        
        return x
