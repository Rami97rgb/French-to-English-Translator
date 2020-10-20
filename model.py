import torch
import torch.nn as nn
import torch.nn.functional as F

#execute on gpu if available, else on cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#multi-head attention block
class SelfAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, mask = False):
        super().__init__()
        #split the embeded word into multiple heads that run in parallel
        self.num_heads = num_heads
        self.head_size = embed_dim // num_heads
        #optionally use mask to hide the next part of the sentence (used for the decoder)
        self.mask = mask
        assert(num_heads * self.head_size == embed_dim, "embed dim and number of heads aren't compatible")
        #linear layers
        self.lin1 = nn.Linear(self.head_size, self.head_size, bias = False)
        self.lin2 = nn.Linear(embed_dim, embed_dim, bias = False)

    def forward(self, queries, keys, values):
        b, t2 = queries.size(0), queries.size(1)
        h = self.num_heads
        d = self.head_size
        t = values.size(1)
        
        #receive queries, keys and values and pass through linear layers
        queries = self.lin1(queries.reshape(b, t2, h , d))
        keys = self.lin1(keys.reshape(b, t, h , d))
        values = self.lin1(values.reshape(b, t, h , d))
        
        #scaled dot product attention
        queries = queries.transpose(1,2).reshape(b * h, t2, d)
        keys = keys.transpose(1,2).reshape(b * h, t, d)
        matmul1 = torch.bmm(queries, keys.transpose(1,2))
        scale = (matmul1 / (d ** (1/2)))

        if self.mask:
            indices = torch.triu_indices(t2, t, offset = 1)
            scale[:, indices[0], indices[1]] = float('-inf')

        soft = F.softmax(scale, dim=2)
        values = values.transpose(1,2).reshape(b *  h, t, d)
        matmul2 = torch.bmm(soft, values)

        #concat and linear layer
        out = self.lin2(matmul2.reshape(b, h, t2, d).transpose(1, 2).reshape(b, t2, h * d))

        return out

#transformer block
class TransformerBlock(nn.Module):

    def __init__(self, embed_dim, num_heads, expansion_size, drop = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        #use multi-head attention block
        self.attentionblock = SelfAttention(self.embed_dim, self.num_heads)

        #feed forward block
        self.ff = nn.Sequential(nn.Linear(embed_dim, expansion_size),
                                nn.ReLU(),
                                nn.Linear(expansion_size, embed_dim))
        
        #normalization layer
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(drop)

    def forward(self, queries, keys, values):

        #attention
        mha = self.attentionblock(queries, keys, values)

        #add & norm 1 block
        addnorm1 = self.norm(queries + self.dropout(mha))

        #feed forward
        feedfwd = self.ff(addnorm1)

        #add & norm block
        addnorm2 = self.norm(addnorm1 + self.dropout(feedfwd))

        return addnorm2


#encoder block
class Encoder(nn.Module):

    def __init__(self, embed_dim, num_heads, expansion_size, num_layers, dict_size, max_len=100, drop = 0.1):
        super().__init__()
        self.num_layers = num_layers

        #word embedding
        self.embed = nn.Embedding(dict_size, embed_dim)

        #positional encoding
        self.embed_pos = nn.Embedding(max_len, embed_dim)
        self.dropout = nn.Dropout(drop)

        #tranformer block
        self.transformblock = TransformerBlock(embed_dim, num_heads, expansion_size, drop)

    def forward(self, inp):
        b, t = inp.size()

        #combine tranformer block with embedding and positional encoding
        input_embed = self.embed(inp)
        pos = torch.arange(t).repeat(b, 1).to(device)
        pos_embed = self.embed_pos(pos)
        input_pos = self.dropout(pos_embed + input_embed)
        out = input_pos

        #make 'n' layers of encoder block
        for i in range(self.num_layers):
            out = self.transformblock.forward(out, out, out)

        return out


#decoder block
class Decoder(nn.Module):

    def __init__(self, embed_dim, num_heads, expansion_size, num_layers, dict_size, max_len=100, drop = 0.1):
        super().__init__()
        self.num_layers = num_layers

        #word embedding
        self.embed = nn.Embedding(dict_size, embed_dim)

        #positional encoding
        self.embed_pos = nn.Embedding(max_len, embed_dim)
        self.dropout = nn.Dropout(drop)

        #attention block with mask
        self.attentionblock = SelfAttention(embed_dim, num_heads, mask = True)

        #norm layer
        self.norm = nn.LayerNorm(embed_dim)

        #transformer block
        self.transformblock = TransformerBlock(embed_dim, num_heads, expansion_size, drop)
        self.lin = nn.Linear(embed_dim, dict_size)

    def forward(self, inp, inp_enc):

        #same architecture as the encoder with masked attention and addnorm block added before transformer block
        b, t = inp.size()
        input_embed = self.embed(inp)
        pos = torch.arange(t).repeat(b, 1).to(device)
        pos_embed = self.embed_pos(pos)
        input_pos = self.dropout(pos_embed + input_embed)
        out = input_pos

        for i in range(self.num_layers):
            out = self.norm(out + self.dropout(self.attentionblock.forward(out, out, out)))
            out = self.transformblock.forward(out, inp_enc, inp_enc)

        out =  self.dropout(self.lin(out))

        return out

#complete transformer
class Transformer(nn.Module):

    def __init__(self, embed_dim, num_heads, expansion_size, num_layers, src_dict_size, trg_dict_size, max_len=100, drop = 0.1):
        super().__init__()
        self.enc = Encoder(embed_dim, num_heads, expansion_size, num_layers, src_dict_size, max_len, drop)
        self.dec = Decoder(embed_dim, num_heads, expansion_size, num_layers, trg_dict_size, max_len, drop)

    def forward(self, inp, out):
        
        #combine encoder with decoder
        return self.dec.forward(out, self.enc.forward(inp))