import torch
import torch.nn as nn
import torch.nn.functional as F

class WordEncoding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(WordEncoding, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)

    def init_weights(self):
        d = self.d_model**(-0.5)
        self.embedding.weight.data.uniform_(-d, d)

    def forward(self,x):
        return self.embedding(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super(PositionalEncoding, self).__init__()
        self.pos_embedding = nn.Embedding(max_len,d_model)

    def init_weights(self):
        self.pos_embedding.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x):
        bs, seq = x.size()  # S :batch,  N : max_len
        pos = torch.arange(seq, dtype=torch.long, device=x.device).unsqueeze(0).expand((bs, seq))
        return self.pos_embedding(pos)

class SegmentEncoding(nn.Module):
    def __init__(self, type_num, d_model):
        super(SegmentEncoding, self).__init__()
        self.segment_embeddings = nn.Embedding(type_num, d_model) # type_token_num : 1 or 2, d_model : dimension

    def init_weights(self):
        self.segment_embeddings.weight.data.normal_(mean=0.02, std=0.02)

    def forward(self, type_input=None): # typ_input : 0, 1 으로 마스킹 되있는 input data, x.size == type_input.size
        return self.segment_embeddings(type_input)



class ScaledDotProduct(nn.Module):
    def __init__(self, dropout= 0.1, device=None):
        super(ScaledDotProduct, self).__init__()
        self.dropout = dropout
        self.device = device

    def forward(self, query, key, value, attn_mask=None):

        # (bs * h, seq_q, dimension/h) (128, 56, 64)
        # (bs * h, seq_k, dimension/h)
        # (bs * h, seq_v, dimension/h)
        # (bs, seq_k)
        bs_h, K, dimension_k = key.size()
        _, Q, _ = query.size()
        attn_output = torch.bmm(query, key.transpose(1,2))/dimension_k**0.5  # attn_output : (bs, seq_q, seq_k)
        # attn mask
        bs, _ = attn_mask.size()
        h = bs_h // bs
        assert bs_h == h*bs
        attn_mask = attn_mask.eq(0).float().repeat(h,1).unsqueeze(1).repeat(1,Q,1).contiguous()
        attn_output += attn_mask*(-1e-10)
        attn_output = F.softmax(attn_output, dim=-1)
        attn_output = F.dropout(attn_output, p=self.dropout)
        output = torch.bmm(attn_output,value) # output : (bs, seq_q, d_model)

        return output


class MultiheadAttention_In(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiheadAttention_In, self).__init__()
        self.num_heads = num_heads
        self.fc_q = nn.Linear(d_model, d_model, bias=False)
        self.fc_k = nn.Linear(d_model, d_model, bias=False)
        self.fc_v = nn.Linear(d_model, d_model, bias=False)

    def init_weights(self):
        self.fc_q.weight.data.normal_(mean=0.0, std=0.02)
        self.fc_k.weight.data.normal_(mean=0.0, std=0.02)
        self.fc_v.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, query, key, value): # (bs, seq, embedding_dim)
        bs, seq, d_model = query.size()
        head_dim = d_model// self.num_heads
        assert head_dim * self.num_heads == d_model
        q = self.fc_q(query)
        q = torch.cat(torch.chunk(q, self.num_heads, dim=2), dim=0).contiguous()

        k = self.fc_k(key)
        k = torch.cat(torch.chunk(k, self.num_heads, dim=2), dim=0).contiguous()

        v = self.fc_v(value)
        v = torch.cat(torch.chunk(v, self.num_heads, dim=2), dim=0).contiguous()
        return q, k, v

class MultiheadAttention_Out(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiheadAttention_Out, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.linear = nn.Linear(d_model, d_model)

    def init_weights(self):
        self.linear.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, attn_output):
        bs_h,seq,_ = attn_output.size()
        bs = bs_h // self.num_heads
        assert bs * self.num_heads == bs_h

        attn_output = torch.cat(torch.chunk(attn_output,self.num_heads, dim=0), dim=2).contiguous() # (bs,seq_q,embedding_dim)
        return self.linear(attn_output)