import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, Dropout, LayerNorm

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.pos_embedding = nn.Embedding(max_len,d_model)

    def init_weights(self):
        self.pos_embedding.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x):
        S, N = x.size() # S :max_len,  N : batch
        pos = torch.arange(S, dtype=torch.long, device=x.device).unsqueeze(0).expand((N,S)).t()
        return self.pos_embedding(pos)


class TokenTypeEncoding(nn.Module):
    def __init__(self, type_token_num, d_model):
        super(TokenTypeEncoding, self).__init__()
        self.token_type_embeddings = nn.Embedding(type_token_num, d_model) # type_token_num : 1 or 2, d_model : dimension

    def init_weights(self):
        self.token_type_embeddings.weight.data.normal_(mean=0.02, std=0.02)

    def forward(self, seq_input, token_type_input):
        S, N = seq_input.size() # S : max_len, N : batch
        if token_type_input is None: # type_token_num ==1 (한 종류의 문장만 들어감)
            token_type_input = torch.zeros((S, N), dtype=torch.long, device=seq_input.device) # 0 인덱스의 임베딩값 가져옴.
        return self.token_type_embeddings(token_type_input)

class MultiheadAttentionInProjection(nn.Module):
    def __init__(self, embed_dim, num_heads, kdim=None, vdim=None):
        super(MultiheadAttentionInProjection, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim

        self.num_heads = num_heads
        self.fc_q = nn.Linear(embed_dim, embed_dim) # query
        self.fc_k = nn.Linear(embed_dim, self.kdim) # key
        self.fc_v = nn.Linear(embed_dim, self.vdim) # value

    def init_weights(self):
        self.fc_q = self.fc_q.weight.data.normal_(mean=0.0, std=0.02)
        self.fc_k = self.fc_k.weight.data.normal_(mean=0.0, std=0.02)
        self.fc_v = self.fc_v.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, query, key, value):
        print("query size {}".format(query.size()))
        tgt_len, bsz, embed_dim = query.size(0), query.size(1), query.size(2)

        head_dim = embed_dim // self.num_heads
        assert head_dim * self.num_heads == embed_dim # embed_dim must be diviable by num_heads

        q = self.fc_q(query)
        q = torch.cat(torch.chunk(q, self.num_heads, dim=2),dim=1).transpose(0,1)
         
        k = self.fc_k(key)
        k = torch.cat(torch.chunk(k, self.num_heads, dim=2), dim=1).transpose(0, 1)
        
        v = self.fc_v(value)
        v = torch.cat(torch.chunk(v, self.num_heads, dim=2), dim=1).transpose(0, 1)
        
        return q, k, v

class ScaledDotProduct(nn.Module):
    def __init__(self, dropout=0.1):
        super(ScaledDotProduct, self).__init__()
        self.dropout=dropout

    def forward(self, query, key, value, attn_mask=None):
        _, _, dimension_k = key.size()
        attn_output_weights = torch.bmm(query, key.transpose(1, 2))/(dimension_k **0.5) ## (bs*h , sequence_q, sequence_k)

        if attn_mask is not None:
            attn_output_weights += attn_mask #???

        attn_output_weights = nn.functional.softmax(attn_output_weights, dim=-1) # 각 q 에 대한 모든 k의 확률 값 계싼
        attn_output_weights = nn.functional.dropout(attn_output_weights, p=self.dropout)
        attn_output = torch.bmm(attn_output_weights, value)

        return attn_output

class MultiheadAttentionOutProjection(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiheadAttentionOutProjection, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.linear = nn.Linear(embed_dim, embed_dim)

    def init_weights(self):
        self.linear.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, attn_output): # attn_output (bs * h, sequence_k, embedding_dim/h)
        batch_heads, tgt_len = att_output.size(0), att_output.size(1)
        bsz = batch_heads //self.num_heads
        assert bsz * self.num_heads == batch_heads

        attn_output = torch.cat(torch.chunk(attn_outptut,self.num_heads, dim=0), dim=2).transpose(0,1) # (sequence_k, bs, embedding_dim)
        return self.linear(attn_output)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="gelu"):
        super(TransformerEncoderLayer, self).__init__()
        self.attn_in_proj = MultiheadAttentionInProjection(d_model, nhead)
        self.scaled_dot_product = ScaledDotProduct(dropout=dropout)
        self.attn_out_proj = MultiheadAttentionOutProjection(d_model, nhead)
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        if activation =="relu":
            self.activation = F.relu
        elif activation =="gelu":
            sefl.activation = F.gelu
        else:
            raise RuntimeError("only relu/gelu are supported, not {}".format(activation))

    def init_weights(self):
        self.attn_in_proj.init_weights()
        self.attn_out_proj.init_weights()
        self.linear1.weight.data.normal_(mean=0.0, std=0.02)
        self.linear2.weight.data.normal_(mean=0.0, std=0.02)
        self.norm1.bias.data.zero_()
        self.norm1.weight.data.fill_(1.0)
        self.norm2.bias.data.zero_()
        self.norm2.weight.data.fill_(1.0)

    def forward(self, src,src_mask=None, src_key_padding_mask=None):
        query, key, value = self.attn_in_proj(src, src, src)
        attn_out = self.scaled_dot_product(query, key, value, attn_mask=src_mask)
        src2 = self.attn_out_proj(attn_out)
        src = self.norm1(src + self.dropout1(src2))
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = ModuleList([copy.deepcopy(encoder_layer) for i in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def init_weights(self):
        for mod in self.layers:
            mod.init_weights()

    def forward(self, src, mask=None, src_key_padding_mask=None):
        for mod in self.layers:
            src = mod(src, src_mask=mask, src_key_padding_mask=src_key_padding_mask) ## src_key_padding_mask ???

        if self.norm is not None: ##????
            output = self.norm(src)

        return output

class BertEmbedding(nn.Module):
    def __init__(self, ntoken, ninp, dropout=0.5):
        super(BertEmbedding, self).__init__()
        self.ninp = ninp # embedding dimension
        self.ntoken = ntoken # vocab size
        self.dropout = dropout
        self.pos_embed = PositionalEncoding(ninp)
        self.emb = nn.Embedding(ntoken, ninp)
        self.tok_type_embed = TokenTypeEncoding(2, ninp)
        self.norm = LayerNorm(ninp)
        self.dropout= Dropout(dropout)

    def init_weights(self):
        self.embed.weight.data.normal_(mean=0.0, std=0.02)
        self.pos_embed.init_weights()
        self.tok_type_embed.init_weights()

    def forward(self, src, token_type_input):
        src = self.embed.forward(src)+self.pos_embed.forward(src)+self.tok_type_embed.forward(src, token_type_input)
        return self.dropout(self.norm(src))

class BertModel(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(BertModel, self).__init__()
        self.model_type = 'Transformer'
        self.bert_embed = BertEmbedding(ntoken, ninp)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.ninp = ninp
        self.init_weights()

    def init_weights(self):
        self.bert_embed.init_weights()
        self.transformer_encoder.init_weights()

    def forward(self, src, token_type_input):
        src = self.bert_embed(src, token_type_input)
        output = self.transformer_encoder(src)
        return output

class MLMTask(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        """
        :param ntoken: vocab_size
        :param ninp: bert hidden node num
        :param nhead: multi head num
        :param nhid: feedforward dimension num
        :param nlayers: encoder layers num
        :param dropout: dropout rate
        """
        ## contain transformer encoder pluss mlm head

        super(MLMTask, self).__init__()
        self.bert_model = BertModel(ntoken, ninp, nhead, nhid, nlayers, dropout=0.5)
        self.mlm_span = nn.Linear(ninp, ninp)
        self.activation = F.gelu()
        self.norm_layer = torch.nn.LayerNorm(ninp, eps=1e-12)
        self.mlm_head = nn.Linear(ninp, ntoken)

    def forward(self, src, token_type_input=None):
        src = src.transpose(0,1)
        output = self.bert_model(src, token_type_input)
        output = self.mlm_span(output)
        output = self.activation(output)
        output = self.norm_layer(output)
        output = self.mlm_head(output) # 전체 vocab에 대해서 예측하는건가...? softmax 사용 왜 안함 ...?
        return output

    class NextSentenceTask(nn.Module):
        ## pretrained bert model and linear layer
        def __init__(self, bert_model):
            super(NextSentencetask, self).__init__()
            self.bert_model = bert_model
            self.linear_layer = nn.Linear(bert_model.ninp, bert_model.ninp)
            self.ns_span = nn.Linear(bert_model.ninp,2)
            self.activation = nn.Tanh()

        def forward(self, src, token_type_input):
            src = src.transpose(0,1)
            output = self.bert_model(src, token_type_input)
            # cls token to classifier
            output = self.activation(self.linear_layer(output[0]))
            output = self.ns_span(output)
            return output

    class QuestionAnswerTask(nn.Module):
        # contain a pretrain BERT model and a linear layer
        def __init__(self, bert_model):
            super(QuestionAnswerTask, self).__init__()
            self.bert_model = bert_model
            self.activation = F.gelu()
            self.qa_span = nn.Linear(bert_model.ninp, 2)

        def forward(self, src, token_type_input):
            output = self.bert_model(src, token_type_input)
            output = output.transpose(0,1)
            output = self.activation(output)
            pos_output = self.qa_span(output)
            start_pos, end_pos = pos_output.split(1, dim=-1)
            start_pos = start_pos.squeeze(-1)
            end_pos = end_pos.squeeze(-1)
            return start_pos, end_pos













