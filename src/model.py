import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, Dropout, LayerNorm
from src.module import *

class EncoderLayer(nn.Module):

    def __init__(self, d_model, num_heads, dim_feedforward=2048, dropout=0.1,device=None):
        super(EncoderLayer, self).__init__()
        self.attn_in = MultiheadAttention_In(d_model, num_heads)
        self.scaled_dot = ScaledDotProduct(dropout=dropout, device=device)
        self.attn_out = MultiheadAttention_Out(d_model, num_heads)
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = nn.GELU()

    def init_weights(self):
        self.attn_in.init_weights()
        self.attn_out.init_weights()
        self.linear1.weight.data.normal_(mean=0.0, std=0.02)
        self.linear2.weight.data.normal_(mean=0.0, std=0.02)
        self.norm1.bias.data.zero_()
        self.norm1.weight.data.fill_(1.0)
        self.norm2.bias.data.zero_()
        self.norm2.weight.data.fill_(1.0)

    def forward(self, input, input_mask=None):
        query, key, value = self.attn_in(input, input, input)
        attn_out = self.scaled_dot(query, key, value, input_mask)
        out1 = self.attn_out(attn_out)
        out = self.norm1(input+self.dropout1(out1))
        out2= self.linear2(self.dropout(self.activation(self.linear1(out))))
        out = self.norm2(out1+self.dropout2(out2))
        return out

class TransformerEncoder(nn.Module):
    def __init__(self, layer, num_layers, device):
        super(TransformerEncoder, self).__init__()
        self.layers = ModuleList([copy.deepcopy(layer) for i in range(num_layers)])
        self.layers.to(device)
        self.num_layers = num_layers

    def init_weights(self):
        for layer in self.layers:
            layer.init_weights()

    def forward(self, input, mask=None):
        for layer in self.layers:
            output = layer(input, input_mask=mask)
        return output

class BertEmbedding(nn.Module):
    def __init__(self, vocab, dimension, device,dropout=0.1):
        super(BertEmbedding, self).__init__()
        self.dim = dimension # embedding dimension
        self.vocab = vocab # vocab size
        self.dropout = dropout
        self.pos_embed = PositionalEncoding(dimension)
        self.word_embed = WordEncoding(vocab, dimension)
        self.seg_embed = SegmentEncoding(3, dimension) # 2 check!!! 
        self.norm = LayerNorm(dimension)
        self.dropout= Dropout(dropout)

    def init_weights(self):
        self.word_embed.init_weights()
        self.pos_embed.init_weights()
        self.seg_embed.init_weights()

    def forward(self, src, type_input):
        src = self.word_embed.forward(src)+self.pos_embed.forward(src)+self.seg_embed.forward(type_input)
        return self.dropout(self.norm(src))


class BertModel(nn.Module):
    def __init__(self, vocab, dimension, num_heads, dim_feedforward, num_layers, dropout, device):

        super(BertModel, self).__init__()
        self.bert_embed = BertEmbedding(vocab, dimension, device)
        encoder = EncoderLayer(dimension, num_heads, dim_feedforward, dropout, device)
        self.transformer_encoder = TransformerEncoder(encoder, num_layers, device)
        self.dimension = dimension
        self.init_weights()

    def init_weights(self):
        self.bert_embed.init_weights()
        self.transformer_encoder.init_weights()

    def forward(self, src, type_input):
        output = self.bert_embed(src, type_input)
        output = self.transformer_encoder(input=output, mask=src)
        return output


class BERT_PretrainModel(nn.Module):
    def __init__(self, config, args, device):
        
        super(BERT_PretrainModel, self).__init__()
        vocab = 30001
        dimension = config.model.hidden
        num_heads = config.model.num_head
        num_layers = config.pretrain.num_layers
        dim_feedforward = config.model.dim_feedforward
        dropout = config.model.d_rate

        self.bert_model = BertModel(vocab, dimension, num_heads, dim_feedforward, num_layers, dropout, device)

        # MLMTask
        self.mlm_span = nn.Linear(dimension, dimension)
        self.activation = nn.GELU()
        self.norm_layer = torch.nn.LayerNorm(dimension, eps=1e-12)
        self.mlm_head = nn.Linear(dimension, vocab, bias=False)

    def forward(self, src, token_type_input=None):
        output = self.bert_model(src, token_type_input)

        # masked token prediction
        mlm_output = self.mlm_span(output)
        mlm_output = self.activation(mlm_output)
        mlm_output = self.norm_layer(mlm_output)
        mlm_output = self.mlm_head(mlm_output)

        return mlm_output



class Next_Sentence_Prediction(nn.Module): # use pretrained model
    def __init__(self, config, args, device):
        super(BERT, self).__init__()
        dimension = config.model.hidden

        self.bert = BERT_PretrainModel(ocnfig, args, device)
        self.ns_span = nn.Linear(dimension, dimension)
        self.ns_head = nn.Linear(dimension,2)
        self.activation = nn.Tanh()

    def forward(self, src, token_type_input=None):
        output = self.bert(src, token_type_input)
        # next sentence prediction
        ns_output = self.activation(self.ns_span(output[:, 0, :]))
        ns_output = self.ns_head(ns_output)
        return ns_output


# class BERT_PretrainModel(nn.Module):
#     def __init__(self, config, args, device):
#         super(BERT_PretrainModel, self).__init__()
#         vocab = 30001
#         dimension = config.model.hidden
#         num_heads = config.model.num_head
#         num_layers = config.pretrain.num_layers
#         dim_feedforward = config.model.dim_feedforward
#         dropout = config.model.d_rate
#
#         self.bert_model = BertModel(vocab, dimension, num_heads, dim_feedforward, num_layers, dropout, device)
#
#         # MLMTask
#         self.mlm_span = nn.Linear(dimension, dimension)
#         self.activation = nn.GELU()
#         self.norm_layer = torch.nn.LayerNorm(dimension, eps=1e-12)
#         self.mlm_head = nn.Linear(dimension, vocab)
#
#         # NSTask
#         self.ns_span = nn.Linear(dimension, dimension)
#         self.ns_head = nn.Linear(dimension, 2)
#         self.activation = nn.Tanh()
#
#     def forward(self, src, token_type_input=None):
#         output = self.bert_model(src, token_type_input)
#
#         # masked token prediction
#         mlm_output = self.mlm_span(output)
#         mlm_output = self.activation(mlm_output)
#         mlm_output = self.norm_layer(mlm_output)
#         mlm_output = self.mlm_head(mlm_output)
#
#         # next sentence prediction
#         ns_output = self.activation(self.ns_span(output[:, 0, :]))
#         ns_output = self.ns_head(ns_output)
#
#         return mlm_output, ns_output














