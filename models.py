import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertModel


class Attn_MLP(nn.Module):
    def __init__(self, args):
        super(Attn_MLP, self).__init__()
        self.fc1 = nn.Linear(args.input_dim, args.hidden_layer[2], bias=True)
        self.cls = nn.Linear(args.hidden_layer[2], args.cls_type, bias=True)
        self.reg_1 = nn.Linear(args.hidden_layer[2], 1, bias=True)
        self.reg_2 = nn.Linear(args.hidden_layer[2], 1, bias=True)
        self.reg_3 = nn.Linear(args.hidden_layer[2], 1, bias=True)
        self.aux_cls_1 = nn.Linear(args.hidden_layer[2], 2, bias=True)
        self.aux_cls_2 = nn.Linear(args.hidden_layer[2], 2, bias=True)
        self.attention_layer = nn.Linear(args.input_dim, args.input_dim)
        self.dropout = nn.Dropout(args.dropout)
        self.attn_dropout = nn.Dropout(args.attn_dropout)

    def forward(self, x):
        attention_weights = F.softmax(self.attn_dropout(self.attention_layer(x)), dim=1)
        weighted_x = x * attention_weights
        weighted_x = self.dropout(F.gelu(self.fc1(weighted_x)))
        cls_out = self.cls(weighted_x)
        reg_out_1 = self.reg_1(weighted_x)
        reg_out_2 = self.reg_2(weighted_x)
        reg_out_3 = self.reg_3(weighted_x)
        aux_cls_out_1 = self.aux_cls_1(weighted_x)
        aux_cls_out_2 = self.aux_cls_2(weighted_x)
        return cls_out, aux_cls_out_1, aux_cls_out_2, reg_out_1, reg_out_2, reg_out_3, attention_weights

