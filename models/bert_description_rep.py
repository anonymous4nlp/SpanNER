# encoding: utf-8


import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel

from models.classifier import MultiNonLinearClassifier, SingleLinearClassifier
from allennlp.modules.span_extractors import EndpointSpanExtractor
from torch.nn import functional as F
from tokenizers import BertWordPieceTokenizer
import numpy as np

class BertDescRep(BertPreTrainedModel):
    def __init__(self, config):
        super(BertDescRep, self).__init__(config)
        self.bert = BertModel(config)
        # self.tokenzier = tokenizer
        self.max_desc_length = 128

        # all_tokens, all_typeids, attention_mask = self.get_descrip()
        # self.desc_tokens = all_tokens
        # self.desc_typeids = all_typeids
        # self.desc_attention_mask = attention_mask
        self.linear1 = nn.Linear(config.hidden_size, 100)
        self.linear2 = nn.Linear(100,5)
        # self.args = args


    def forward(self,tokenizer):
        # for label- description
        cuda6 = torch.device('cuda:6')
        desc_tokens, desc_typeids, desc_attention_mask = self.get_descrip(tokenizer)
        desc_tokens = desc_tokens.to(cuda6)
        desc_typeids = desc_typeids.to(cuda6)
        desc_attention_mask = desc_attention_mask.to(cuda6)
        # print("desc_tokens: ", desc_tokens)
        # print("desc_typeids: ", desc_typeids)
        # print("desc_attention_mask: ", desc_attention_mask)
        desc_rep = self.bert(desc_tokens, token_type_ids=desc_typeids, attention_mask=desc_attention_mask)
        # print("desc_rep: ", desc_rep)
        # print("desc_token_rep.shape: ", desc_token_rep.shape)
        # print("desc_emb.shape: ", desc_emb.shape)
        desc_emb = desc_rep[1] # (n_class,dim)
        desc_emb = F.relu(desc_emb)
        desc_emb = self.linear1(desc_emb)
        # desc_emb = F.relu(desc_emb)
        # desc_emb = self.linear2(desc_emb)

        return desc_emb  # (n_class,dim)

    def get_descrip(self,tokenizer):
        label_desc = {"O": "the given span is not a named entity",
            "ORG": "organization entities are limited to named corporate, governmental, or other organizational entities.",
            "PER": "person entities are named persons or family.",
            "LOC": "location entities are the name of politically or geographically defined locations such as cities, provinces, countries, international regions, bodies of water, mountains, etc.",
            "MISC": "examples of miscellaneous entities include events, nationalities, products and works of art."
                      }
        label2idx = {"O": 0, "ORG": 1, "PER": 2, "LOC": 3, "MISC": 4}
        # label_desc = self.get_query()
        all_labels = []
        all_tokens = []
        all_typeids = []

        for label, desc in label_desc.items():
            all_labels.append(label)
            # descs.append(desc)
            desc_tokens = tokenizer.encode(desc, add_special_tokens=True)
            tokens = desc_tokens.ids  # subword index
            type_ids = desc_tokens.type_ids  # the split of two sentence on the subword-level, 0 for first sent, 1 for

            # padding to the max length.

            tokens = self.pad(tokens, 0)
            type_ids = self.pad(type_ids, 1)
            all_tokens.append(tokens)
            all_typeids.append(type_ids)
        print("all_labels: ", all_labels)
        all_tokens = torch.LongTensor(all_tokens)
        all_typeids = torch.LongTensor(all_typeids)


        attention_mask = (all_tokens != 0).long()

        return [all_tokens,all_typeids,attention_mask]

    def pad(self, lst, value=None, max_length=None):
        max_length = max_length or self.max_desc_length
        while len(lst) < max_length:
            lst.append(value)
        return lst
