import torch
import torch.nn as nn
from transformers import BertModel


class BaselineModel(nn.Module):
    def __init__(self, bert_path):
        super(BaselineModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(768, 52)

    def forward(self, text, mask):
        bert_out = self.bert(text, attention_mask=mask)[1]
        bert_out = self.dropout(bert_out)
        out_linear = self.fc(bert_out)
        output = torch.sigmoid(out_linear)
        return output
