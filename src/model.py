import torch
import torch.nn as nn
from transformers import BertModel, BertConfig


class BERTSingleLabelModel(nn.Module):
    def __init__(self, model_name, output_dim):
        super().__init__()
        config = BertConfig.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name, config=config)
        self.fc = nn.Linear(config.hidden_size, output_dim)
    
    def forward(self, ids, mask):
        last_hidden_state = self.model(ids, mask)['last_hidden_state']
        outputs = torch.mean(last_hidden_state, dim=1)
        logit = self.fc(outputs)

        return logit

class BERTMultiLabelModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        config = BertConfig.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name, config=config)
        self.fcT = nn.Linear(config.hidden_size, 4)
        self.fcN = nn.Linear(config.hidden_size, 4)
        self.fcM = nn.Linear(config.hidden_size, 2)
    
    def forward(self, ids, mask):
        last_hidden_state = self.model(ids, mask)['last_hidden_state']
        outputs = torch.mean(last_hidden_state, dim=1)
        logitT = self.fcT(outputs)
        logitN = self.fcN(outputs)
        logitM = self.fcM(outputs)

        return logitT, logitN, logitM
    