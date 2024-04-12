from transformers import BertModel
import torch.nn as nn

class FrozenLM(nn.Module):
    def __init__(self, pretrained_model_name='emilyalsentzer/Bio_ClinicalBERT'):
        super(FrozenLM, self).__init__()
        self.language_model = BertModel.from_pretrained(pretrained_model_name)
        for param in self.language_model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.language_model(input_ids=input_ids, attention_mask=attention_mask)
        sentence_representation = outputs.last_hidden_state[:, 0, :]
        return sentence_representation