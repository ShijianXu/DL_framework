import torch
import torch.nn as nn


class ECG_Text_CLIP(nn.Module):
    def __init__(self,
        tokenizer,
        text_encoder, 
        ecg_encoder
    ):
        super(ECG_Text_CLIP, self).__init__()
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.ecg_encoder = ecg_encoder

        self.num_params_text_encoder = sum(param.numel() for param in self.text_encoder.parameters())
        self.num_params_ecg_encoder = sum(param.numel() for param in self.ecg_encoder.parameters())
        
    def text_process(self, text_data):
        text_prompt = "The report of the ECG is that {text}"
        prompt_list = [text_prompt.replace("{text}", report) for report in text_data]
        tokens = self.tokenizer(prompt_list, padding=True, truncation=True, return_tensors='pt', max_length=100)
        return tokens

    def forward(self, ecg_data, tokens):
        input_ids, attention_mask = tokens['input_ids'], tokens['attention_mask']

        text_embedding = self.text_encoder(input_ids, attention_mask)
        ecg_embedding = self.ecg_encoder(ecg_data)
        return text_embedding, ecg_embedding
    
    def get_num_params(self):
        print(f"Text_encoder num of params: {self.num_params_text_encoder}")
        print(f"ECG_encoder num of params: {self.num_params_ecg_encoder}")
        return self.num_params_text_encoder + self.num_params_ecg_encoder
    
    def everything_to(self, device):
        pass

    def compute_loss(self, text_embedding, ecg_embedding, criterion):
        return criterion(text_embedding, ecg_embedding)

    def process_batch(self, batch, criterion, device):
        ecg_data, text_data = batch[0].float().to(device), batch[1]
        tokens = self.text_process(text_data).to(device)

        text_embedding, ecg_embedding = self.forward(ecg_data, tokens)

        loss = self.compute_loss(text_embedding, ecg_embedding, criterion)
        return loss



if __name__ == "__main__":
    from transformers import BertTokenizer, BertModel
    tokenizer = BertTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    text_encoder = BertModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')

    prompt = "The report of the ECG is that sinusrhythmus linkstyp qrs(t) abnormal    inferiorer infarkt     alter unbest."
    inputs = tokenizer(prompt, return_tensors='pt', max_length=100, padding=True, truncation=True)
    print(inputs)

    text_embedding = text_encoder(**inputs)
    last_hidden_state = text_embedding[0]                   # (batch_size, sequence_length, hidden_size)
    present = text_embedding.last_hidden_state[:, 0, :]     # (batch_size, hidden_size), the first token [CLS]'s embeddings, 
                                                            # which is the sentence representation

    print(last_hidden_state.shape)          # (1, 36, 768)
    print(present.shape)                    # (1, 768)
