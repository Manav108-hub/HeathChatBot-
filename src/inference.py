import torch
from transformers import AutoTokenizer
from transformer_model import ChatTransformer

class ChatBot:
    def __init__(self, model_path, config_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ChatTransformer.load_from_checkpoint(model_path).to(self.device)
        self.tokenizer = self.model.tokenizer
        self.model.eval()
        
    def generate_response(self, question, max_length=128):
        inputs = self.tokenizer(
            question,
            max_length=self.model.config['model']['max_seq_length'],
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        outputs = torch.full(
            (1, 1), 
            self.tokenizer.cls_token_id, 
            dtype=torch.long, 
            device=self.device
        )
        
        for _ in range(max_length):
            with torch.no_grad():
                logits = self.model(inputs.input_ids, outputs)
            
            next_token = logits[0, -1].argmax()
            outputs = torch.cat([outputs, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
            
            if next_token == self.tokenizer.sep_token_id:
                break
                
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)