import torch
from transformers import AutoTokenizer
from transformer_model import ChatTransformer
import yaml

class ChatBot:
    def __init__(self, model_path, config_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        # Estimate dummy num_training_steps (not needed for inference)
        dummy_steps = 100

        # Load model with required args
        self.model = ChatTransformer.load_from_checkpoint(
            model_path,
            config=config,
            tokenizer=tokenizer,
            num_training_steps=dummy_steps
        ).to(self.device)

        self.tokenizer = tokenizer
        self.model.eval()

    def generate_response(self, question, max_length=128):
        inputs = self.tokenizer(
            question,
            max_length=self.model.config['model']['max_seq_length'],
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(self.device)

        decoder_input = torch.full(
            (1, 1),
            self.tokenizer.cls_token_id,
            dtype=torch.long,
            device=self.device
        )

        generated_tokens = set()
        for _ in range(max_length):
            with torch.no_grad():
                logits = self.model(inputs.input_ids, decoder_input)

            next_token = logits[0, -1].argmax()

            # Early stop if repeated too much
            if next_token.item() in generated_tokens:
                break
            generated_tokens.add(next_token.item())

            decoder_input = torch.cat([decoder_input, next_token.unsqueeze(0).unsqueeze(0)], dim=1)

            if next_token == self.tokenizer.sep_token_id:
                break

        return self.tokenizer.decode(decoder_input[0], skip_special_tokens=True)


