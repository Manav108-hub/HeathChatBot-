import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import get_linear_schedule_with_warmup
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        return x + self.pe[:x.size(1)].unsqueeze(0)
    
class ChatTransformer(pl.LightningModule):
    def __init__(self, config, tokenizer, num_training_steps):
        super().__init__()
        self.tokenizer = tokenizer
        self.config = config
        self.num_training_steps = num_training_steps
        
        self.embedding = nn.Embedding(
            tokenizer.vocab_size,
            config['model']['d_model']
        )
        
        self.pos_encoder = PositionalEncoding(
            config['model']['d_model'],
            config['model']['max_seq_length']
        )
        
        # Use decoder-only architecture
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config['model']['d_model'],
            nhead=config['model']['nhead'],
            dim_feedforward=config['model']['dim_feedforward'],
            dropout=config['model']['dropout'],
            batch_first=True
        )
        
        self.transformer = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config['model']['num_decoder_layers']
        )
        
        self.fc_out = nn.Linear(
            config['model']['d_model'],
            tokenizer.vocab_size
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
    def create_causal_mask(self, size):
        # Create lower triangular mask (allows attending to previous positions)
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask
        
    def create_padding_mask(self, tensor):
        # Create mask for padding tokens
        return (tensor == self.tokenizer.pad_token_id)
        
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        
        # Create causal mask
        tgt_mask = self.create_causal_mask(seq_len).to(input_ids.device)
        
        # Create padding mask
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()
        else:
            key_padding_mask = self.create_padding_mask(input_ids)
        
        # Embeddings with scaling
        x = self.embedding(input_ids) * math.sqrt(self.config['model']['d_model'])
        x = self.pos_encoder(x)
        
        # For decoder-only, memory is the same as input
        memory = x
        
        # Transform
        output = self.transformer(
            tgt=x,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=key_padding_mask,
            memory_key_padding_mask=key_padding_mask
        )
        
        # Project to vocabulary size
        output = self.fc_out(output)
        return output
    
    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        # Debug: Check for any invalid values
        if torch.any(labels < -100) or torch.any(labels >= self.tokenizer.vocab_size):
            print(f"Warning: Invalid label values found. Min: {labels.min()}, Max: {labels.max()}")
            print(f"Vocab size: {self.tokenizer.vocab_size}")
        
        # Forward pass with input shifted by one position for language modeling
        outputs = self(input_ids[:, :-1], attention_mask[:, :-1])
        
        # Calculate loss (predict next token)
        targets = labels[:, 1:].contiguous()
        
        # Ensure targets are within valid range
        valid_targets = (targets >= 0) & (targets < self.tokenizer.vocab_size)
        if not torch.all(valid_targets | (targets == -100)):
            print(f"Warning: Some targets are out of range: {targets[~valid_targets & (targets != -100)]}")
        
        loss = self.criterion(
            outputs.reshape(-1, outputs.shape[-1]),
            targets.reshape(-1)
        )
        
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        # Forward pass
        outputs = self(input_ids[:, :-1], attention_mask[:, :-1])
        
        # Calculate loss
        targets = labels[:, 1:].contiguous()
        
        # Additional safety check
        if torch.any(targets >= self.tokenizer.vocab_size) and torch.any(targets != -100):
            invalid_mask = (targets >= self.tokenizer.vocab_size) & (targets != -100)
            print(f"Warning in validation: Invalid targets found: {targets[invalid_mask]}")
            # Clamp invalid values to -100
            targets[invalid_mask] = -100
        
        loss = self.criterion(
            outputs.reshape(-1, outputs.shape[-1]),
            targets.reshape(-1)
        )
        
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(self.config['training']['learning_rate']),
            weight_decay=float(self.config['training']['weight_decay'])
        )
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(self.config['training']['warmup_steps']),
            num_training_steps=self.num_training_steps
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }