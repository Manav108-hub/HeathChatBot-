import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import Transformer
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
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(0)]
    
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
        
        self.transformer = Transformer(
            d_model=config['model']['d_model'],
            nhead=config['model']['nhead'],
            num_encoder_layers=config['model']['num_encoder_layers'],
            num_decoder_layers=config['model']['num_decoder_layers'],
            dim_feedforward=config['model']['dim_feedforward'],
            dropout=config['model']['dropout'],
            batch_first=False
        )
        
        self.fc_out = nn.Linear(
            config['model']['d_model'],
            tokenizer.vocab_size
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        
    def create_padding_mask(self, tensor):
        return (tensor == self.tokenizer.pad_token_id).bool()
        
    def create_causal_mask(self, size):
        mask = self.transformer.generate_square_subsequent_mask(size).to(self.device)
        return mask.bool()
        
    def forward(self, src, tgt):
        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1) 
        
        src_padding_mask = self.create_padding_mask(src.transpose(0, 1))
        tgt_padding_mask = self.create_padding_mask(tgt.transpose(0, 1))
        tgt_mask = self.create_causal_mask(tgt.size(0))
        
        src = self.embedding(src) * math.sqrt(self.config['model']['d_model'])
        tgt = self.embedding(tgt) * math.sqrt(self.config['model']['d_model'])
        
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)
        
        # Transform
        output = self.transformer(
            src, tgt,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask
        )
        
        # Project to vocabulary size
        output = self.fc_out(output)
        return output
    
    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        labels = batch['labels']
        
        outputs = self(input_ids, labels[:, :-1])
        outputs = outputs.transpose(0, 1)
        
        loss = self.criterion(
            outputs.reshape(-1, outputs.shape[-1]),
            labels[:, 1:].reshape(-1)
        )
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        labels = batch['labels']
        
        outputs = self(input_ids, labels[:, :-1])
        outputs = outputs.transpose(0, 1)
        
        loss = self.criterion(
            outputs.reshape(-1, outputs.shape[-1]),
            labels[:, 1:].reshape(-1)
        )
        self.log('val_loss', loss, prog_bar=True)
        
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
        
        return [optimizer], [scheduler]