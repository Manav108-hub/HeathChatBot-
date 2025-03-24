import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import AutoTokenizer
import json
import os

class QADataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pairs = [(q["short_question"], q["short_answer"])
                    for q in data if q["label"] == 1.0]
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        question, answer = self.pairs[idx]
        encoding = self.tokenizer(
            text=question,
            text_target=answer,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['labels'].squeeze()
        }
        
class QADataModule(pl.LightningDataModule):
    def __init__(self, data_path, tokenizer_name, batch_size, max_length):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = min(4, os.cpu_count() or 1) if torch.cuda.is_available() else 0
        self.pin_memory = torch.cuda.is_available()
        
    def setup(self, stage=None):
        with open(self.data_path) as f:
            data = json.load(f)
            dataset = QADataset(data, self.tokenizer, self.max_length)
            train_size = int(0.9 * len(dataset))
            val_size = len(dataset) - train_size
            self.train_data, self.val_data = torch.utils.data.random_split(
                dataset, [train_size, val_size]
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_data, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_data, 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )