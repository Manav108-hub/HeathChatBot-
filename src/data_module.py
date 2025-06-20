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
        # Filter for positive examples and create question-answer pairs
        self.pairs = [(q["short_question"], q["short_answer"])
                     for q in data if q["label"] == 1.0]
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        question, answer = self.pairs[idx]
        
        # Format as conversational text
        full_text = f"Question: {question} Answer: {answer}{self.tokenizer.eos_token}"
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        # For language modeling, labels are the same as input_ids
        labels = input_ids.clone()
        
        # Important: Set padding tokens to -100 so they're ignored in loss calculation
        # This must match the ignore_index in CrossEntropyLoss
        labels[attention_mask == 0] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
        
class QADataModule(pl.LightningDataModule):
    def __init__(self, data_path, tokenizer_name, batch_size, max_length):
        super().__init__()
        self.data_path = data_path
        self.tokenizer_name = tokenizer_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = min(4, os.cpu_count() or 1) if torch.cuda.is_available() else 0
        self.pin_memory = torch.cuda.is_available()
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Add special tokens if they don't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.bos_token is None:
            self.tokenizer.bos_token = self.tokenizer.eos_token
            
    def setup(self, stage=None):
        # Load data
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Create dataset
        dataset = QADataset(data, self.tokenizer, self.max_length)
        
        # Debug: Print a sample to verify tokenization
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"\nSample data point:")
            print(f"Input IDs shape: {sample['input_ids'].shape}")
            print(f"Input IDs min/max: {sample['input_ids'].min()}/{sample['input_ids'].max()}")
            print(f"Labels min/max: {sample['labels'].min()}/{sample['labels'].max()}")
            print(f"Tokenizer vocab size: {self.tokenizer.vocab_size}")
            
            # Decode to verify
            decoded_input = self.tokenizer.decode(sample['input_ids'], skip_special_tokens=False)
            print(f"Decoded input: {decoded_input[:100]}...")
        
        # Split into train and validation
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        
        self.train_data, self.val_data = torch.utils.data.random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # For reproducibility
        )
        
        print(f"Training samples: {len(self.train_data)}")
        print(f"Validation samples: {len(self.val_data)}")
    
    def train_dataloader(self):
        return DataLoader(
            self.train_data, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True  # Drop last incomplete batch
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_data, 
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False
        )