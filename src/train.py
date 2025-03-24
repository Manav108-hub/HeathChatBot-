import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from data_module import QADataModule
from transformer_model import ChatTransformer
from transformers import AutoTokenizer
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

config_path = ROOT_DIR / "src" / "config" / "params.yaml"
data_path = ROOT_DIR / "data" / "raw" / "train_data_chatbot_processed.json"
model_path = ROOT_DIR / "models"

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
dm = QADataModule(
    data_path=str(data_path),
    tokenizer_name='bert-base-uncased',
    batch_size=config['training']['batch_size'],
    max_length=config['model']['max_seq_length']
)
dm.setup()
num_training_steps = len(dm.train_dataloader()) * config['training']['epochs']

model = ChatTransformer(config, dm.tokenizer, num_training_steps)

checkpoint_callback = ModelCheckpoint(
    dirpath=str(model_path),
    filename='chatbot-{epoch:02d}-{val_loss:.2f}',
    save_top_k=3,
    monitor='val_loss'
)

trainer = pl.Trainer(
    max_epochs=config['training']['epochs'],
    accumulate_grad_batches=config['training']['accumulation_steps'],
    callbacks=[checkpoint_callback],
    accelerator='auto'
)

trainer.fit(model, dm)