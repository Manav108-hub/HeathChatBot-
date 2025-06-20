import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from data_module import QADataModule
from transformer_model import ChatTransformer
from transformers import AutoTokenizer
from pathlib import Path
import torch

# Set paths
ROOT_DIR = Path(__file__).resolve().parent.parent
config_path = ROOT_DIR / "src" / "config" / "params.yaml"
data_path = ROOT_DIR / "data" / "raw" / "train_data_chatbot_processed.json"
model_path = ROOT_DIR / "models"
log_path = ROOT_DIR / "logs"

# Create directories if they don't exist
model_path.mkdir(exist_ok=True)
log_path.mkdir(exist_ok=True)

# Load configuration
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

print("Configuration loaded:")
print(f"Model config: {config['model']}")
print(f"Training config: {config['training']}")

# Initialize tokenizer (changed to GPT-2)
tokenizer_name = 'gpt2'
print(f"Loading tokenizer: {tokenizer_name}")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

# Add padding token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("Added pad_token = eos_token")

print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

# Initialize data module
print("Setting up data module...")
dm = QADataModule(
    data_path=str(data_path),
    tokenizer_name=tokenizer_name,
    batch_size=config['training']['batch_size'],
    max_length=config['model']['max_seq_length']
)
dm.setup()

# Calculate number of training steps
num_training_steps = len(dm.train_dataloader()) * config['training']['epochs']
print(f"Total training steps: {num_training_steps}")

# Initialize model
print("Initializing model...")
model = ChatTransformer(config, dm.tokenizer, num_training_steps)

# Print model info
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Setup callbacks
checkpoint_callback = ModelCheckpoint(
    dirpath=str(model_path),
    filename='chatbot-{epoch:03d}-{val_loss:.3f}',
    save_top_k=3,
    monitor='val_loss',
    mode='min',
    save_last=True,
    verbose=True
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    mode='min',
    verbose=True
)

# Setup logger
logger = TensorBoardLogger(
    save_dir=str(log_path),
    name='chatbot_experiment'
)

# Initialize trainer
print("Initializing trainer...")
trainer = pl.Trainer(
    max_epochs=config['training']['epochs'],
    accumulate_grad_batches=config['training']['accumulation_steps'],
    callbacks=[checkpoint_callback, early_stopping],
    logger=logger,
    accelerator='auto',
    devices='auto',
    precision='16-mixed' if torch.cuda.is_available() else 32,
    gradient_clip_val=1.0,  # Gradient clipping
    log_every_n_steps=10,
    val_check_interval=0.5,  # Validate twice per epoch
    enable_progress_bar=True,
    enable_model_summary=True
)

# Print training info
print("\nTraining configuration:")
print(f"Epochs: {config['training']['epochs']}")
print(f"Batch size: {config['training']['batch_size']}")
print(f"Learning rate: {config['training']['learning_rate']}")
print(f"Accumulation steps: {config['training']['accumulation_steps']}")
print(f"Effective batch size: {config['training']['batch_size'] * config['training']['accumulation_steps']}")

# Start training
print("\nStarting training...")
try:
    trainer.fit(model, dm)
    print("\nTraining completed successfully!")
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")
    print(f"Best validation loss: {checkpoint_callback.best_model_score:.4f}")
    
except KeyboardInterrupt:
    print("\nTraining interrupted by user")
except Exception as e:
    print(f"\nTraining failed with error: {e}")
    raise

print("\nTraining script finished.")