import torch
import yaml
from transformers import AutoTokenizer
from transformer_model import ChatTransformer
from pathlib import Path

class ChatBot:
    def __init__(self, model_path, config_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model
        print(f"Loading model from: {model_path}")
        self.model = ChatTransformer.load_from_checkpoint(model_path, map_location=self.device)
        self.model.to(self.device)
        self.model.eval()
        
        # Get tokenizer from model
        self.tokenizer = self.model.tokenizer
        print(f"Tokenizer loaded: {self.tokenizer.__class__.__name__}")
        
    def generate_response(self, question, max_length=128, temperature=0.8, top_k=50, top_p=0.9):
        """
        Generate a response to a question using the trained model.
        
        Args:
            question (str): The input question
            max_length (int): Maximum length of generated response
            temperature (float): Sampling temperature (higher = more random)
            top_k (int): Top-k sampling parameter
            top_p (float): Top-p (nucleus) sampling parameter
        """
        # Format input like training data
        input_text = f"Question: {question} Answer:"
        
        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            return_tensors='pt',
            truncation=True,
            max_length=self.model.config['model']['max_seq_length'] - max_length
        ).to(self.device)
        
        # Generate response
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=inputs['input_ids'].shape[1] + max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=2
            )
        
        # Decode response
        full_response = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # Extract just the answer part
        if "Answer:" in full_response:
            answer = full_response.split("Answer:")[-1].strip()
        else:
            answer = full_response.strip()
            
        return answer
    
    def generate_simple(self, question, max_length=128):
        """
        Simple generation without advanced sampling (for debugging)
        """
        input_text = f"Question: {question} Answer:"
        
        inputs = self.tokenizer(
            input_text,
            return_tensors='pt',
            truncation=True,
            max_length=self.model.config['model']['max_seq_length'] - max_length
        ).to(self.device)
        
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        # Generate token by token
        for _ in range(max_length):
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask)
                logits = outputs[:, -1, :]  # Get last token logits
                
                # Get most likely next token
                next_token_id = torch.argmax(logits, dim=-1, keepdim=True)
                
                # Stop if we hit EOS token
                if next_token_id.item() == self.tokenizer.eos_token_id:
                    break
                
                # Append to input
                input_ids = torch.cat([input_ids, next_token_id], dim=1)
                attention_mask = torch.cat([attention_mask, torch.ones_like(next_token_id)], dim=1)
        
        # Decode full response
        full_response = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        
        # Extract answer
        if "Answer:" in full_response:
            answer = full_response.split("Answer:")[-1].strip()
        else:
            answer = full_response.strip()
            
        return answer

# Add generation method to ChatTransformer class
def generate(self, input_ids, attention_mask=None, max_length=128, temperature=1.0, 
             top_k=50, top_p=0.9, do_sample=True, pad_token_id=None, eos_token_id=None, 
             no_repeat_ngram_size=0):
    """
    Generate text using the model
    """
    self.eval()
    
    if pad_token_id is None:
        pad_token_id = self.tokenizer.pad_token_id
    if eos_token_id is None:
        eos_token_id = self.tokenizer.eos_token_id
    
    batch_size = input_ids.shape[0]
    seq_len = input_ids.shape[1]
    
    # Keep track of generated tokens
    generated = input_ids.clone()
    
    # Generate tokens one by one
    for _ in range(max_length):
        # Forward pass
        with torch.no_grad():
            outputs = self(generated, attention_mask)
            next_token_logits = outputs[:, -1, :]  # Get last position logits
            
            if do_sample:
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                
                # Top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = False
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample from the filtered distribution
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Stop if we hit EOS token
            if next_token.item() == eos_token_id:
                break
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Update attention mask if provided
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)
    
    return generated

# Add the generate method to ChatTransformer class
ChatTransformer.generate = generate

# Example usage
if __name__ == "__main__":
    # Example paths - adjust as needed
    model_path = "models/chatbot-best.ckpt"  # Path to your trained model
    
    try:
        # Initialize chatbot
        bot = ChatBot(model_path)
        
        # Test questions
        test_questions = [
            "What is artificial intelligence?",
            "How does machine learning work?",
            "What is the capital of France?",
            "Explain neural networks"
        ]
        
        print("Testing chatbot responses:")
        print("=" * 50)
        
        for question in test_questions:
            print(f"\nQuestion: {question}")
            try:
                response = bot.generate_response(question, max_length=50)
                print(f"Answer: {response}")
            except Exception as e:
                print(f"Error generating response: {e}")
                
    except Exception as e:
        print(f"Error initializing chatbot: {e}")
        print("Make sure you have a trained model checkpoint available.")