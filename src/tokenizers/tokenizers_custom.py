import os
import sys

# Ensure src namespace resolution
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from src.data_loader import DataLoaderFactory

def get_training_corpus(dataset, batch_size=1000):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]["sentence"]

def build_bpe_tokenizer():
    print("Initializing Data Architect's BPE Tokenization Pipeline...")
    
    # 1. Initialize tokenizer
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    
    # 2. Get dataset
    factory = DataLoaderFactory(dataset_name="sst2")
    train_dataset = factory.dataset['train']
    
    # 3. Setup trainer
    # We'll use a standard vocabulary size of 15,000 for this assignment per Phase 3 specifications
    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], vocab_size=15000)
    
    # 4. Train
    print("Training BPE tokenizer exclusively on SST-2 training corpus...")
    tokenizer.train_from_iterator(get_training_corpus(train_dataset), trainer=trainer)
    
    # 5. Save
    save_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'bpe_tokenizer.json')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tokenizer.save(save_path)
    
    # 6. Report
    vocab_size = tokenizer.get_vocab_size()
    print(f"BPE Tokenizer successfully trained and saved to {save_path}")
    print(f"EXACT_VOCAB_SIZE: {vocab_size}")

if __name__ == "__main__":
    build_bpe_tokenizer()
