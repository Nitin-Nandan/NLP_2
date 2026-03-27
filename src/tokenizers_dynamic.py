import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tokenizers import Tokenizer, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from src.data_loader import DataLoaderFactory

def get_training_corpus(dataset, batch_size=1000):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]["sentence"]

def build_dynamic_tokenizer():
    print("Initializing Data Architect's Dynamic BPE Tokenization Pipeline (BPE-Dropout)...")
    
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    
    factory = DataLoaderFactory(dataset_name="sst2")
    train_dataset = factory.dataset['train']
    
    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], vocab_size=10000)
    
    print("Training BPE tokenizer for Dynamic Dropout execution...")
    tokenizer.train_from_iterator(get_training_corpus(train_dataset), trainer=trainer)
    
    save_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'dynamic_tokenizer.json')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tokenizer.save(save_path)
    
    vocab_size = tokenizer.get_vocab_size()
    print(f"\nDynamic Tokenizer saved to {save_path}")
    print(f"EXACT_VOCAB_SIZE: {vocab_size}")

if __name__ == "__main__":
    build_dynamic_tokenizer()
