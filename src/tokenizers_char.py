import os
import sys

# Ensure src namespace resolution
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tokenizers import Tokenizer, Regex
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Split
from src.data_loader import DataLoaderFactory
from transformers import PreTrainedTokenizerFast

def get_training_corpus(dataset, batch_size=1000):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]["sentence"]

def build_char_tokenizer():
    print("Initializing Data Architect's Character-level Tokenization Pipeline...")
    
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    
    # By isolating every single character regex, BPE can NEVER merge cross-character boundaries.
    # This creates a mathematically exact character-level tokenizer from the standard BPE backbone.
    tokenizer.pre_tokenizer = Split(Regex("."), behavior="isolated")
    
    factory = DataLoaderFactory(dataset_name="sst2")
    train_dataset = factory.dataset['train']
    
    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], vocab_size=500)
    
    print("Training Char-level tokenizer exclusively on SST-2 training corpus...")
    tokenizer.train_from_iterator(get_training_corpus(train_dataset), trainer=trainer)
    
    save_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'char_tokenizer.json')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tokenizer.save(save_path)
    
    # 99th Percentile Sequence Length Analysis
    print("Executing Sequence Length Analysis...")
    lengths = []
    
    hf_tokenizer = PreTrainedTokenizerFast(tokenizer_file=save_path)
    hf_tokenizer.pad_token = "[PAD]"
    hf_tokenizer.unk_token = "[UNK]"
    
    for item in train_dataset:
        encoded = hf_tokenizer.encode(item['sentence'])
        lengths.append(len(encoded))
        
    lengths.sort()
    index = int(len(lengths) * 0.99)
    p99_length = lengths[index]
    vocab_size = tokenizer.get_vocab_size()
    
    print(f"\nChar Tokenizer saved to {save_path}")
    print(f"EXACT_VOCAB_SIZE: {vocab_size}")
    print(f"P99_SEQ_LENGTH: {p99_length}")

if __name__ == "__main__":
    build_char_tokenizer()
