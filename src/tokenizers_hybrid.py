import os
import sys

# Ensure src namespace resolution
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tokenizers import Tokenizer, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from src.data_loader import DataLoaderFactory
from transformers import PreTrainedTokenizerFast

def get_training_corpus(dataset, batch_size=1000):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]["sentence"]

def build_hybrid_tokenizer():
    print("Initializing Data Architect's Hybrid Tokenization Pipeline...")
    
    # We use a standard BPE model with a mandatory character fallback.
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    
    factory = DataLoaderFactory(dataset_name="sst2")
    train_dataset = factory.dataset['train']
    
    print("Extracting base character alphabet to explicitly enforce Character-level fallback for rare words...")
    alphabet = set()
    for item in train_dataset:
        for char in item['sentence']:
            alphabet.add(char)
            
    # Restricted vocab_size (5000) forces BPE to only map highly frequent words to full tokens.
    # initial_alphabet guarantees that any unmerged rare word degrades iteratively down to individual characters, 
    # instead of emitting an [UNK] classification block.
    trainer = BpeTrainer(
        vocab_size=5000, 
        initial_alphabet=list(alphabet),
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    )
    
    print("Training Hybrid (Word+Char) tokenizer exclusively on SST-2 training corpus...")
    tokenizer.train_from_iterator(get_training_corpus(train_dataset), trainer=trainer)
    
    save_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'hybrid_tokenizer.json')
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
    
    print(f"\nHybrid Tokenizer saved to {save_path}")
    print(f"EXACT_VOCAB_SIZE: {vocab_size}")
    print(f"P99_SEQ_LENGTH: {p99_length}")

if __name__ == "__main__":
    build_hybrid_tokenizer()
