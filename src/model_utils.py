import torch
import torch.nn as nn
from transformers import BertForSequenceClassification

class CoreModelManager:
    """Manages PyTorch architectures and specific embedding resizing requirements."""
    def __init__(self, num_labels: int = 2, new_vocab_size: int = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_labels = num_labels
        self.model = self._load_baseline(new_vocab_size)

    def _load_baseline(self, new_vocab_size):
        print("Loading standard bert-base-uncased baseline architecture...")
        model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased', 
            num_labels=self.num_labels
        )
        if new_vocab_size is not None:
            print(f"Resizing token embeddings to {new_vocab_size}...")
            model.resize_token_embeddings(new_vocab_size)
            
        return model.to(self.device)

    def print_parameter_count(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        return total_params

    def get_differential_optimizer_params(self, base_lr=2e-5, new_layer_lr=1e-3):
        print("Configuring differential learning rates...")
        word_embeddings_params = []
        encoder_params = []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if 'word_embeddings' in n:
                word_embeddings_params.append(p)
            else:
                encoder_params.append(p)
                
        return [
            {"params": encoder_params, "lr": base_lr},
            {"params": word_embeddings_params, "lr": new_layer_lr}
        ]
