import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, BertConfig

class CoreModelManager:
    """Manages PyTorch architectures and specific embedding resizing requirements."""
    def __init__(self, num_labels: int = 2):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_labels = num_labels
        self.model = self._load_baseline()

    def _load_baseline(self):
        print("Loading standard bert-base-uncased baseline architecture...")
        model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased', 
            num_labels=self.num_labels
        )
        return model.to(self.device)

    def print_parameter_count(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        return total_params
