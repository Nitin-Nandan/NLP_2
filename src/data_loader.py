import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

class SST2Dataset(Dataset):
    """PyTorch Dataset for SST-2 handling standard or custom tokenizers."""
    def __init__(self, data, tokenizer, max_length: int = 128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['sentence']
        label = item['label']

        # Tokenize - dynamically handles any standard HuggingFace/tokenizers format
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class DataLoaderFactory:
    """Factory object to govern the Hugging Face datasets and DataLoader instantiation."""
    def __init__(self, dataset_name: str = "sst2"):
        self.dataset_name = dataset_name
        self._load_hf_dataset()

    def _load_hf_dataset(self):
        print(f"Downloading {self.dataset_name} dataset via Hugging Face...")
        # SST-2 is part of the glue benchmark suite
        self.dataset = load_dataset("glue", self.dataset_name)
        print("Dataset loaded successfully.")

    def get_dataloaders(self, tokenizer, batch_size: int = 32, max_length: int = 128):
        """Creates formatted PyTorch DataLoaders for train and validation splits."""
        train_ds = SST2Dataset(self.dataset['train'], tokenizer, max_length)
        val_ds = SST2Dataset(self.dataset['validation'], tokenizer, max_length)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader
