import torch
import time
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from transformers import get_linear_schedule_with_warmup, BertTokenizerFast
from torch.optim import AdamW

from src.data_loader import DataLoaderFactory
from src.model_utils import CoreModelManager
from src.training_engine import DualLogger, train_epoch, evaluate

if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    sys.stdout = DualLogger(os.path.join("results", "train_output.txt"))
    print("Initiating Phase 2 Baseline Execution via ML-Scientist")
    
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    factory = DataLoaderFactory(dataset_name="sst2")
    train_loader, val_loader = factory.get_dataloaders(tokenizer, batch_size=32, max_length=128)
    
    manager = CoreModelManager(num_labels=2)
    device = manager.device
    model = manager.model
    params_count = manager.print_parameter_count()
    
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    EPOCHS = 3
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps*0.1), num_training_steps=total_steps)
    scaler = torch.amp.GradScaler('cuda' if 'cuda' in str(device) else 'cpu', enabled=('cuda' in str(device)))
    
    start_time = time.time()
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, scaler)
        print(f"Epoch {epoch+1} Training Loss: {train_loss:.4f}")
        
    print("Evaluating Baseline-WordPiece Model...")
    val_acc, val_f1 = evaluate(model, val_loader, device)
    total_time = time.time() - start_time
    print(f"Baseline Training complete! Acc: {val_acc:.4f} | F1: {val_f1:.4f} | Time: {total_time:.2f}s")
    print(f"BASELINE_FINAL_METRICS|{val_acc:.4f}|{val_f1:.4f}|{total_time:.2f}|{params_count}")
