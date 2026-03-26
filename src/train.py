import torch
import time
import csv
import os
import sys

# Ensure src namespace resolution
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from transformers import get_linear_schedule_with_warmup, BertTokenizerFast
from torch.optim import AdamW
from src.data_loader import DataLoaderFactory
from src.model_utils import CoreModelManager

def calculate_metrics(predictions, true_labels):
    """Calculates Accuracy and F1 Score for Binary Classification without external libs."""
    acc = sum([p == t for p, t in zip(predictions, true_labels)]) / len(true_labels)
    tp = sum([p == 1 and t == 1 for p, t in zip(predictions, true_labels)])
    fp = sum([p == 1 and t == 0 for p, t in zip(predictions, true_labels)])
    fn = sum([p == 0 and t == 1 for p, t in zip(predictions, true_labels)])
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return acc, f1

def train_epoch(model, dataloader, optimizer, scheduler, device, max_steps=None):
    model.train()
    total_loss = 0
    
    for step, batch in enumerate(dataloader):
        if max_steps and step >= max_steps:
            break
            
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        if step % 10 == 0:
            print(f"Step {step} | Loss: {loss.item():.4f}")
            
    return total_loss / (step + 1)

def evaluate(model, dataloader, device, max_steps=None):
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            if max_steps and step >= max_steps:
                break
                
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            
    acc, f1 = calculate_metrics(predictions, true_labels)
    return acc, f1

if __name__ == "__main__":
    import os, sys
    os.makedirs('results', exist_ok=True)
    
    class DualLogger:
        """Route standard print output to console AND strictly to results/train_output.txt"""
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "w", encoding='utf-8')
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
        def flush(self):
            self.terminal.flush()
            self.log.flush()

    sys.stdout = DualLogger(os.path.join("results", "train_output.txt"))
    
    print("Initiating Phase 2 Baseline Execution via ML-Scientist")
    
    # Verify environment dependencies
    try:
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        sys.exit(1)
        
    factory = DataLoaderFactory(dataset_name="sst2")
    train_loader, val_loader = factory.get_dataloaders(tokenizer, batch_size=16, max_length=128)
    
    manager = CoreModelManager(num_labels=2)
    device = manager.device
    model = manager.model
    params_count = manager.print_parameter_count()
    
    # Baseline does not reset the mapping, so uniform standard learning rate is utilized.
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    
    # Limit to 50 steps so we don't violate the >2m training loop threshold (Decision Gate)
    TEST_STEPS = 50 
    print(f"Configured to run an infrastructure verification of {TEST_STEPS} steps.")
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=TEST_STEPS)
    
    start_time = time.time()
    train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, max_steps=TEST_STEPS)
    print("Evaluating baseline...")
    val_acc, val_f1 = evaluate(model, val_loader, device, max_steps=TEST_STEPS)
    
    total_time = time.time() - start_time
    print(f"Evaluation complete! Acc: {val_acc:.4f} | F1: {val_f1:.4f} | Time: {total_time:.2f}s")
    
    metrics = {
        "Model": "Baseline-WordPiece",
        "Accuracy": val_acc,
        "F1_Score": val_f1,
        "Training_Time_seconds": total_time,
        "Parameters": params_count
    }
    
    print("\n[!] Awaiting 'tracking' skill implementation to record results to CSV.")
