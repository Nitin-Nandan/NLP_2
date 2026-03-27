import torch
import time
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from transformers import get_linear_schedule_with_warmup, PreTrainedTokenizerFast
from torch.optim import AdamW
from src.data_loader import DataLoaderFactory
from src.model_utils import CoreModelManager

def calculate_metrics(predictions, true_labels):
    acc = sum([p == t for p, t in zip(predictions, true_labels)]) / len(true_labels)
    tp = sum([p == 1 and t == 1 for p, t in zip(predictions, true_labels)])
    fp = sum([p == 1 and t == 0 for p, t in zip(predictions, true_labels)])
    fn = sum([p == 0 and t == 1 for p, t in zip(predictions, true_labels)])
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return acc, f1

def train_epoch(model, dataloader, optimizer, scheduler, device, scaler):
    model.train()
    total_loss = 0
    device_type = 'cuda' if 'cuda' in str(device) else 'cpu'
    
    for step, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=(device_type == 'cuda')):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += loss.item()
        
        if step % 200 == 0:
            print(f"Step {step}/{len(dataloader)} | Loss: {loss.item():.4f}")
            
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    predictions = []
    true_labels = []
    device_type = 'cuda' if 'cuda' in str(device) else 'cpu'
    
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=(device_type == 'cuda')):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            
    acc, f1 = calculate_metrics(predictions, true_labels)
    return acc, f1

if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    class DualLogger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "a", encoding='utf-8')
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
        def flush(self):
            self.terminal.flush()
            self.log.flush()
        def isatty(self):
            return hasattr(self.terminal, 'isatty') and self.terminal.isatty()

    sys.stdout = DualLogger(os.path.join("results", "train_output.txt"))
    print("Initiating Phase 3 BPE Training Execution via ML-Scientist")
    
    tokenizer_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'bpe_tokenizer.json')
    try:
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
        tokenizer.pad_token = "[PAD]"
        tokenizer.unk_token = "[UNK]"
        tokenizer.cls_token = "[CLS]"
        tokenizer.sep_token = "[SEP]"
        tokenizer.mask_token = "[MASK]"
    except Exception as e:
        sys.exit(1)
        
    factory = DataLoaderFactory(dataset_name="sst2")
    train_loader, val_loader = factory.get_dataloaders(tokenizer, batch_size=32, max_length=128)
    
    manager = CoreModelManager(num_labels=2, new_vocab_size=15000)
    device = manager.device
    model = manager.model
    params_count = manager.print_parameter_count()
    
    opt_params = manager.get_differential_optimizer_params(base_lr=2e-5, new_layer_lr=1e-3)
    optimizer = AdamW(opt_params, eps=1e-8)
    
    EPOCHS = 3
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps*0.1), num_training_steps=total_steps)
    scaler = torch.amp.GradScaler('cuda' if 'cuda' in str(device) else 'cpu', enabled=('cuda' in str(device)))
    
    start_time = time.time()
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, scaler)
        print(f"Epoch {epoch+1} Training Loss: {train_loss:.4f}")
        
    print("Evaluating BPE Model...")
    val_acc, val_f1 = evaluate(model, val_loader, device)
    total_time = time.time() - start_time
    print(f"BPE Training complete! Acc: {val_acc:.4f} | F1: {val_f1:.4f} | Time: {total_time:.2f}s")
    print(f"BPE_FINAL_METRICS|{val_acc:.4f}|{val_f1:.4f}|{total_time:.2f}|{params_count}")
