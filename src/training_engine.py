import torch
import sys

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

def evaluate(model, dataloader, device, dynamic_dropout=False):
    model.eval()
    predictions = []
    true_labels = []
    device_type = 'cuda' if 'cuda' in str(device) else 'cpu'
    
    original_dropout = None
    if dynamic_dropout and hasattr(dataloader.dataset.tokenizer, 'backend_tokenizer'):
        original_dropout = dataloader.dataset.tokenizer.backend_tokenizer.model.dropout
        dataloader.dataset.tokenizer.backend_tokenizer.model.dropout = None
        
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
            
    if dynamic_dropout and original_dropout is not None:
        dataloader.dataset.tokenizer.backend_tokenizer.model.dropout = original_dropout
            
    acc, f1 = calculate_metrics(predictions, true_labels)
    return acc, f1

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
