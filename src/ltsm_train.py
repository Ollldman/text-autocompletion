from tqdm import tqdm

def train_epoch(model, loader, optimizer, criterion, device):
    """
    :return: train_loss (total_loss / len(loader))
    """
    model.train()
    total_loss = 0
    total_batches = 0
    print("Start epoch training!")
    for x_batch, y_batch in tqdm(loader, ascii=True, desc="Training!"):
        x = x_batch.to(device)
        y = y_batch.to(device)

        optimizer.zero_grad()
        logits,_ = model(x)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1
    return total_loss / total_batches if total_batches > 0 else float('inf')