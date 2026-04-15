import torch
import mlflow
from tqdm import tqdm

def train(dataloader, model, loss_fn, optimizer, device, epoch):
    model.train()
    loop = tqdm(dataloader, desc="Training")

    total_loss = 0
    
    for batch, (X,y) in enumerate(loop):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred.squeeze(), y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    mlflow.log_metric("train_loss", avg_loss, step=epoch)
    
    return avg_loss   

def test(dataloader, model, loss_fn, device, epoch, save_pred_dir=None):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    loop = tqdm(dataloader, desc="Evaluating")
    save_pred = []
    with torch.no_grad():
        for batch, (X, y) in enumerate(loop):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred.squeeze(), y).item()
            correct += ((pred.squeeze() > 0).float() == y).float().sum().item()
            save_pred.extend((pred.squeeze() > 0).float().cpu().tolist())
    if save_pred_dir:
        with open(save_pred_dir, "w+") as f:
            f.write(str(save_pred))
    test_loss /= num_batches
    correct /= size
    print(f"Eval Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    mlflow.log_metric("test_loss", test_loss, step=epoch)
    mlflow.log_metric("accuracy", correct, step=epoch)
    return test_loss

