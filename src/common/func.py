import torch
from tqdm import tqdm

def train(dataloader, model, loss_fn, optimizer, device):
    model.train()
    loop = tqdm(dataloader, desc="Training")
    for batch, (X,y) in enumerate(loop):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred.squeeze(), y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        

def test(dataloader, model, loss_fn, device, save_pred_dir=None):
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
    return test_loss

