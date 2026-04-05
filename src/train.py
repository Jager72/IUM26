import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from common.dataset import StarbucksDataset
from common.model import StarbucksModel
from common.func import train, test

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

train_ds = StarbucksDataset("./artifacts/train.csv")
eval_ds = StarbucksDataset("./artifacts/eval.csv")
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)
eval_dl = DataLoader(eval_ds, batch_size=64)

input_dim = train_ds.data.shape[1]
model = StarbucksModel(0.2, input_dim).to(device)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

for t in range(30):
    train(train_dl, model, loss_fn, optimizer, device)
    loss = test(eval_dl, model, loss_fn, device)
    scheduler.step(loss)

print("Done!")
torch.save({
    "model_state_dict": model.state_dict(),
    "dropout": 0.2,
    "input_dim": input_dim,
    "optimizer_state_dict": optimizer.state_dict(),
}, "./artifacts/starbucks_model.pth")