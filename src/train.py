import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from common.dataset import StarbucksDataset
from common.model import StarbucksModel
from common.func import train, test

import mlflow
mlflow.set_experiment("Training Starbucks Sex")

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

train_ds = StarbucksDataset("./artifacts/train.csv")
eval_ds = StarbucksDataset("./artifacts/eval.csv")
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)
eval_dl = DataLoader(eval_ds, batch_size=64)

input_dim = train_ds.data.shape[1]
dropout = 0.2
lr = 0.0001
epochs = 30
batch_size = 64
model = StarbucksModel(dropout, input_dim).to(device)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)




with mlflow.start_run() as run:
    mlflow.log_param("dropout", dropout)
    mlflow.log_param("input_dim", input_dim)
    mlflow.log_param("lr", lr)
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("optimizer", "Adam")
    mlflow.log_param("loss_fn", "BCEWithLogitsLoss")
    mlflow.log_param("scheduler", "ReduceLROnPlateau")
    for epoch in range(epochs):
        train(train_dl, model, loss_fn, optimizer, device, epoch)
        loss = test(eval_dl, model, loss_fn, device, epoch)
        scheduler.step(loss)
        current_lr = optimizer.param_groups[0]["lr"]
        mlflow.log_metric("lr", current_lr, step=epoch)

    
    model_path = "./artifacts/starbucks_model.pth"
    model_info = mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path="model",
    )

    mlflow.log_artifact(model_path)
    mlflow.pytorch.log_model(model, "model")
    print(model_info.model_uri)
    with open("./artifacts/run_id.txt", "w", encoding="utf-8") as f:
        f.write(run.info.run_id)
print("Done!")