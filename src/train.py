import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from common.dataset import StarbucksDataset
from common.model import StarbucksModel
from common.func import train, test

import mlflow
import mlflow.pytorch
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:8080")
mlflow.set_experiment("Training Starbucks Sex")

client = MlflowClient()

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

train_ds = StarbucksDataset("./artifacts/train.csv")
eval_ds = StarbucksDataset("./artifacts/eval.csv")

batch_size = 64
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
eval_dl = DataLoader(eval_ds, batch_size=batch_size)

input_dim = train_ds.data.shape[1]
dropout = 0.2
lr = 0.0001
epochs = 30
registered_model_name = "starbucks-sex-classifier"

model = StarbucksModel(dropout, input_dim).to(device)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    patience=3,
    factor=0.5
)

os.makedirs("./artifacts", exist_ok=True)

input_example = train_ds.data[:5]
model.eval()

with torch.no_grad():
    sample_x = torch.tensor(train_ds.data.iloc[:5].to_numpy(), dtype=torch.float32).to(device)
    sample_y = model(sample_x).detach().cpu().numpy()

signature = infer_signature(input_example, sample_y)

model_card_path = "./artifacts/model_card.md"
model_card_content = f"""# Starbucks Sex Classifier
## Overview
Binary classification model trained in PyTorch and tracked with MLflow.

## Model details
- **Model class:** `StarbucksModel`
- **Input dimension:** {input_dim}
- **Dropout:** {dropout}
- **Loss:** BCEWithLogitsLoss
- **Optimizer:** Adam
- **Learning rate:** {lr}
- **Scheduler:** ReduceLROnPlateau(patience=3, factor=0.5)
- **Epochs:** {epochs}
- **Batch size:** {batch_size}

## Data
- **Train dataset:** `./artifacts/train.csv`
- **Eval dataset:** `./artifacts/eval.csv`

## Intended use
Model for binary sex classification

## Training notes
This model was trained with manual training/evaluation loops and logged to MLflow.
"""

with open(model_card_path, "w", encoding="utf-8") as f:
    f.write(model_card_content)

with mlflow.start_run() as run:
    mlflow.log_param("dropout", dropout)
    mlflow.log_param("input_dim", input_dim)
    mlflow.log_param("lr", lr)
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("optimizer", "Adam")
    mlflow.log_param("loss_fn", "BCEWithLogitsLoss")
    mlflow.log_param("scheduler", "ReduceLROnPlateau")
    mlflow.log_param("registered_model_name", registered_model_name)

    for epoch in range(epochs):
        train(train_dl, model, loss_fn, optimizer, device, epoch)
        loss = test(eval_dl, model, loss_fn, device, epoch)
        scheduler.step(loss)

        current_lr = optimizer.param_groups[0]["lr"]
        mlflow.log_metric("eval_loss", loss, step=epoch)
        mlflow.log_metric("lr", current_lr, step=epoch)

    
    model_path = "./artifacts/starbucks_model.pth"
    torch.save(model.state_dict(), model_path)
    mlflow.log_artifact(model_path)
    model_info = mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path="model",
        registered_model_name=registered_model_name,
        signature=signature,
        input_example=input_example,
        extra_files=[model_card_path],
    )

    model_versions = client.search_model_versions(f"name='{registered_model_name}'")

    current_version = None
    for mv in model_versions:
        if mv.run_id == run.info.run_id:
            current_version = mv.version
            break

    if current_version is None:
        raise RuntimeError("Could not find registered model version for this run.")

    client.update_model_version(
        name=registered_model_name,
        version=current_version,
        description=model_card_content,
    )

    with open("./artifacts/model_version.txt", "w", encoding="utf-8") as f:
        f.write(str(current_version))

    with open("./artifacts/run_id.txt", "w", encoding="utf-8") as f:
        f.write(run.info.run_id)

    print("Model URI:", model_info.model_uri)
    print("Run ID:", run.info.run_id)
    print("Registered model:", registered_model_name)
    print("Registered version:", current_version)

print("Done!")