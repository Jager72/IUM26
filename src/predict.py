import torch
from torch.utils.data import DataLoader
from common.dataset import StarbucksDataset
from common.model import StarbucksModel

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--include-confusion-matrix", type=bool, default=False)
args = parser.parse_args()


device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

pth = torch.load("./artifacts/starbucks_model.pth")
model = StarbucksModel(pth["dropout"], pth["input_dim"]).to(device)
model.load_state_dict(pth["model_state_dict"])
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
optimizer.load_state_dict(pth["optimizer_state_dict"])

eval_ds = StarbucksDataset("./artifacts/eval.csv")
eval_dl = DataLoader(eval_ds, batch_size=64)


model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for X, y in eval_dl:
        pred = (model(X.to(device)).squeeze() > 0).float().cpu()
        all_preds.extend(pred.tolist())
        all_labels.extend(y.tolist())

from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from pathlib import Path

output_path = Path("artifacts/predictionsMetrics.txt")
output_path.parent.mkdir(parents=True, exist_ok=True)

with output_path.open("w", encoding="utf-8") as f:
    f.write(classification_report(
        all_labels,
        all_preds,
        target_names=["not completed", "completed"]
    ))
    f.write("\n")

    if args.include_confusion_matrix:
        label_map = {0.0: "not completed", 1.0: "completed"}
        all_labels_str = [label_map[l] for l in all_labels]
        all_preds_str = [label_map[p] for p in all_preds]

        cm = confusion_matrix(
            all_labels_str,
            all_preds_str,
            labels=["not completed", "completed"]
        )

        df_cm = pd.DataFrame(
            cm,
            index=["actual: not completed", "actual: completed"],
            columns=["pred: not completed", "pred: completed"]
        )

        f.write(df_cm.to_string())
        f.write("\n")


with open("./artifacts/savePred.txt", "w+") as f:
    f.write(str(all_preds))

