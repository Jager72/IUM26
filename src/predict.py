import torch
from torch.utils.data import DataLoader
from common.dataset import StarbucksDataset
from common.model import StarbucksModel
from common.func import train, test

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

import collections
print("Predicted:", collections.Counter(all_preds))
print("Actual:   ", collections.Counter(all_labels))

from sklearn.metrics import classification_report
print(classification_report(all_labels, all_preds, target_names=["not completed", "completed"]))


with open("./artifacts/savePred.txt", "w+") as f:
    f.write(str(all_preds))

