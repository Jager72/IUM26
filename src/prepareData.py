import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent


dataset = pd.read_csv(base_dir / "Data" / "profile.csv", index_col=0)
dataset = dataset.reset_index()
dataset["age"] = pd.to_numeric(dataset["age"], errors="coerce").astype(float)

#Gender
dataset["gender"] = dataset["gender"].fillna("Unknown")
dataset = pd.get_dummies(dataset, columns=["gender"])

#Age
ageMean = dataset["age"].mean(skipna=True)
ageSTD  = dataset["age"].std(skipna=True)

ageNa = dataset["age"].isna()
missingNum = ageNa.sum()
randAge = np.random.normal(loc=ageMean, scale=ageSTD, size=missingNum)
randAge = np.clip(randAge, 18, 100)
dataset.loc[ageNa, "age"] = randAge

ageMean = dataset["age"].mean()
ageSTD  = dataset["age"].std()

#z = (dataset["age"] - ageMean) / ageSTD
outliers = dataset["age"] > 100
randAge = np.random.normal(ageMean, ageSTD, outliers.sum())
randAge = np.clip(randAge, 18, 100)
dataset.loc[outliers, "age"] = randAge
dataset["age"] = MinMaxScaler().fit_transform(dataset[["age"]])

#income
dataset["income"] = dataset["income"].fillna(dataset["income"].median())
low, high = dataset["income"].quantile([0.01, 0.99])
dataset["income"] = dataset["income"].clip(low, high)
dataset["income"] = MinMaxScaler().fit_transform(dataset[["income"]])

dataset["became_member_on"] =pd.to_datetime(dataset["became_member_on"], format="%Y%m%d").astype("int64")
dataset["became_member_on"] = MinMaxScaler().fit_transform(dataset[["became_member_on"]])

#ID ZOSTAWIAM DO ŁĄCZENIA RECORDÓW W ZALEZNOŚCI OD ZADANIA!
dataset2 = pd.read_csv(base_dir / "Data" / "portfolio.csv", index_col=0)

dataset2["reward"] = MinMaxScaler().fit_transform(dataset2[["reward"]])
dataset2["difficulty"] = MinMaxScaler().fit_transform(dataset2[["difficulty"]])
dataset2["duration"] = MinMaxScaler().fit_transform(dataset2[["duration"]])

dataset2 = pd.get_dummies(dataset2, columns=["offer_type"])
channels = pd.get_dummies(dataset2["channels"].explode()).groupby(level=0).sum()
dataset2 = dataset2.drop("channels", axis=1).join(channels)

#----
dataset3 = pd.read_csv(base_dir / "Data" / "transcript.csv", index_col=0)

dataset3["time"] = MinMaxScaler().fit_transform(dataset3[["time"]])
dataset3 = pd.get_dummies(dataset3, columns=["event"])

import ast
dataset3["value"] = dataset3["value"].apply(lambda s: ast.literal_eval(s).get("offer id"))

# #-----

result = (
    dataset3
    .merge(dataset2, left_on="value", right_on="id", how="left")
    .merge(dataset, left_on="person", right_on="id", how="left")
)

result = result.drop(columns=["value"])
result.drop(["id_x","id_y"], axis=1, inplace=True)
result = result.astype({col: float for col in result.select_dtypes(include='bool').columns})
result = result.map(lambda x: float(x) if isinstance(x, bool) else x)


print(np.sum(result["event_offer completed"]))
portfolio_cols = ["reward", "difficulty", "duration", 
                  "offer_type_bogo", "offer_type_discount", 
                  "offer_type_informational",
                  "['email', 'mobile', 'social']",
                  "['web', 'email', 'mobile', 'social']",
                  "['web', 'email', 'mobile']",
                  "['web', 'email']"]

result[portfolio_cols] = result[portfolio_cols].fillna(0)
result = result.dropna()


agg = result.groupby("person").agg(
    time_mean=("time", "mean"),
    
    offers_completed=("event_offer completed", "sum"),
    offers_received=("event_offer received", "sum"),
    offers_viewed=("event_offer viewed", "sum"),
    transactions=("event_transaction", "sum"),
    
    reward_mean=("reward", "mean"),
    difficulty_mean=("difficulty", "mean"),
    duration_mean=("duration", "mean"),
    
    bogo_count=("offer_type_bogo", "sum"),
    discount_count=("offer_type_discount", "sum"),
    informational_count=("offer_type_informational", "sum"),
    
    completion_rate=("event_offer completed", "mean"),
    view_rate=("event_offer viewed", "mean"),
).reset_index()

profile_cols = ["id", "age", "income", "became_member_on",
                "gender_F", "gender_M", "gender_O", "gender_Unknown"]


result = agg.merge(
    dataset[profile_cols],
    left_on="person",
    right_on="id",
    how="left"
).drop(columns=["person", "id"])

result["completed"] = (result["offers_completed"] > 0).astype(float)
result = result.drop(columns=["offers_completed", "completion_rate"])

label_cols = ["completed"]
feature_cols = [c for c in result.columns if c not in label_cols]
result[feature_cols] = MinMaxScaler().fit_transform(result[feature_cols])
 
print("Final shape:", result.shape)
print("Columns:", result.columns.tolist())
print("Completion rate:", result["completed"].mean())
#---

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--cut-off", type=int, default=None, help="Limit number of records")
args = parser.parse_args()
if args.cut_off is not None:
    result = result.head(args.cut_off)

train, test = train_test_split(result, test_size=0.2, random_state=67)
test, eval = train_test_split(test, test_size=0.5, random_state=67)

with open("artifacts/train.csv", "w+") as f:
    train.to_csv(f, index=False)

with open("artifacts/test.csv", "w+") as f:
    test.to_csv(f, index=False)

with open("artifacts/eval.csv", "w+") as f:
    eval.to_csv(f, index=False)
 